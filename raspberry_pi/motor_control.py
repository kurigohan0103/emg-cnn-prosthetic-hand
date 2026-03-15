import os
import time
import spidev
from gpiozero import PWMOutputDevice, DigitalOutputDevice

import config
from logger_setup import setup_logger

logger = setup_logger()


class Potentiometer:
    def __init__(self, channel=0):
        self.channel = channel
        self.spi = spidev.SpiDev()

        try:
            self.spi.open(0, 0)  # SPI0, CE0
            self.spi.max_speed_hz = 1000000  # 1MHz
            logger.info(f"ポテンショメーター初期化完了: CH{channel}")
        except Exception as e:
            logger.error(f"SPI初期化エラー: {e}", exc_info=True)
            raise Exception("ポテンショメーターの初期化に失敗しました")

    # ビット演算
    def read(self):
        cmd = 0x68 | (self.channel << 4)
        resp = self.spi.xfer2([cmd, 0x00])
        value = ((resp[0] & 0x03) << 8) + resp[1]
        return value

    def cleanup(self):
        self.spi.close()
        logger.debug("ポテンショメーター クリーンアップ完了")


class MotorDriver:
    def __init__(self, pwm_pin, dir_pin):
        # frequency=1000 は RPi.GPIO の時と同じ設定です
        self.pwm = PWMOutputDevice(pwm_pin, frequency=1000)
        self.dir = DigitalOutputDevice(dir_pin)

        logger.info(f"モータードライバー初期化(gpiozero): PWM={pwm_pin}, DIR={dir_pin}")

    def move(self, speed, direction):
        # 速度リミッター (0-100)
        safe_speed = max(0, min(100, speed))

        # gpiozero用に変換 (0-100 -> 0.0-1.0)
        value = safe_speed / 100.0

        # 方向制御
        if direction > 0:
            self.dir.on()   # HIGH
        else:
            self.dir.off()  # LOW

        # PWM出力
        self.pwm.value = value

    def stop(self):
        self.pwm.value = 0

    def cleanup(self):
        self.pwm.close()
        self.dir.close()
        logger.debug("モータードライバー クリーンアップ完了")


class HandController:
    def __init__(self):
        logger.info("HandController 初期化開始")

        # モーターとポテンショメーターを初期化
        self.motor = MotorDriver(config.MOTOR_PWM_PIN, config.MOTOR_DIR_PIN)
        self.pot = Potentiometer(config.POT_CHANNEL)

        # フィルタリング用の基準値を初期化
        self.last_valid_value = self.pot.read()

        # 閾値設定
        self.THRESHOLD_REST = getattr(config, 'THRESHOLD_REST', 500)
        self.THRESHOLD_LIFT = getattr(config, 'THRESHOLD_LIFT', 680)

        # 速度設定
        self.SPEED_FAST = config.SPEED_FAST
        self.SPEED_SLOW = config.SPEED_SLOW

        # タイムアウト時間
        self.TIMEOUT = config.MOTOR_TIMEOUT

        self.PWM_1KG = getattr(config, 'PWM_1KG', 50)  # 1kgを持ち上げるのに必要なPWM (0-100)
        self.PWM_MAX = 100  # PWM最大値
        self.PWM_STEP = 5   # PWM増加ステップ
        self.STEP_INTERVAL = 0.1  # PWM増加間隔（秒）
        self.MOVE_THRESHOLD_RATIO = 0.1  # 動き検出閾値（移動量の10%）

        logger.info("HandController 初期化完了")
        logger.info(f"停止閾値設定: REST < {self.THRESHOLD_REST}, LIFT > {self.THRESHOLD_LIFT}")
        logger.info(f"適応制御設定: PWM_1KG={self.PWM_1KG}, STEP={self.PWM_STEP}, INTERVAL={self.STEP_INTERVAL}s")
        logger.info(f"現在位置: {self.pot.read()}")

    def get_filtered_value(self):
        raw_val = self.pot.read()
        diff = abs(raw_val - self.last_valid_value)

        if diff < 50:
            self.last_valid_value = raw_val
            return raw_val
        else:
            return self.last_valid_value

    def _move_until_condition(self, target_type, speed):
        logger.info(f"移動開始: {target_type}へ (速度={speed})")
        start_time = time.time()

        # 移動開始直前に基準値をリセット
        self.last_valid_value = self.pot.read()

        while True:
            current = self.get_filtered_value()

            # 停止判定
            if target_type == 'LIFT':
                if current > self.THRESHOLD_LIFT:
                    self.motor.stop()
                    logger.info(f"LIFT到達完了: 値={current}")
                    return True
                self.motor.move(speed, 1) # 正転

            elif target_type == 'REST':
                if current < self.THRESHOLD_REST:
                    self.motor.stop()
                    logger.info(f"REST到達完了: 値={current}")
                    return True
                self.motor.move(speed, -1) # 逆転

            # タイムアウト
            if time.time() - start_time > self.TIMEOUT:
                self.motor.stop()
                logger.error(f"移動タイムアウト: {time.time() - start_time:.1f}秒経過")
                return False

            # 高速チェック
            time.sleep(0.01)

    def _adaptive_lift(self):
        logger.info("adaptive_lift: 1kgのPWMから探索開始")

        # 開始位置を取得
        start_pos = self.pot.read()
        self.last_valid_value = start_pos

        # 動き検出閾値を計算（移動量の10%）
        move_range = self.THRESHOLD_LIFT - start_pos
        move_threshold = int(move_range * self.MOVE_THRESHOLD_RATIO)
        threshold_pos = start_pos + move_threshold

        logger.info(f"開始位置: {start_pos}")
        logger.info(f"動き検出閾値: {threshold_pos} (開始+{move_threshold})")
        logger.info(f"目標位置: {self.THRESHOLD_LIFT}")
        logger.info(f"開始PWM: {self.PWM_1KG}")

        current_pwm = self.PWM_1KG
        found_pwm = None
        start_time = time.time()

        while current_pwm <= self.PWM_MAX:
            self.motor.move(current_pwm, 1)  # 正転
            time.sleep(self.STEP_INTERVAL)

            current_pos = self.get_filtered_value()
            logger.debug(f"PWM={current_pwm}, 位置={current_pos}")

            # 動き検出
            if current_pos >= threshold_pos:
                found_pwm = current_pwm
                logger.info(f"★ 動き検出! PWM={current_pwm}, 位置={current_pos}")
                break

            # タイムアウトチェック
            if time.time() - start_time > self.TIMEOUT:
                self.motor.stop()
                logger.error("フェーズ1 タイムアウト")
                return False

            # PWMを上げる
            current_pwm += self.PWM_STEP
            logger.debug(f"PWM上昇: {current_pwm}")

        # PWM上限到達チェック
        if found_pwm is None:
            self.motor.stop()
            logger.error(f"PWM上限({self.PWM_MAX})到達: 持ち上げられません")
            return False

        logger.info(f"固定PWM={found_pwm}で移動継続")

        phase2_start = time.time()

        while True:
            current_pos = self.get_filtered_value()

            # 目標到達チェック
            if current_pos >= self.THRESHOLD_LIFT:
                self.motor.stop()
                logger.info(f"★ 目標到達! 位置={current_pos}")
                logger.info(f"使用PWM={found_pwm}")
                return True

            # タイムアウトチェック
            if time.time() - phase2_start > self.TIMEOUT:
                self.motor.stop()
                logger.error("フェーズ2 タイムアウト")
                return False

            time.sleep(0.01)

    def execute_task(self, task_id):
        logger.info(f"タスク受信: Task{task_id}")

        if task_id == 1:
            # Task 1: 高速でLIFT方向へ
            logger.info("Task 1: [FAST] LIFT方向へ移動")
            return self._move_until_condition('LIFT', self.SPEED_FAST)

        elif task_id == 2:
            # Task 2: 低速でLIFT方向へ
            logger.info("Task 2: [SLOW] LIFT方向へ移動")
            return self._move_until_condition('LIFT', self.SPEED_SLOW)

        elif task_id == 3:
            # Task 3: 高速でREST方向へ
            logger.info("Task 3: [FAST] REST方向へ移動")
            return self._move_until_condition('REST', self.SPEED_FAST)

        elif task_id == 4:
            # Task 4: 適応的PWM制御（パターン①）
            logger.info("Task 4: [ADAPTIVE] パターン① - 1kgのPWMから探索")
            return self._adaptive_lift()

        elif task_id == 5:
            # Task 5: 適応的PWM制御（パターン①）
            logger.info("Task 5: [ADAPTIVE] パターン① - 1kgのPWMから探索")
            return self._adaptive_lift()

        elif task_id == 6:
            # Task 6: 高速でREST方向へ
            logger.info("Task 6: [FAST] REST方向へ移動")
            return self._move_until_condition('REST', self.SPEED_FAST)

        else:
            logger.error(f"未定義のタスク番号: {task_id}")
            return False

    def get_position(self):
        return self.pot.read()

    def move_to_rest(self):
        logger.info("REST方向へ移動")
        return self._move_until_condition('REST', self.SPEED_FAST)

    def move_to_lift(self):
        logger.info("LIFT方向へ移動")
        return self._move_until_condition('LIFT', self.SPEED_FAST)

    def stop(self):
        self.motor.stop()
        logger.debug("モーター停止")

    def cleanup(self):
        logger.info("HandController クリーンアップ開始")
        self.motor.stop()
        self.motor.cleanup()
        self.pot.cleanup()
        logger.info("HandController クリーンアップ完了")


if __name__ == "__main__":
    print("HandController テスト")

    controller = None

    try:
        controller = HandController()

        print(f"\n現在位置: {controller.get_position()}")
        print(f"REST閾値: {controller.THRESHOLD_REST}")
        print(f"LIFT閾値: {controller.THRESHOLD_LIFT}")
        print(f"1kgのPWM: {controller.PWM_1KG}")
        print("テストコマンド:")
        print("  1: [FAST] LIFT方向へ")
        print("  2: [SLOW] LIFT方向へ")
        print("  3: [FAST] REST方向へ")
        print("  6: [FAST] REST方向へ")
        print("  7: パターン①を直接実行")
        print("  r: REST位置へ")
        print("  l: LIFT位置へ")
        print("  p: 現在位置表示")
        print("  s: 設定値表示")
        print("  q: 終了")

        while True:
            user_input = input("\nコマンド >> ").strip().lower()

            if user_input == 'q':
                break

            elif user_input == 'r':
                controller.move_to_rest()

            elif user_input == 'l':
                controller.move_to_lift()

            elif user_input == 'p':
                print(f"現在位置: {controller.get_position()}")

            elif user_input == 's':
                print(f"REST閾値: {controller.THRESHOLD_REST}")
                print(f"LIFT閾値: {controller.THRESHOLD_LIFT}")
                print(f"1kgのPWM: {controller.PWM_1KG}")
                print(f"PWM最大値: {controller.PWM_MAX}")
                print(f"PWMステップ: {controller.PWM_STEP}")
                print(f"ステップ間隔: {controller.STEP_INTERVAL}秒")
                print(f"動き検出閾値: {controller.MOVE_THRESHOLD_RATIO*100}%")
                print(f"タイムアウト: {controller.TIMEOUT}秒")

            elif user_input == '7':
                result = controller._adaptive_lift()
                print(f"結果: {'成功' if result else '失敗'}")

            elif user_input.isdigit():
                task_num = int(user_input)
                if 1 <= task_num <= 6:
                    controller.execute_task(task_num)
                else:
                    print("タスク番号は1-6で入力してください")

            else:
                print("無効なコマンドです")

    except KeyboardInterrupt:
        print("\n\n強制終了")

    except Exception as e:
        print(f"\nエラー: {e}")

    finally:
        if controller:
            controller.cleanup()
        print("\nテスト終了")
