from gpiozero import Button
import threading
import time
import signal
import sys

import config
from logger_setup import setup_logger
from led_control import LEDController
from motor_control import HandController
from inference import InferenceEngine
from inference_thread import InferenceThread
from control_thread import ControlThread
from myo_thread import MyoThread


logger = setup_logger()

class ProstheticHandSystem:
    def __init__(self):
        logger.info("義手制御システム起動")
        
        # 状態管理
        self.system_running = False
        self.system_lock = threading.Lock()
        self.last_press_time = 0
        
        # イベントフラグ
        self.error_flag = threading.Event()
        self.inference_done = threading.Event()
        self.control_done = threading.Event()
        self.control_done.set()  # 初期状態はTrue
        
        # 共有変数
        self.result_lock = threading.Lock()
        self.latest_result = {'value': None}
        
        # コントローラ
        self.led_ctrl = None
        self.hand_controller = None
        self.inference_engine = None

        # スレッド
        self.myo_thread = None
        self.inference_thread = None
        self.control_thread = None
        
        # ボタン
        self.start_button = Button(config.BUTTON_START, pull_up=True, bounce_time=config.DEBOUNCE_TIME)
        self.stop_button = Button(config.BUTTON_STOP, pull_up=True, bounce_time=config.DEBOUNCE_TIME)
        
        # ボタンイベント設定
        self.start_button.when_pressed = self.on_start_button
        self.stop_button.when_pressed = self.on_stop_button
        
        logger.info("初期化完了")
    
    def on_start_button(self):
        # デバウンスチェック
        current_time = time.time()
        if current_time - self.last_press_time < config.MIN_PRESS_INTERVAL:
            logger.debug("デバウンス: ボタン押下を無視")
            return
        self.last_press_time = current_time
        
        # 状態チェック
        with self.system_lock:
            if self.system_running:
                logger.info("既にシステム稼働中")
                return
            self.system_running = True
        
        logger.info("起動ボタン押下検出")
        self.start_system()
    
    def start_system(self):
        try:
            logger.info("システム起動開始")
            
            # Myo受信スレッド起動
            self.myo_thread = MyoThread()
            self.myo_thread.connect()
            self.myo_thread.start()

            # コントローラ初期化
            self.led_ctrl = LEDController()
            self.hand_controller = HandController()
            self.inference_engine = InferenceEngine(self.myo_thread)
            
            # モデル読み込みチェック
            if not self.inference_engine.model:
                raise Exception("モデル読み込み失敗")
            
            # LED1点灯
            self.led_ctrl.status_on()
            
            # フラグリセット
            self.error_flag.clear()
            self.inference_done.clear()
            self.control_done.set() # True
            
            # スレッド起動
            self.inference_thread = InferenceThread(
                self.inference_engine,
                self.error_flag,
                self.inference_done,
                self.control_done,
                self.result_lock,
                self.latest_result
            )
            
            self.control_thread = ControlThread(
                self.hand_controller,
                self.led_ctrl,
                self.error_flag,
                self.inference_done,
                self.control_done,
                self.result_lock,
                self.latest_result
            )
            
            self.inference_thread.start()
            self.control_thread.start()
            
            # エラー監視開始
            threading.Thread(target=self.monitor_error, daemon=True).start()
            
            logger.info("システム起動完了")
            
        except Exception as e:
            logger.error(f"起動エラー: {e}", exc_info=True)
            self.handle_error()
    
    def monitor_error(self):
        self.error_flag.wait()
        logger.warning("エラー検出")
        self.handle_error()
    
    def handle_error(self):
        logger.warning("エラーハンドリング開始")
        
        # スレッド停止
        if self.inference_thread: # Trueの場合（起動済み）
            self.inference_thread.join(timeout=2)
        if self.control_thread:
            self.control_thread.join(timeout=2)
        
        if self.hand_controller:
            self.hand_controller.stop()
        
        # LED2,3消灯、LED1点滅
        if self.led_ctrl:
            self.led_ctrl.led_task_1_3.off()
            self.led_ctrl.led_task_4_6.off()
            self.led_ctrl.status_error_blink()
        
        # 状態リセット
        with self.system_lock:
            self.system_running = False
        
        logger.warning("エラーハンドリング完了（リセット待機）")
    
    def on_stop_button(self):
        # エラー時は即座にリセット
        if self.error_flag.is_set():
            logger.info("エラーリセット")
            self.reset_system()
            return
        
        # 稼働中は長押しチェック
        if self.system_running:
            logger.debug("長押しチェック開始")
            time.sleep(config.LONG_PRESS_TIME)
            
            if self.stop_button.is_pressed:
                logger.info("長押し検出: システム停止")
                self.stop_system()
    
    def stop_system(self):
        logger.info("システム停止開始")
        
        # エラーフラグを立ててスレッド停止
        self.error_flag.set()
        
        # スレッド終了待機
        if self.inference_thread:
            self.inference_thread.join(timeout=2)
        if self.control_thread:
            self.control_thread.join(timeout=2)
        
        # ハードウェア停止
        if self.hand_controller:
            self.hand_controller.cleanup()
        if self.led_ctrl:
            self.led_ctrl.all_off()
        if self.myo_thread:
            self.myo_thread.stop()

        # 状態リセット
        with self.system_lock:
            self.system_running = False

        self.error_flag.clear()

        logger.info("システム停止完了")
    
    def reset_system(self):
        logger.info("システムリセット")
        
        # LED停止
        if self.led_ctrl:
            self.led_ctrl.all_off()
        
        # フラグクリア
        self.error_flag.clear()
        
        # 状態リセット
        with self.system_lock:
            self.system_running = False
        
        logger.info("待機状態に戻りました")
    
    def cleanup(self):
        logger.info("システム終了処理開始")
        
        if self.system_running:
            self.stop_system()
        
        logger.info("義手制御システム終了")


def signal_handler(signum, frame):
    logger.info("\n終了シグナル受信")
    system.cleanup()
    sys.exit(0)


if __name__ == "__main__":
    # シグナルハンドラ登録
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # システム起動
    system = ProstheticHandSystem()
    
    logger.info("起動ボタンを押してください")
    logger.info("停止: ボタン2を2秒長押し")
    logger.info("Ctrl+Cで終了")
    
    # メインスレッドは待機
    signal.pause()