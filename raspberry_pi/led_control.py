import time
import threading
from gpiozero import LED

import config
from logger_setup import setup_logger


logger = setup_logger()

class LEDController:
    def __init__(self):
        self.led_status = LED(config.LED_STATUS)
        self.led_task_1_3 = LED(config.LED_TASK_1_3)
        self.led_task_4_6 = LED(config.LED_TASK_4_6)
        
        self.error_blink_thread = None # エラー点滅用スレッド
        self.stop_blink = threading.Event()
        
        logger.info("LED Controller Initialization Complete")
    
    def status_on(self):
        self.stop_blink.set() # エラー点滅を停止
        self.led_status.on()
        logger.debug("LED1: Always ON")
    
    def status_error_blink(self):
        self.stop_blink.clear()
        self.error_blink_thread = threading.Thread(
            target=self._blink_error,
            daemon=True # メインプログラムが終了したらこのスレッドも強制終了
        )
        self.error_blink_thread.start()
        logger.warning("LED1: Start Error Flashing.")
    
    def _blink_error(self):
        while not self.stop_blink.is_set(): # Falseの間ループ
            self.led_status.on()
            time.sleep(config.ERROR_BLINK_INTERVAL)
            self.led_status.off()
            time.sleep(config.ERROR_BLINK_INTERVAL)
    
    def blink_task_led(self, result):
        if result < 1 or result > 6:
            logger.error(f"不正な推論結果: {result}")
            return
            
        if result in [1, 2, 3]:
            led = self.led_task_1_3
            count = result
            led_name = "LED2"
        else:  # 4, 5, 6
            led = self.led_task_4_6
            count = result - 3
            led_name = "LED3"
        
        logger.info(f"{led_name}: {count}回点滅開始")
        
        for i in range(count):
            led.on()
            time.sleep(config.LED_BLINK_ON_TIME)
            led.off()
            time.sleep(config.LED_BLINK_OFF_TIME)
        
        logger.debug(f"{led_name}: 点滅完了")
    
    def all_off(self):
        self.stop_blink.set()
        self.led_status.off()
        self.led_task_1_3.off()
        self.led_task_4_6.off()
        logger.debug("全LED消灯")
    
    def cleanup(self):
        self.all_off()
        logger.info("LEDコントローラ終了")


if __name__ == "__main__":
    led_ctrl = LEDController()

    # ステータスLED点灯テスト
    print("LED1点灯テスト")
    led_ctrl.status_on()
    time.sleep(2)

    # Task LED点滅テスト（結果=3）
    print("LED2を3回点滅")
    led_ctrl.blink_task_led(3)
    time.sleep(2)

    # エラー点滅テスト
    print("LED1エラー点滅")
    led_ctrl.status_error_blink()
    time.sleep(5)

    led_ctrl.cleanup()