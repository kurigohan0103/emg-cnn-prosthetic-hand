import threading

from logger_setup import setup_logger


logger = setup_logger()

class ControlThread:
    def __init__(self, hand_controller, led_ctrl, error_flag, inference_done, control_done, result_lock, latest_result):
        """
        Args:
            hand_controller: MotorControllerインスタンス
            led_ctrl: LEDControllerインスタンス
            error_flag: threading.Event
            inference_done: threading.Event
            control_done: threading.Event
            result_lock: threading.Lock
            latest_result: dict
        """
        self.motor = hand_controller
        self.led = led_ctrl
        self.error_flag = error_flag
        self.inference_done = inference_done
        self.control_done = control_done
        self.result_lock = result_lock
        self.latest_result = latest_result
        
        self.thread = None
    
    def start(self):
        self.thread = threading.Thread(
            target=self._run,
            name="ControlThread",
            daemon=True
        )
        self.thread.start()
        logger.info("制御スレッド起動")
    
    def _run(self):
        logger.debug("制御ループ開始")
        
        while not self.error_flag.is_set():
            # 推論完了を待つ
            self.inference_done.wait()
            self.inference_done.clear()
            
            if self.error_flag.is_set():
                break
            
            # 結果を取得
            with self.result_lock:
                result = self.latest_result['value']
            
            # 結果の妥当性チェック
            if result < 1 or result > 6:
                logger.error(f"不正な結果値: {result}")
                self.error_flag.set()
                break
            
            logger.info(f"制御開始: Task{result}")
            
            # モーターとLEDを並行実行
            motor_thread = threading.Thread(
                target=self._execute_motor_safe,
                args=(result,)
            )
            led_thread = threading.Thread(
                target=self.led.blink_task_led,
                args=(result,)
            )
            
            motor_thread.start()
            led_thread.start()
            
            # 両方の完了を待つ
            motor_thread.join()
            led_thread.join()
            
            logger.info(f"制御完了: Task{result}")
            
            # 推論スレッドに通知
            self.control_done.set()
        
        logger.debug("制御ループ終了")
    
    def _execute_motor_safe(self, task_id):
        try:
            success = self.motor.execute_task(task_id)
            
            # execute_task()はboolを返す（タイムアウト時はFalse）
            if not success:
                logger.error(f"モーター制御失敗: Task{task_id}")
                self.error_flag.set()
                
        except Exception as e:
            logger.error(f"モーター制御エラー: {e}", exc_info=True)
            self.error_flag.set()

    def join(self, timeout=2):
        if self.thread:
            self.thread.join(timeout)