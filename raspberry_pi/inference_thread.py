import threading
import time
from logger_setup import setup_logger

logger = setup_logger()

class InferenceThread:
    def __init__(self, inference_engine, error_flag, inference_done, control_done, result_lock, latest_result):
        """
        Args:
            inference_engine: InferenceEngineインスタンス
            error_flag: threading.Event (エラーフラグ)
            inference_done: threading.Event (推論完了通知)
            control_done: threading.Event (制御完了通知)
            result_lock: threading.Lock (結果共有用ロック)
            latest_result: dict (結果格納用、{'value': None})
        """
        self.engine = inference_engine
        self.error_flag = error_flag
        self.inference_done = inference_done
        self.control_done = control_done
        self.result_lock = result_lock
        self.latest_result = latest_result
        
        self.thread = None
    
    def start(self):
        self.thread = threading.Thread(
            target=self._run,
            name="InferenceThread",
            daemon=True
        )
        self.thread.start()
        logger.info("推論スレッド起動")
    
    def _run(self):
        logger.debug("推論ループ開始")
        
        while not self.error_flag.is_set():
            # 制御完了を待つ
            self.control_done.wait()
            self.control_done.clear()
            
            if self.error_flag.is_set():
                break
            
            # 推論実行
            logger.debug("推論開始")
            result = self.engine.run_inference()
            
            if result is None:
                logger.error("推論失敗")
                self.error_flag.set()
                break
            
            # 結果を共有変数に格納
            with self.result_lock:
                self.latest_result['value'] = result
            
            logger.info(f"推論完了: {result}")
            
            # 制御スレッドに通知
            self.inference_done.set()
        
        logger.debug("推論ループ終了")
    
    def join(self, timeout=2):
        if self.thread:
            self.thread.join(timeout)