import threading
import time
import collections

import config
from myoraw import MyoRaw, DataCategory, EMGMode
from logger_setup import setup_logger


logger = setup_logger()


class MyoThread:
    def __init__(self, tty=config.MYO_PORT, buffer_size=config.WINDOW_SIZE * 2):
        self.tty = tty
        self.buffer_size = buffer_size
        self.emg_buffer = collections.deque(maxlen=buffer_size)
        self.buffer_lock = threading.Lock()
        self.myo = None
        self.running = False
        self.thread = None

    def connect(self):
        logger.info(f"Myoデバイスに接続中: {self.tty}")
        self.myo = MyoRaw(tty=self.tty)
        self.myo.subscribe(EMGMode.RAW)
        self.myo.set_sleep_mode(1)  # スリープ無効
        logger.info("Myo接続成功")
        self.myo.vibrate(1)

    def start(self):
        def emg_handler(timestamp, emg, moving, characteristic_num):
            with self.buffer_lock:
                self.emg_buffer.append(emg)

        self.myo.add_handler(DataCategory.EMG, emg_handler)
        self.running = True
        self.thread = threading.Thread(
            target=self._run,
            name="MyoThread",
            daemon=True
        )
        self.thread.start()
        logger.info("Myo受信スレッド起動")

    def _run(self):
        while self.running:
            try:
                self.myo.run(0.1)
            except Exception as e:
                logger.error(f"Myo受信エラー: {e}", exc_info=True)
                break

    def get_data(self, num_samples, timeout=10.0):
        start_time = time.time()
        while time.time() - start_time < timeout:
            with self.buffer_lock:
                if len(self.emg_buffer) >= num_samples:
                    data = list(self.emg_buffer)[-num_samples:]
                    return data
            time.sleep(0.05)
        logger.error(
            f"データ取得タイムアウト: "
            f"{len(self.emg_buffer)}/{num_samples}サンプル"
        )
        return None

    def clear_buffer(self):
        with self.buffer_lock:
            self.emg_buffer.clear()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        if self.myo:
            try:
                self.myo.disconnect()
                logger.info("Myo切断完了")
            except Exception as e:
                logger.error(f"Myo切断エラー: {e}", exc_info=True)


if __name__ == "__main__":
    print("MyoThread テスト")

    myo_thread = MyoThread()

    try:
        myo_thread.connect()
        myo_thread.start()

        print(f"3秒間データ受信中...")
        time.sleep(3)

        with myo_thread.buffer_lock:
            print(f"バッファサイズ: {len(myo_thread.emg_buffer)}")

        print(f"{config.WINDOW_SIZE}サンプル取得テスト...")
        myo_thread.clear_buffer()
        data = myo_thread.get_data(config.WINDOW_SIZE)

        if data:
            print(f"取得成功: {len(data)}サンプル")
            print(f"先頭データ: {data[0]}")
        else:
            print("取得失敗（タイムアウト）")

    finally:
        myo_thread.stop()
        print("\nテスト完了")
