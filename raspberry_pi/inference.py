import numpy as np
import time
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import butter, filtfilt, iirnotch

import config
from logger_setup import setup_logger


logger = setup_logger()

class EMGNet(nn.Module):
    def __init__(self, sig_size, num_classes):
        super().__init__()
        self.sig_size = sig_size
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(1, 16, (64, 1))
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 3, (1, 8), padding='same')
        self.ln = nn.BatchNorm2d(3)
        self.fc1 = nn.Linear(self._get_fc_input_size(), 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
    
    def _get_fc_input_size(self):
        with torch.no_grad():
            x = torch.zeros(self.sig_size)
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = torch.flatten(x, 1)
        return x.shape[1]
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.ln(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def bandpass_filter(data, lowcut, highcut, fs, order=4):
    w0 = 50 / (fs / 2)
    b, a = iirnotch(w0, 30)
    data = filtfilt(b, a, data)
    
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

class InferenceEngine:
    def __init__(self, myo_thread):
        # デバイス設定
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        logger.info(f"使用デバイス: {self.device}")

        self.fs = config.SAMPLING_FREQ
        self.window_size = config.WINDOW_SIZE
        self.lowcut = config.LOWCUT
        self.highcut = config.HIGHCUT
        self.num_channels = 8
        self.num_tasks = config.NUM_TASKS
        self.use_confidence_threshold = config.USE_CONFIDENCE_THRESHOLD
        self.confidence_threshold = config.CONFIDENCE_THRESHOLD
        self.max_retries = config.MAX_RETRIES
        self.retry_interval = config.RETRY_INTERVAL

        # Myoスレッド
        self.myo_thread = myo_thread

        if self.use_confidence_threshold:
            logger.info(
                f"信頼度チェック: 有効 "
                f"(閾値={self.confidence_threshold}, "
                f"最大試行={self.max_retries}回)"
            )
        else:
            logger.info("信頼度チェック: 無効")

        # モデル読み込み
        self.model = None
        self.load_model()
    
    def load_model(self):
        try:
            logger.info(f"モデル読み込み開始: {config.MODEL_PATH}")
            
            # モデル構造を作成: SigSize: (1, 1, 990, 8) の形状
            sig_size = (1, 1, self.window_size, self.num_channels)
            self.model = EMGNet(sig_size, self.num_tasks)
            self.model = self.model.to(self.device)
            
            # 重みを読み込み
            self.model.load_state_dict(torch.load(config.MODEL_PATH, map_location=self.device))
            
            # 評価モード
            self.model.eval()
            
            logger.info("モデル読み込み成功")
            logger.debug(f"モデル構造: {self.model}")
            
            return True
            
        except Exception as e:
            logger.error(f"モデル読み込みエラー: {e}", exc_info=True)
            return False
    
    def _get_emg_data(self):
        logger.debug(f"{self.window_size}サンプルのデータ収集開始")

        self.myo_thread.clear_buffer()
        time.sleep(0.3)
        data = self.myo_thread.get_data(self.window_size)

        if data is None:
            return None

        logger.debug(f"データ収集完了: {len(data)}サンプル")
        return np.array(data)  # shape: (990, 8)
    
    def _apply_filters(self, data):
        logger.debug("フィルタリング開始")
        
        filtered_data = np.zeros_like(data, dtype=np.float32)
        
        # チャンネルごとにフィルタ適用
        for ch in range(data.shape[1]):
            filtered_data[:, ch] = bandpass_filter(
                data[:, ch],
                self.lowcut,
                self.highcut,
                self.fs
            )
        
        logger.debug("フィルタリング完了")
        return filtered_data
    
    def _preprocess(self, raw_data):
        logger.debug("前処理開始")
        
        # フィルタリング
        filtered = self._apply_filters(raw_data)
        
        # Tensor化 + 次元追加
        # (990, 8) → (1, 1, 990, 8)
        tensor = torch.Tensor(filtered)
        tensor = tensor.unsqueeze(0).unsqueeze(0)  # batch, channel次元追加
        tensor = tensor.to(self.device)
        
        logger.debug(f"前処理完了: 出力形状={tensor.shape}")
        
        return tensor
    
    def run_inference_single(self):
        try:
            logger.debug("推論開始")
            
            # データ取得
            raw_data = self._get_emg_data()
            
            if raw_data is None:
                logger.error("データ取得失敗")
                return {
                    'task': None,
                    'confidence': 0.0,
                    'success': False
                }
            
            # データ形状チェック
            if raw_data.shape != (self.window_size, self.num_channels):
                logger.error(
                    f"データ形状エラー: {raw_data.shape} != "
                    f"({self.window_size}, {self.num_channels})"
                )
                return {
                    'task': None,
                    'confidence': 0.0,
                    'success': False
                }
            
            # 前処理
            input_data = self._preprocess(raw_data)
            
            # 推論実行
            with torch.no_grad():
                output = self.model(input_data)
            
            # 結果取得
            # output shape: (1, num_tasks)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_class].item()
            
            # クラス変換（0-indexedを1-indexedに）
            predicted_task = predicted_class + 1
            
            # 範囲チェック
            if predicted_task < 1 or predicted_task > self.num_tasks:
                logger.error(
                    f"推論結果が範囲外: {predicted_task} "
                    f"(期待範囲: 1-{self.num_tasks})"
                )
                return {
                    'task': None,
                    'confidence': 0.0,
                    'success': False
                }
            
            logger.debug(
                f"推論完了: Task{predicted_task}, "
                f"信頼度: {confidence:.3f}"
            )
            return {
                'task': predicted_task,
                'confidence': confidence,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"推論エラー: {e}", exc_info=True)
            return {
                'task': None,
                'confidence': 0.0,
                'success': False
            }
    
    def run_inference(self):
        # 信頼度チェックが無効の場合
        if not self.use_confidence_threshold:
            logger.info("")
            logger.info("推論開始（信頼度チェック無効モード）")
            logger.info("")
            
            result = self.run_inference_single()
            
            if result['success']:
                logger.info(
                    f"推論完了: Task{result['task']}, "
                    f"信頼度: {result['confidence']:.3f} "
                    f"(チェックなし)"
                )
                logger.info("")
                return result['task']
            else:
                logger.error("推論失敗")
                logger.info("")
                return None
        
        # 信頼度チェックが有効の場合
        logger.info("")
        logger.info(
            f"推論開始（最大{self.max_retries}回試行、"
            f"閾値: {self.confidence_threshold}）"
        )
        logger.info("")
        
        for attempt in range(self.max_retries):
            logger.info(f"\n--- 試行 {attempt + 1}/{self.max_retries} ---")
            
            result = self.run_inference_single()
            
            if not result['success']:
                logger.warning(f"試行{attempt + 1}: 推論失敗")
                
                if attempt < self.max_retries - 1:
                    logger.info(f"{self.retry_interval}秒後に再試行します...")
                    time.sleep(self.retry_interval)
                continue
            
            task = result['task']
            confidence = result['confidence']
            
            # 信頼度チェック
            if confidence >= self.confidence_threshold:
                logger.info(
                    f"成功！(試行{attempt + 1}): "
                    f"Task{task}, 信頼度: {confidence:.3f} "
                    f"(閾値: {self.confidence_threshold})"
                )
                logger.info("")
                return task
            else:
                logger.warning(
                    f"信頼度不足（試行{attempt + 1}): "
                    f"{confidence:.3f} < {self.confidence_threshold}"
                )
                
                if attempt < self.max_retries - 1:
                    logger.info(f"{self.retry_interval}秒後に再試行します...")
                    time.sleep(self.retry_interval)
        
        logger.error(
            f"\n失敗: {self.max_retries}回試行しましたが、"
            f"信頼度が閾値（{self.confidence_threshold}）を超えませんでした"
        )
        logger.info("")
        return None

    def cleanup(self):
        logger.info("InferenceEngine クリーンアップ完了")


if __name__ == "__main__":
    from myo_thread import MyoThread

    print("InferenceEngine テスト")

    myo_thread = MyoThread()
    myo_thread.connect()
    myo_thread.start()

    engine = InferenceEngine(myo_thread)

    try:
        # 推論テスト（3回）
        for i in range(3):
            print(f"\n--- 推論 {i+1} ---")
            result = engine.run_inference()

            if result:
                print(f"結果: Task {result}")
            else:
                print("推論失敗")

            time.sleep(1)

    finally:
        myo_thread.stop()
        engine.cleanup()
        print("\nテスト完了")
