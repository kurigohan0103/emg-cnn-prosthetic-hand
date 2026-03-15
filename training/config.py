import os
import torch


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# プロジェクトルート 
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

# データルート
DATA_ROOT = os.path.join(PROJECT_ROOT, 'data')

# 生データディレクトリ
RAW_DATA_DIR = os.path.join(DATA_ROOT, 'raw')

# CNN用データディレクトリ
PROCESSED_DATA_DIR = os.path.join(DATA_ROOT, 'processed')

# 分析結果ディレクトリ
ANALYSIS_DIR = os.path.join(DATA_ROOT, 'analysis')

# 生データファイルのパスパターン
def get_raw_data_path(subject):
    return os.path.join(RAW_DATA_DIR, f'Sub{subject}_*', '*_emg.csv')

# タイミングファイルのパス
def get_timing_path(subject):
    return os.path.join(PROCESSED_DATA_DIR, f'Sub{subject}_*', '*_timing.csv')

# CNN用データファイルのパスパターン
def get_processed_data_path(subject):
    return os.path.join(PROCESSED_DATA_DIR, f'Sub{subject}_*', 'CNN', '*_emg.csv')


# 実験設定
SEED = 123
BATCH_SIZE = 8
EPOCHS = 100
LEARNING_RATE = 0.001
TRAIN_RATIO = 0.75


# 被験者設定
AVAILABLE_SUBJECTS = [0, 1, 2, 3]
SUBJECTS = [2]  # デフォルト設定


# データ設定
WINDOW_SIZE = 434
TRIM_DURATION = 0.3
TRIM_MODE = 'front'
FILTER_ENABLED = True
LOWCUT = 20.0
HIGHCUT = 80.0
SAMPLING_RATE = 200


# モデル設定
NUM_CLASSES = 6
TASKS = [0, 1, 2, 3, 4, 5]


# デバイス設定
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 層化分割設定
STRATIFY_BY_SUBJECT = True
STRATIFY_BY_TASK = True


# モデル保存設定
MODELS_DIR = os.path.join(CURRENT_DIR, 'models', 'saved_models')
DEFAULT_MODEL_NAME = 'model_weight.pth'
BEST_MODEL_NAME = 'best_model.pth'
FINAL_MODEL_NAME = 'final_model.pth'

# チェックポイント設定
SAVE_CHECKPOINT_EVERY = None
KEEP_LAST_N_CHECKPOINTS = 3

# 早期終了設定
EARLY_STOPPING = False
EARLY_STOPPING_PATIENCE = 20


# パス確認
if __name__ == '__main__':
    print("Configuration Check")
    print(f"Current directory: {CURRENT_DIR}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Data root: {DATA_ROOT}")
    print(f"Raw data directory: {RAW_DATA_DIR}")
    print(f"Processed data directory: {PROCESSED_DATA_DIR}")
    print(f"Analysis directory: {ANALYSIS_DIR}")
    print(f"Models directory: {MODELS_DIR}")
    print(f"\nExample paths:")
    print(f"  Raw data: {get_raw_data_path(2)}")
    print(f"  Timing: {get_timing_path(2)}")
    print(f"  Processed: {get_processed_data_path(2)}")
    
    # ディレクトリの存在確認
    print("\nDirectory existence check:")
    for name, path in [
        ('Raw data', RAW_DATA_DIR),
        ('Processed data', PROCESSED_DATA_DIR),
        ('Analysis', ANALYSIS_DIR),
        ('Models', MODELS_DIR)
    ]:
        exists = os.path.exists(path)
        status = "Exists" if exists else "Not found"
        print(f"  {name:20} {status:15} {path}")
    
    print(f"\nDevice: {DEVICE}")
    print(f"Subjects: {SUBJECTS}")
    print(f"Window size: {WINDOW_SIZE}")