import os
import glob
from config import PROCESSED_DATA_DIR, get_processed_data_path, get_timing_path


# データ構造をチェック
def check_data_structure():
    print("\nData Structure Check")
    
    # データディレクトリの確認
    print(f"\nData Directory: {PROCESSED_DATA_DIR}")
    
    if not os.path.exists(PROCESSED_DATA_DIR):
        print("Data directory does not exist!")
        print(f"   Please create: {PROCESSED_DATA_DIR}")
        return
    
    print("Data directory exists")
    
    # 被験者ディレクトリの確認
    subject_dirs = glob.glob(os.path.join(PROCESSED_DATA_DIR, 'Sub*'))
    subject_dirs.sort()
    
    print(f"\nFound {len(subject_dirs)} subject directories:")
    
    for subject_dir in subject_dirs:
        subject_name = os.path.basename(subject_dir)
        print(f"\n  {subject_name}")
        
        # CNNディレクトリ
        cnn_dir = os.path.join(subject_dir, 'CNN')
        if os.path.exists(cnn_dir):
            emg_files = glob.glob(os.path.join(cnn_dir, '*_emg.csv'))
            print(f"     CNN/ ({len(emg_files)} EMG files)")
        else:
            print(f"     CNN/ directory missing")
        
        # タイミングファイル
        timing_files = glob.glob(os.path.join(subject_dir, '*_timing.csv'))
        if len(timing_files) > 0:
            print(f"     Timing file: {os.path.basename(timing_files[0])}")
        else:
            print(f"     No timing file")
        
        # セッションファイル
        session_files = glob.glob(os.path.join(subject_dir, '*_session_*.csv'))
        if len(session_files) > 0:
            print(f"     Session files: {len(session_files)}")
    
    print()

if __name__ == '__main__':
    check_data_structure()