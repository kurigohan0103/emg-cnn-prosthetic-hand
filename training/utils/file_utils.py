import glob
import os
import numpy as np
from tqdm import tqdm


# ファイルリストを取得
def get_file_list(path_pattern):
    files = sorted(glob.glob(path_pattern))
    if len(files) == 0:
        raise FileNotFoundError(f"No files found: {path_pattern}")
    return files

# 複数のEMGファイルを読み込んで結合
def load_emg_files(file_list, timing_file, emg_dataset_class, **kwargs):
    
    EMG = None
    Label = None
    
    print(f"\nLoading {len(file_list)} EMG files")
    
    successful_files = 0
    
    for i, fname in enumerate(tqdm(file_list, desc="Processing files")):
        try:
            data = emg_dataset_class(
                data_name=fname,
                timing_file=timing_file,
                **kwargs
            )
            
            EMG_, Label_ = data.get_data
            
            # データが有効かチェック
            if EMG_ is None or Label_ is None:
                print(f"\nWarning: {os.path.basename(fname)} returned None data")
                continue
            
            if EMG_.shape[0] == 0:
                print(f"\nWarning: {os.path.basename(fname)} has no samples")
                continue
            
            if i == 0 or EMG is None:
                EMG = EMG_
                Label = Label_
            else:
                EMG = np.concatenate((EMG, EMG_), axis=0)
                Label = np.concatenate((Label, Label_), axis=0)
            
            successful_files += 1
                
        except Exception as e:
            print(f"\nError processing {os.path.basename(fname)}: {e}")
            continue
    
    # 少なくとも1つのファイルが正常に読み込まれたかチェック
    if EMG is None or Label is None:
        raise ValueError(f"Failed to load any valid data from {len(file_list)} files")
    
    print(f"\nLoading completed")
    print(f"   Successfully loaded: {successful_files}/{len(file_list)} files")
    print(f"   EMG shape: {EMG.shape}")
    print(f"   Label shape: {Label.shape}")
    
    return EMG, Label


# 複数被験者のデータを読み込んで結合
def load_multiple_subjects(subjects, emg_dataset_class, 
                          get_data_path_func, get_timing_path_func,
                          **kwargs):
    print(f"\nLoading data from {len(subjects)} subject(s): {subjects}")
    
    all_EMG = []
    all_Label = []
    all_SubjectIDs = []
    
    for subject in subjects:
        print(f"\nLoading Subject {subject}")
        
        try:
            # ファイルリスト取得
            data_files = get_file_list(get_data_path_func(subject))
            timing_files = get_file_list(get_timing_path_func(subject))
            timing_file = timing_files[0]
            
            print(f"Found {len(data_files)} data files")
            print(f"Timing file: {os.path.basename(timing_file)}")
            
            # データ読み込み
            EMG, Label = load_emg_files(
                file_list=data_files,
                timing_file=timing_file,
                emg_dataset_class=emg_dataset_class,
                **kwargs
            )
            
            # データの有効性を再確認
            if EMG is None or Label is None:
                print(f"Warning: Subject {subject} returned None data, skipping...")
                continue
            
            if EMG.shape[0] == 0:
                print(f"Warning: Subject {subject} has no valid samples, skipping...")
                continue
            
            # 被験者IDを作成（各サンプルに対応）
            subject_ids = np.full(len(Label), subject, dtype=int)
            
            all_EMG.append(EMG)
            all_Label.append(Label)
            all_SubjectIDs.append(subject_ids)
            
            print(f"Subject {subject}: {len(Label)} samples")
            
            # タスク分布を表示
            unique_tasks, task_counts = np.unique(Label, return_counts=True)
            print(f"\n   Task distribution:")
            for task, count in zip(unique_tasks, task_counts):
                print(f"     Task{task}: {count} samples")
        
        except Exception as e:
            print(f"Error loading Subject {subject}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # データが1つも読み込まれなかった場合
    if len(all_EMG) == 0:
        raise ValueError(
            f"No valid subject data was loaded from subjects: {subjects}\n"
            f"Please check:\n"
            f"  1. Data files exist in the specified directories\n"
            f"  2. Timing files are correct\n"
            f"  3. File formats are valid"
        )
    
    # 全被験者のデータを結合
    print(f"\nCombining all subjects")
    
    EMG_combined = np.concatenate(all_EMG, axis=0)
    Label_combined = np.concatenate(all_Label, axis=0)
    SubjectIDs_combined = np.concatenate(all_SubjectIDs, axis=0)
    
    print(f"\nCombined data:")
    print(f"   Total samples: {len(Label_combined)}")
    print(f"   EMG shape: {EMG_combined.shape}")
    print(f"   Label shape: {Label_combined.shape}")
    
    # 被験者ごとのサンプル数
    print(f"\n   Samples per subject:")
    for subject in subjects:
        count = np.sum(SubjectIDs_combined == subject)
        percentage = count / len(SubjectIDs_combined) * 100
        print(f"     Subject {subject}: {count} samples ({percentage:.1f}%)")
    
    # タスクごとのサンプル数
    print(f"\n   Samples per task:")
    unique_tasks, task_counts = np.unique(Label_combined, return_counts=True)
    for task, count in zip(unique_tasks, task_counts):
        percentage = count / len(Label_combined) * 100
        print(f"     Task{task}: {count} samples ({percentage:.1f}%)")
    

    return EMG_combined, Label_combined, SubjectIDs_combined