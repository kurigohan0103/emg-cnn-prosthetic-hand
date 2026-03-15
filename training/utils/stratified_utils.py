import numpy as np
from sklearn.model_selection import train_test_split


# 層化分割用のキーを作成（被験者ID + タスクID）
def create_stratification_key(subject_ids, labels):

    stratify_keys = np.array([
        f"{sub}_{task}" 
        for sub, task in zip(subject_ids, labels)
    ])
    
    return stratify_keys

# 被験者とタスクを考慮した層化分割
def stratified_split_with_subjects(indices, subject_ids, labels, train_ratio=0.75, random_state=42, stratify_by_subject=True, stratify_by_task=True):
    if stratify_by_subject and stratify_by_task:
        # 被験者とタスクの両方で層化
        stratify_keys = create_stratification_key(subject_ids, labels)
        
        # 各キーのサンプル数をチェック
        unique_keys, key_counts = np.unique(stratify_keys, return_counts=True)
        
        # サンプル数が1のキーを見つける
        single_sample_keys = unique_keys[key_counts == 1]
        
        if len(single_sample_keys) > 0:
            print(f"\nWarning: {len(single_sample_keys)} subject-task combinations have only 1 sample")
            print(f"   These will be randomly assigned to train or test")
            
            # サンプル数が2以上のキーのみで層化分割
            mask = np.array([key not in single_sample_keys for key in stratify_keys])
            
            # 層化可能なインデックス
            stratifiable_indices = [idx for idx, m in zip(indices, mask) if m]
            stratifiable_keys = stratify_keys[mask]
            
            # 層化分割
            if len(stratifiable_indices) > 0:
                train_idx_strat, test_idx_strat = train_test_split(
                    stratifiable_indices,
                    test_size=1 - train_ratio,
                    stratify=stratifiable_keys,
                    random_state=random_state
                )
            else:
                train_idx_strat, test_idx_strat = [], []
            
            # サンプル数1のインデックスをランダム分割
            single_indices = [idx for idx, m in zip(indices, mask) if not m]
            if len(single_indices) > 0:
                train_idx_single, test_idx_single = train_test_split(
                    single_indices,
                    test_size=1 - train_ratio,
                    random_state=random_state
                )
                
                train_idx = list(train_idx_strat) + list(train_idx_single)
                test_idx = list(test_idx_strat) + list(test_idx_single)
            else:
                train_idx = train_idx_strat
                test_idx = test_idx_strat
        else:
            # 全てのキーでサンプル数が2以上
            train_idx, test_idx = train_test_split(
                indices,
                test_size=1 - train_ratio,
                stratify=stratify_keys,
                random_state=random_state
            )
    
    elif stratify_by_subject:
        # 被験者のみで層化
        train_idx, test_idx = train_test_split(
            indices,
            test_size=1 - train_ratio,
            stratify=subject_ids,
            random_state=random_state
        )
    
    elif stratify_by_task:
        # タスクのみで層化
        train_idx, test_idx = train_test_split(
            indices,
            test_size=1 - train_ratio,
            stratify=labels,
            random_state=random_state
        )
    
    else:
        # 層化なし（ランダム分割）
        train_idx, test_idx = train_test_split(
            indices,
            test_size=1 - train_ratio,
            random_state=random_state
        )
    
    return train_idx, test_idx

# 分割後の分布を表示
def print_split_distribution(train_idx, test_idx, subject_ids, labels, name="Dataset"):
    
    print(f"\n{name} Distribution:")
    print(f"{'─'*80}")
    
    # 訓練データの分布
    train_subjects = subject_ids[train_idx]
    train_labels = labels[train_idx]
    
    print(f"\nTrain Data ({len(train_idx)} samples):")
    print(f"  By Subject:")
    for subject in sorted(np.unique(subject_ids)):
        count = np.sum(train_subjects == subject)
        percentage = count / len(train_idx) * 100
        print(f"    Subject {subject}: {count:4d} samples ({percentage:5.1f}%)")
    
    print(f"\n  By Task:")
    for task in sorted(np.unique(labels)):
        count = np.sum(train_labels == task)
        percentage = count / len(train_idx) * 100
        print(f"    Task{task}: {count:4d} samples ({percentage:5.1f}%)")
    
    # テストデータの分布
    test_subjects = subject_ids[test_idx]
    test_labels = labels[test_idx]
    
    print(f"\nTest Data ({len(test_idx)} samples):")
    print(f"  By Subject:")
    for subject in sorted(np.unique(subject_ids)):
        count = np.sum(test_subjects == subject)
        percentage = count / len(test_idx) * 100
        print(f"    Subject {subject}: {count:4d} samples ({percentage:5.1f}%)")
    
    print(f"\n  By Task:")
    for task in sorted(np.unique(labels)):
        count = np.sum(test_labels == task)
        percentage = count / len(test_idx) * 100
        print(f"    Task{task}: {count:4d} samples ({percentage:5.1f}%)")
    
    # 被験者×タスクのクロス集計
    print(f"\n  Cross-tabulation (Subject × Task):")
    print(f"    {'':10}", end="")
    for task in sorted(np.unique(labels)):
        print(f"Task{task:1d} ", end="")
    print(" Total")
    
    for dataset_name, indices in [("    Train", train_idx), ("    Test", test_idx)]:
        print(f"{dataset_name:10}", end="")
        dataset_subjects = subject_ids[indices]
        dataset_labels = labels[indices]
        
        for task in sorted(np.unique(labels)):
            count = np.sum(dataset_labels == task)
            print(f"{count:5d} ", end="")
        print(f"{len(indices):5d}")