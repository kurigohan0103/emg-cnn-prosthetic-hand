import numpy as np
import torch
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from utils.stratified_utils import (
    stratified_split_with_subjects,
    print_split_distribution
)

# 層化分割（タスクのみ）
def stratified_split(dataset, train_ratio=0.75, random_seed=42, verbose=True):
    if verbose:
        print("\nStratified Split (by Task)")
    
    # ラベルを取得
    all_labels = []
    for i in range(len(dataset)):
        _, label_tensor = dataset[i]
        class_index = label_tensor.argmax().item()
        all_labels.append(class_index)
    
    all_labels = np.array(all_labels)
    
    if verbose:
        print(f"Total samples: {len(all_labels)}")
        print(f"Unique classes: {np.unique(all_labels)}")
    
    # 層化分割
    indices = list(range(len(dataset)))
    train_idx, test_idx = train_test_split(indices, test_size=1 - train_ratio, stratify=all_labels, random_state=random_seed)
    
    if verbose:
        print(f"Train: {len(train_idx)} samples")
        print(f"Test:  {len(test_idx)} samples")
    
    # Subset作成
    train_dataset = Subset(dataset, train_idx)
    test_dataset = Subset(dataset, test_idx)
    
    # クラス分布確認
    if verbose:
        print("\nClass Distribution:")
        _print_distribution(train_dataset, train_idx, all_labels, "Train")
        _print_distribution(test_dataset, test_idx, all_labels, "Test")
    
    return train_dataset, test_dataset

# 層化分割（被験者+タスク）
def stratified_split_with_subject_info(dataset, subject_ids, train_ratio=0.75, random_seed=42, stratify_by_subject=True, stratify_by_task=True, verbose=True):

    if verbose:
        stratify_info = []
        if stratify_by_subject:
            stratify_info.append("Subject")
        if stratify_by_task:
            stratify_info.append("Task")
        print(f"\nStratified Split (by {' and '.join(stratify_info)})")
    
    # ラベルを取得
    all_labels = []
    for i in range(len(dataset)):
        _, label_tensor = dataset[i]
        class_index = label_tensor.argmax().item()
        all_labels.append(class_index)
    
    all_labels = np.array(all_labels)
    
    if verbose:
        print(f"\nTotal samples: {len(all_labels)}")
        print(f"Subjects: {sorted(np.unique(subject_ids))}")
        print(f"Tasks: {sorted(np.unique(all_labels))}")
    
    # 層化分割
    indices = list(range(len(dataset)))
    train_idx, test_idx = stratified_split_with_subjects(
        indices=indices,
        subject_ids=subject_ids,
        labels=all_labels,
        train_ratio=train_ratio,
        random_state=random_seed,
        stratify_by_subject=stratify_by_subject,
        stratify_by_task=stratify_by_task
    )
    
    if verbose:
        print(f"\nSplit completed:")
        print(f"   Train: {len(train_idx)} samples ({len(train_idx)/len(dataset)*100:.1f}%)")
        print(f"   Test:  {len(test_idx)} samples ({len(test_idx)/len(dataset)*100:.1f}%)")
    
    # Subset作成
    train_dataset = Subset(dataset, train_idx)
    test_dataset = Subset(dataset, test_idx)
    
    # 分布確認
    if verbose:
        print_split_distribution(
            train_idx, test_idx,
            subject_ids, all_labels,
            name="Final"
        )
    
    return train_dataset, test_dataset

# クラス分布を表示
def _print_distribution(dataset, indices, all_labels, name):

    labels = all_labels[indices]
    
    print(f"\n{name}:")
    for cls in sorted(np.unique(labels)):
        count = np.sum(labels == cls)
        percentage = (count / len(labels)) * 100
        print(f"  Class {cls}: {count:3d} ({percentage:5.1f}%)")
    print(f"  Total:    {len(labels):3d}")