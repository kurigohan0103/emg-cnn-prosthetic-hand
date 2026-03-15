import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import OneHotEncoder
from scipy.signal import stft


class EMGDataLoader(Dataset):
    
    def __init__(self, emg_data, label, device, subject_ids=None, use_stft=False, stft_params=None, shuffle_data=True, random_seed=42, verbose=True):
        
        self.device = device
        self.use_stft = use_stft
        self.stft_params = stft_params or {'fs': 200, 'nperseg': 256}
        self.verbose = verbose
        
        self._validate_input(emg_data, label, subject_ids)
        
        # 被験者IDを保持
        self.subject_ids = subject_ids if subject_ids is not None else np.zeros(len(label), dtype=int)
        
        if shuffle_data:
            emg_data, label, self.subject_ids = self._shuffle_data(emg_data, label, self.subject_ids, random_seed)
        
        if use_stft:
            self.emg_data = self._apply_stft(emg_data)
            if self.verbose:
                print(f"STFT applied")
                print(f"   fs={self.stft_params['fs']}, nperseg={self.stft_params['nperseg']}")
        else:
            self.emg_data = emg_data
        
        self.label = self._encode_labels(label)
    
    # 入力データの妥当性を検証
    def _validate_input(self, emg_data, label, subject_ids):
        if len(emg_data) != len(label):
            raise ValueError(
                f"Mismatch: EMG={len(emg_data)}, Label={len(label)}"
            )
        
        if subject_ids is not None and len(subject_ids) != len(label):
            raise ValueError(
                f"Mismatch: Subject IDs={len(subject_ids)}, Label={len(label)}"
            )
        
        if len(emg_data) == 0:
            raise ValueError("Dataset is empty")
        
        if self.verbose:
            print(f"Input validation OK: {len(emg_data)} samples")
            if subject_ids is not None:
                unique_subjects = np.unique(subject_ids)
                print(f"   Subjects: {list(unique_subjects)}")
    
    # データをシャッフル（被験者IDも一緒に）
    def _shuffle_data(self, emg_data, label, subject_ids, random_seed):
        np.random.seed(random_seed)
        indices = np.random.permutation(len(emg_data))
        
        if self.verbose:
            print(f"Data shuffled (seed={random_seed})")
            print(f"   New order: {indices[:5].tolist()}...")
        
        return emg_data[indices], label[indices], subject_ids[indices]
    
    # ラベルをOne-Hot形式に変換
    def _encode_labels(self, label):
        num_classes = np.unique(label)
        
        if self.verbose:
            print(f"\nLabel encoding")
            print(f"   Classes: {len(num_classes)} {list(num_classes)}")
        
        encoder = OneHotEncoder(sparse_output=False)
        
        if label.ndim == 1:
            label = label.reshape(-1, 1)
        
        result = encoder.fit_transform(label)
        
        if self.verbose:
            print(f"   One-Hot shape: {result.shape}")
            print(f"   Example: {label[0]} → {result[0]}")
        
        return result
    
    # 短時間フーリエ変換を適用
    def _apply_stft(self, data, fs=200, nperseg=256):
        _, _, zxx = stft(data, fs=fs, nperseg=nperseg)
        return zxx
    
    def __len__(self):
        return len(self.emg_data)
    
    def __getitem__(self, index):
        emg_tensor = torch.tensor(self.emg_data[index], dtype=torch.float32, device=self.device).unsqueeze(0)
        
        label_tensor = torch.tensor(self.label[index], dtype=torch.float32, device=self.device)
        
        return emg_tensor, label_tensor
    
    # クラス分布を取得
    def get_class_distribution(self):
        # One-Hotから元のラベルに戻す
        original_labels = np.argmax(self.label, axis=1)
        unique, counts = np.unique(original_labels, return_counts=True)
        return dict(zip(unique, counts))
    
    # 被験者IDを取得
    def get_subject_ids(self):
        return self.subject_ids.copy()
    
    # データセット情報を表示
    def print_info(self):
        print("\nDataset Information")
        print(f"Samples: {len(self)}")
        print(f"EMG shape: {self.emg_data[0].shape}")
        print(f"Label shape: {self.label[0].shape}")
        print(f"Device: {self.device}")
        print(f"STFT: {self.use_stft}")
        
        # 被験者分布
        unique_subjects = np.unique(self.subject_ids)
        if len(unique_subjects) > 1 or unique_subjects[0] != 0:
            print(f"\nSubject Distribution:")
            for subject in sorted(unique_subjects):
                count = np.sum(self.subject_ids == subject)
                percentage = (count / len(self)) * 100
                print(f"  Subject {subject}: {count:3d} samples ({percentage:5.1f}%)")
        
        # クラス分布
        distribution = self.get_class_distribution()
        print(f"\nClass Distribution:")
        for cls, count in sorted(distribution.items()):
            percentage = (count / len(self)) * 100
            print(f"  Class {cls}: {count:3d} samples ({percentage:5.1f}%)")
        print()