import os 

import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, stft
from sklearn.utils import class_weight, resample


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

# フィルターの適用
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    w0 = 50 / (fs / 2)  # 正規化角周波数
    b, a = iirnotch(w0, 30)
    data = filtfilt(b, a, data)
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    filtered_data = filtfilt(b, a, data)

    return filtered_data


class EMGdataset:  
    def __init__(self, data_name, timing_file, trim_duration, trim_mode='both', window_size=None, step_size=None, auto_balance=True, balance_strategy='min', balanced_per_rep=False, filter_enabled=False, lowcut=20, highcut=80):
        
        if not isinstance(window_size, int) or window_size <= 0:
            raise ValueError(
                f"window_size must be a positive integer, got {window_size}"
            )
        
        if trim_mode not in ['front', 'back', 'both']:
            raise ValueError(
                f"trim_mode must be 'front', 'back', or 'both', got '{trim_mode}'"
            )
        
        if trim_duration < 0:
            raise ValueError(
                f"trim_duration must be non-negative, got {trim_duration}"
            )
        
        if balance_strategy not in ['min', 'max', 'mean']:
            raise ValueError(
                f"balance_strategy must be 'min', 'max', or 'mean', got '{balance_strategy}'"
            )

        self.data_name = data_name
        self.timing_file = timing_file
        self.trim_duration = trim_duration
        self.trim_mode = trim_mode
        self.window_size = window_size
        self.step_size = step_size if step_size is not None else window_size # 指定されなければ window_size と同じ（オーバーラップなし）
        self.filter_enabled = filter_enabled
        self.lowcut = lowcut
        self.highcut = highcut
        self.fs = 200.0
        self.auto_balance = auto_balance
        self.balance_strategy = balance_strategy
        self.balanced_per_rep = balanced_per_rep
        self.emg = pd.DataFrame()
        self.adjusted_data = pd.DataFrame()
        self.actions = []
        self.trimmed_actions = []

        print(f"\nTRIM MODE: {self.trim_mode}, trim_duration: {self.trim_duration} s")
        
        self.load_csv()
        self.adjust_timing()
        self.load_timing_info()
        self.extract_actions()
        self.trim_task_edges()
        
        # モード判定
        if self.balanced_per_rep:
            print(f"\nMODE: balanced_per_rep, window_size: {self.window_size} ({self.window_size/self.fs:.2f} s)")
            self._validate_window_size()
            self.step_size = self.window_size
            self.Action, self.Label = self.To_Numpy_balanced_per_rep()
        else:
            if self.step_size == self.window_size or self.step_size is None:
                # オーバーラップなし
                print(f"\nMODE: sliding window (no overlap), window_size: {self.window_size} ({self.window_size/self.fs:.2f} s)")
            else:
                # オーバーラップあり
                overlap_rate = (1 - self.step_size / self.window_size) * 100
                print(f"\nMODE: sliding window (overlap: {overlap_rate:.0f}%), window_size: {self.window_size} ({self.window_size/self.fs:.2f} s), step_size: {self.step_size} ({self.step_size/self.fs:.2f} s)")

            self._validate_window_size()

            self.Action, self.Label = self.To_Numpy_with_windowing()
            
            # auto_balanceの処理
            if self.auto_balance:
                print(f"\nBALANCE: Auto-balance (strategy='{self.balance_strategy}')")
                self.Action, self.Label = self.balance_dataset(strategy=self.balance_strategy)
        
        # データ取得結果の表示
        if self.Action is not None and len(self.Action) > 0:
            print(f"\nRESULT: Action shape: {self.Action.shape}, Label shape: {self.Label.shape}")
            self._print_class_distribution() # クラスごとのデータ分布を表示
        else:
            raise ValueError("Error: Could not get data.")
    

    @property
    def get_data(self):
        return self.Action, self.Label
    

    # データ読み込み
    def load_csv(self):
        self.emg = pd.read_csv(os.path.join(self.data_name)).drop(['moving', 'characteristic_num'], axis=1)
        self.emg = self.emg.rename({'timestamp': 'timestamp_Under'}, axis='columns')
        print(f'emg {self.data_name} has been loaded.')
    
    # 時間を0基準に調整
    def adjust_timing(self):
        first_timestamp = self.emg['timestamp_Under'].iloc[0]
        self.emg['timestamp_Under'] -= first_timestamp
        self.adjusted_data = self.emg.copy()
        return self.adjusted_data
    
    def load_timing_info(self):
        timing_df = pd.read_csv(self.timing_file, sep='\t')    
        current_file = os.path.basename(self.data_name)
        
        matching_row = timing_df[timing_df['file_name'] == current_file]
        
        if matching_row.empty:
            print(f"Warning: {current_file} information not found")
            return None
        
        start_columns = ['start_time_1', 'start_time_2', 'start_time_3', 'start_time_4']
        
        timing_info = {}
        for i, col in enumerate(start_columns, 1):
            if col in matching_row.columns:
                timing_info[f'action{i}_start'] = matching_row[col].iloc[0]
            else:
                print(f"Warning: col {col} not found")
        self.timing_info = timing_info
        return timing_info
    
    def extract_actions(self):
        if self.adjusted_data.empty:
            print("Error: adjusted_data is empty!!")
            return None
        
        if not hasattr(self, 'timing_info') or self.timing_info is None or not self.timing_info:
            print("Error: timing_info is empty!! Please run load_timing_info() first.")
            return None
        
        actions = []
        action_configs = [
            {
                'name': 'action1',
                'start': self.timing_info.get('action1_start', 0),
                'duration': 95,
                'type': 'work',
                'work_duration': 5,
                'rest_duration': 5,
                'total_sets': 10,
                'work_task': 1,
                'rest_task': 0
            },
            {
                'name': 'action2',
                'start': self.timing_info.get('action2_start', 0),
                'duration': 145,
                'type': 'work_up_down',
                'work_up_duration': 2.5,
                'mid_rest_duration': 5,
                'work_down_duration': 2.5,
                'rest_duration': 5,
                'total_sets': 10,
                'work_up_task': 2,
                'mid_rest_task': 7,
                'work_down_task': 3,
                'rest_task': 0
            },
            {
                'name': 'action3',
                'start': self.timing_info.get('action3_start', 0),
                'duration': 95,
                'type': 'work',
                'work_duration': 5,
                'rest_duration': 5,
                'total_sets': 10,
                'work_task': 4,
                'rest_task': 0
            },
            {
                'name': 'action4',
                'start': self.timing_info.get('action4_start', 0),
                'duration': 145,
                'type': 'work_up_down',
                'work_up_duration': 2.5,
                'mid_rest_duration': 5,
                'work_down_duration': 2.5,
                'rest_duration': 5,
                'total_sets': 10,
                'work_up_task': 5,
                'mid_rest_task': 8,
                'work_down_task': 6,
                'rest_task': 0
            }
        ]
        
        for i, config in enumerate(action_configs):
            start_pos = config['start']
            end_pos = start_pos + config['duration']
    
            action = self.adjusted_data.loc[
                (self.adjusted_data['timestamp_Under'] >= start_pos) & 
                (self.adjusted_data['timestamp_Under'] <= end_pos), :
            ].copy()
    
            if action.empty:
                print(f"  Warning: Data does not exist.")
                actions.append(pd.DataFrame())
                continue
    
            action['repetition'] = 0
            action['task'] = 0
    
            current_pos = start_pos
    
            if config['type'] == 'work':
                for j in range(config['total_sets']):
                    work_start = current_pos
                    work_end = work_start + config['work_duration']
                    mask = (action['timestamp_Under'] >= work_start) & (action['timestamp_Under'] <= work_end)
                    action.loc[mask, 'repetition'] = j + 1
                    action.loc[mask, 'task'] = config['work_task']
            
                    current_pos = work_end
            
                    if j < config['total_sets'] - 1:
                        rest_start = current_pos
                        rest_end = rest_start + config['rest_duration']
                
                        mask = (action['timestamp_Under'] >= rest_start) & (action['timestamp_Under'] <= rest_end)
                        action.loc[mask, 'repetition'] = 0
                        action.loc[mask, 'task'] = config['rest_task']
                
                        current_pos = rest_end
    
            elif config['type'] == 'work_up_down':
                for j in range(config['total_sets']):
                    work_up_start = current_pos
                    work_up_end = work_up_start + config['work_up_duration']
            
                    mask = (action['timestamp_Under'] >= work_up_start) & (action['timestamp_Under'] <= work_up_end)
                    action.loc[mask, 'repetition'] = j + 1
                    action.loc[mask, 'task'] = config['work_up_task']
            
                    current_pos = work_up_end
            
                    mid_rest_start = current_pos
                    mid_rest_end = mid_rest_start + config['mid_rest_duration']
            
                    mask = (action['timestamp_Under'] >= mid_rest_start) & (action['timestamp_Under'] <= mid_rest_end)
                    action.loc[mask, 'repetition'] = j + 1
                    action.loc[mask, 'task'] = config['mid_rest_task']
            
                    current_pos = mid_rest_end
            
                    work_down_start = current_pos
                    work_down_end = work_down_start + config['work_down_duration']
            
                    mask = (action['timestamp_Under'] >= work_down_start) & (action['timestamp_Under'] <= work_down_end)
                    action.loc[mask, 'repetition'] = j + 1
                    action.loc[mask, 'task'] = config['work_down_task']
            
                    current_pos = work_down_end
            
                    if j < config['total_sets'] - 1:
                        rest_start = current_pos
                        rest_end = rest_start + config['rest_duration']
                
                        mask = (action['timestamp_Under'] >= rest_start) & (action['timestamp_Under'] <= rest_end)
                        action.loc[mask, 'repetition'] = 0
                        action.loc[mask, 'task'] = config['rest_task']
                
                        current_pos = rest_end
    
            actions.append(action)
        
        if len(actions) >= 1:
            self.action1 = actions[0]
        else:
            self.action1 = pd.DataFrame()
    
        if len(actions) >= 2:
            self.action2 = actions[1]
        else:
            self.action2 = pd.DataFrame()
    
        if len(actions) >= 3:
            self.action3 = actions[2]
        else:
            self.action3 = pd.DataFrame()
    
        if len(actions) >= 4:
            self.action4 = actions[3]
        else:
            self.action4 = pd.DataFrame()
        
        self.actions = actions
        return actions
    
    def trim_task_edges(self):
        if not hasattr(self, 'actions') or not self.actions:
            raise ValueError("Error: actions data is empty! Please run extract_actions() first.")
    
        trimmed_actions = []
    
        for action_idx, action in enumerate(self.actions, 1):
            if action.empty:
                print(f"Warning: action{action_idx} is empty!")
                trimmed_actions.append(pd.DataFrame())
                continue
        
            trimmed_data = []
        
            for rep in action['repetition'].unique():
                rep_data = action[action['repetition'] == rep]
            
                for task in rep_data['task'].unique():
                    if task == 0: # 休憩タスク(task=0)はトリミングしない
                        task_data = rep_data[rep_data['task'] == task].copy()
                        trimmed_data.append(task_data)
                        continue
                
                    task_data = rep_data[rep_data['task'] == task].copy()
                
                    if task_data.empty:
                        continue
                
                    task_start = task_data['timestamp_Under'].min()
                    task_end = task_data['timestamp_Under'].max()
                    task_duration = task_end - task_start

                    # トリミング範囲
                    if self.trim_mode == 'front':
                        trim_start = task_start + self.trim_duration
                        trim_end = task_end
                        removed_duration = self.trim_duration
                    
                    elif self.trim_mode == 'back':
                        trim_start = task_start
                        trim_end = task_end - self.trim_duration
                        removed_duration = self.trim_duration
                        
                    else:  # 'both'
                        trim_start = task_start + self.trim_duration
                        trim_end = task_end - self.trim_duration
                        removed_duration = self.trim_duration * 2

                    # トリミング後のデータが有効かチェック
                    if trim_start >= trim_end:
                        error_msg = (
                            f"\nError: trim_duration is too large\n"
                            f"  Location: action{action_idx}, repetition={rep}, task={task}\n"
                            f"  Trim mode: '{self.trim_mode}'\n"
                            f"  Task duration: {task_duration:.2f} s ({int(task_duration * self.fs)} samples)\n"
                            f"  trim_duration: {self.trim_duration} s\n"
                        )
                    
                        if self.trim_mode == 'front':
                            error_msg += (
                                f"  Trying to remove: Front {self.trim_duration} s\n"
                                f"  Remaining: {task_duration - self.trim_duration:.2f} s\n"
                            )
                        elif self.trim_mode == 'back':
                            error_msg += (
                                f"  Trying to remove: Back {self.trim_duration} s\n"
                                f"  Remaining: {task_duration - self.trim_duration:.2f} s\n"
                            )
                        else:  # 'both'
                            error_msg += (
                                f"  Trying to remove: Front {self.trim_duration} s + Back {self.trim_duration} s = {removed_duration} s\n"
                                f"  Remaining: {task_duration - removed_duration:.2f} s\n"
                            )
                    
                        raise ValueError(error_msg)
                
                    # トリミング実行
                    trimmed_task = task_data[(task_data['timestamp_Under'] >= trim_start) & (task_data['timestamp_Under'] <= trim_end)].copy()
                
                    trimmed_data.append(trimmed_task)
        
            if trimmed_data:
                trimmed_action = pd.concat(trimmed_data, ignore_index=False).sort_index()
                trimmed_actions.append(trimmed_action)
                print(f"action{action_idx}: Trimmed from {len(action)} to {len(trimmed_action)} rows")
            else:
                trimmed_actions.append(pd.DataFrame())
    
        self.trimmed_actions = trimmed_actions
        if len(trimmed_actions) >= 1:
            self.trimmed_action1 = trimmed_actions[0]
        if len(trimmed_actions) >= 2:
            self.trimmed_action2 = trimmed_actions[1]
        if len(trimmed_actions) >= 3:
            self.trimmed_action3 = trimmed_actions[2]
        if len(trimmed_actions) >= 4:
            self.trimmed_action4 = trimmed_actions[3]
    
        return trimmed_actions
    

    def To_Numpy_balanced_per_rep(self):

        if not hasattr(self, 'trimmed_actions') or not self.trimmed_actions:
            raise ValueError("Error: trimmed_actions is empty!")
        
        action_list = []
        label_list = []
        
        task_stats = {}
        
        for action_idx in range(len(self.trimmed_actions)):
            full_data = self.trimmed_actions[action_idx].to_numpy()
            
            if full_data.size == 0:
                continue
            
            emg_data = full_data[:, 1:-2]
            rep_data = full_data[:, -2]
            task_data = full_data[:, -1]
            
            # フィルタ処理
            if self.filter_enabled:
                    filtered_sig = np.zeros(emg_data.shape)
                    for ch in range(emg_data.shape[-1]):
                        filtered_sig[:, ch] = bandpass_filter(emg_data[:, ch], self.lowcut, self.highcut, self.fs)
                    emg_data = filtered_sig
            
            unique_task = np.unique(task_data)
            
            for task_num in unique_task:
                if task_num in [0, 7, 8]:
                    continue
                
                task_mask = (task_data == task_num)
                task_emg_data = emg_data[task_mask]
                task_rep_data = rep_data[task_mask]
                
                task_key = f"Task {int(task_num)}"
                if task_key not in task_stats:
                    task_stats[task_key] = 0
                
                for rep_num in np.unique(task_rep_data):
                    if rep_num == 0:
                        continue
                    
                    rep_mask = (task_rep_data == rep_num)
                    segment_data = task_emg_data[rep_mask]
                    
                    available_length = len(segment_data)
                    
                    # セグメントが十分な長さがあるか確認
                    if available_length < self.window_size:
                        continue
                    
                    # 各repetitionから中央1つだけ取得
                    start_pos = (available_length - self.window_size) // 2
                    window_data = segment_data[start_pos:start_pos + self.window_size]
                    
                    action_list.append(window_data)
                    label_list.append(task_num)
                    task_stats[task_key] += 1
        
        if not action_list:
            print("Warning: No valid data found")
            return np.array([]), np.array([])
        
        Action = np.array(action_list)
        Label = np.array(label_list)[:, np.newaxis]
        
        # 統計情報の表示
        print(f"\n  Acquisition statistics:")
        for task_key, count in sorted(task_stats.items()):
            print(f"    {task_key}: {count} sample (one from each rep.)")
        
        return Action, Label
    

    def To_Numpy_with_windowing(self):
        if not hasattr(self, 'trimmed_actions') or not self.trimmed_actions:
            raise ValueError("Error: trimmed_actions is empty! Please run trim_task_edges() first.")
        
        action_list = []
        label_list = []
        window_stats = {}
        
        for action_idx in range(len(self.trimmed_actions)):
            full_data = self.trimmed_actions[action_idx].to_numpy()
            
            if full_data.size == 0:
                continue
            
            emg_data = full_data[:, 1:-2]
            rep_data = full_data[:, -2]
            task_data = full_data[:, -1]
            
            if self.filter_enabled:
                b, a = signal.butter(4, [self.lowcut, self.highcut], btype='band', fs=self.fs)
                emg_data = signal.filtfilt(b, a, emg_data, axis=0)
            
            unique_task = np.unique(task_data)
            
            for task_num in unique_task:
                if task_num in [0, 7, 8]:
                    continue
                
                task_mask = (task_data == task_num)
                task_emg_data = emg_data[task_mask]
                task_rep_data = rep_data[task_mask]
                
                task_key = f"action{action_idx+1}_task{int(task_num)}"
                window_stats[task_key] = {'total_windows': 0, 'reps': 0}
                
                for rep_num in np.unique(task_rep_data):
                    if rep_num == 0:
                        continue
                    
                    rep_mask = (task_rep_data == rep_num)
                    segment_data = task_emg_data[rep_mask]
                    
                    available_length = len(segment_data)
                    
                    if available_length < self.window_size:
                        continue
                    
                    n_windows = (available_length - self.window_size) // self.step_size + 1
                    
                    for w in range(n_windows):
                        start = w * self.step_size
                        end = start + self.window_size
                        
                        if end > available_length:
                            break
                        
                        window_data = segment_data[start:end]
                        
                        action_list.append(window_data)
                        label_list.append(task_num)
                        
                        window_stats[task_key]['total_windows'] += 1
                    
                    window_stats[task_key]['reps'] += 1
        
        if not action_list:
            print("Warning: No valid data found")
            return np.array([]), np.array([])
        
        Action = np.array(action_list)
        Label = np.array(label_list)[:, np.newaxis]
        
        print(f"\n  Window Generation Statistics:")
        for task_key, stats in window_stats.items():
            if stats['reps'] > 0:
                avg_windows = stats['total_windows'] / stats['reps']
                print(f"    {task_key}: {stats['total_windows']} windows "
                      f"({stats['reps']} reps, average {avg_windows:.1f} windows/rep)")
        
        return Action, Label
    

    def balance_dataset(self, strategy='min'):
        
        if self.Action is None or len(self.Action) == 0:
            raise ValueError(
                "Cannot balance dataset: No data available.\n"
                "Possible causes:\n"
                "  - Data generation failed during initialization\n"
                "  - All segments were filtered out due to size constraints\n"
                "  - trim_duration is too large\n"
                "Please check your initialization parameters."
            )
        
        if not isinstance(self.Action, np.ndarray):
            raise TypeError(
                f"Action must be a numpy array, got {type(self.Action)}"
            )
        
        if self.Label is None or len(self.Label) == 0:
            raise ValueError(
                "Cannot balance dataset: No labels available."
            )

        unique_labels = np.unique(self.Label)
        label_counts = [np.sum(self.Label == label) for label in unique_labels]
        
        # サンプリング数を決定
        if strategy == 'min':
            n_samples = min(label_counts)
            print(f"  adjust to the minimum sample size: {n_samples}")
        elif strategy == 'max':
            n_samples = max(label_counts)
            print(f"  adjust to the maximum sample size: {n_samples}")
        elif strategy == 'mean':
            n_samples = int(np.mean(label_counts))
            print(f"  adjust to the average sample size: {n_samples}")
        elif isinstance(strategy, int):
            n_samples = strategy
            print(f"  adjust to the specified sample size: {n_samples}")
        else:
            raise ValueError("strategy must be 'min', 'max', or an integer")
        
        balanced_data = []
        balanced_labels = []
        
        for label in unique_labels:
            # 各クラスのデータを抽出
            mask = (self.Label == label).flatten()
            class_data = self.Action[mask]
            class_labels = self.Label[mask]
            
            current_count = len(class_data)
            
            # リサンプリング
            if current_count > n_samples:
                # ダウンサンプリング
                resampled_data, resampled_labels = resample(
                    class_data, class_labels,
                    n_samples=n_samples,
                    replace=False,
                    random_state=42
                )
                print(f"    Task {int(label)}: {current_count} → {n_samples} sample (reduction)")
            elif current_count < n_samples:
                # アップサンプリング
                resampled_data, resampled_labels = resample(
                    class_data, class_labels,
                    n_samples=n_samples,
                    replace=True,
                    random_state=42
                )
                print(f"    Task {int(label)}: {current_count} → {n_samples} sample (augmentation)")
            else:
                resampled_data = class_data
                resampled_labels = class_labels
                print(f"    Task {int(label)}: {current_count} sample (no changes)")
            
            balanced_data.append(resampled_data)
            balanced_labels.append(resampled_labels)
        
        # 結合してシャッフル
        balanced_Action = np.vstack(balanced_data)
        balanced_Label = np.vstack(balanced_labels)
        
        # シャッフル
        indices = np.random.RandomState(42).permutation(len(balanced_Action))
        balanced_Action = balanced_Action[indices]
        balanced_Label = balanced_Label[indices]
        
        return balanced_Action, balanced_Label
    

    # クラスウェイトを計算（不均衡対策用）
    def get_class_weights(self):
    
        if self.Label is None or len(self.Label) == 0:
            return None
        
        class_weights = class_weight.compute_class_weight(
            'balanced',
            classes=np.unique(self.Label),
            y=self.Label.flatten()
        )
        
        class_weight_dict = dict(enumerate(class_weights))
        
        print("\n-- class weight--")
        for label, weight in class_weight_dict.items():
            print(f"  Task {label}: {weight:.3f}")
        
        return class_weight_dict
    

    def check_pre_trim_lengths(self):

        print(f"\n-- pre-trimming segment length --")
    
        for action_idx, action in enumerate(self.actions, 1):
            for task in action['task'].unique():
                if task in [0, 7, 8]:
                    continue
            
                task_data = action[action['task'] == task]
                for rep in task_data['repetition'].unique():
                    if rep == 0:
                        continue
                
                    rep_data = task_data[task_data['repetition'] == rep]
                    length = len(rep_data)
                    duration = length / self.fs
                
                    print(f"  action{action_idx} task{int(task)} rep{int(rep)}: "
                        f"{length}sample ({duration:.2f} s)")
    
    def check_segment_lengths(self):

        print(f"\n-- segment lengths (window_size={self.window_size}) --")
    
        for action_idx, action in enumerate(self.trimmed_actions, 1):
            for task in action['task'].unique():
                if task in [0, 7, 8]:
                    continue
            
                task_data = action[action['task'] == task]
                for rep in task_data['repetition'].unique():
                    if rep == 0:
                        continue
                
                    rep_data = task_data[task_data['repetition'] == rep]
                    length = len(rep_data)
                    duration = length / self.fs
                
                    print(f"  action{action_idx} task{int(task)} rep{int(rep)}: "
                        f"{length}sample ({duration:.2f} s)")
    

    # クラスごとのデータ分布を表示
    def _print_class_distribution(self):

        unique_labels, counts = np.unique(self.Label, return_counts=True)
        print(f"\n  class distribution:")
        for label, count in zip(unique_labels, counts):
            percentage = count / len(self.Label) * 100
            print(f"    Task {int(label)}: {count:4d} sample ({percentage:5.1f} %)")
    

    # ユーザーが指定したwindow_sizeが最小セグメント長より大きい場合の処理
    def _validate_window_size(self):

        if not hasattr(self, 'trimmed_actions') or not self.trimmed_actions:
            raise ValueError("trimmed_actions is empty!")
    
        # 最小セグメント長を検出
        min_length = float('inf')
        min_info = {}
    
        for action_idx, action in enumerate(self.trimmed_actions, 1):
            if action.empty:
                continue
        
            for task in action['task'].unique():
                if task in [0, 7, 8]:  # 休憩タスクはスキップ
                    continue
            
                task_data = action[action['task'] == task]
                for rep in task_data['repetition'].unique():
                    if rep == 0:
                        continue
                
                    rep_data = task_data[task_data['repetition'] == rep]
                    length = len(rep_data)
                
                    if length < min_length:
                        min_length = length
                        min_info = {
                            'action': action_idx,
                            'task': int(task),
                            'rep': int(rep),
                            'length': length,
                            'duration': length / self.fs
                        }
    
        if min_length == float('inf'):
            return  # セグメントが見つからない場合はスキップ
    
        print(f"  minimum segment length: {min_length} sample ({min_length/self.fs:.2f} s) "
            f"[action{min_info['action']} task{min_info['task']} rep{min_info['rep']}]")
    
        # window_sizeと比較
        if self.window_size > min_length:
            error_msg = (
                f"\nError: window_size is too large\n"
                f"  指定されたwindow_size: {self.window_size}sample ({self.window_size/self.fs:.2f} s)\n"
                f"  minimum segment length:      {min_length}sample ({min_length/self.fs:.2f} s)\n"
                f"  minimum segment position:    action{min_info['action']} task{min_info['task']} rep{min_info['rep']}\n"
                f"\n"
                f"If this continues, the data for the following tasks will become unusable.:\n"
            )
        
            # 使用できないタスクをリストアップ
            skipped_tasks = set()
            for action_idx, action in enumerate(self.trimmed_actions, 1):
                if action.empty:
                    continue
                for task in action['task'].unique():
                    if task in [0, 7, 8]:
                        continue
                    task_data = action[action['task'] == task]
                    for rep in task_data['repetition'].unique():
                        if rep == 0:
                            continue
                        rep_data = task_data[task_data['repetition'] == rep]
                        if len(rep_data) < self.window_size:
                            skipped_tasks.add(int(task))
        
            if skipped_tasks:
                error_msg += f"  → Task {sorted(skipped_tasks)}\n"
        
            raise ValueError(error_msg)
    
        elif self.window_size > min_length * 0.8:
            # 80%以上の場合は警告
            print(f"\nWarning: window_size is {(self.window_size/min_length*100):.0f} % of the minimum segment length.")
            print(f"  Some tasks may only be able to allocate a small amount of window space.")