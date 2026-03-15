import sys
import os
import glob
import numpy as np
import pandas as pd
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config


# 指定されたwindow_sizeとオーバーラップ設定で比較分析
class WindowSizeComparison:
    def __init__(self, subjects=None, trim_duration=None, trim_mode=None):
        self.subjects = subjects if subjects else config.SUBJECTS
        self.trim_duration = trim_duration if trim_duration is not None else config.TRIM_DURATION
        self.trim_mode = trim_mode if trim_mode is not None else config.TRIM_MODE
        self.fs = 200.0
        
        self.segment_info = []
    
    # 全被験者のセグメント情報を収集
    def analyze_all_subjects(self):
        print(f"\nWindow Size Comparison Analysis")
        print(f"Subjects: {self.subjects}, Trim: {self.trim_mode} ({self.trim_duration} s), fs: {self.fs} Hz")
        
        for subject_id in self.subjects:
            self._analyze_subject(subject_id)
        
        if len(self.segment_info) == 0:
            print("\n  No segments found")
            return None
        
        df = pd.DataFrame(self.segment_info)
        self.df = df
        
        # 基本統計を表示
        self._print_basic_statistics(df)
        
        # 複数のwindow_size設定で比較
        self._compare_window_configurations(df)
        
        return df
    
    # 1被験者のデータを分析
    def _analyze_subject(self, subject_id):
        try:
            subject_dir = glob.glob(os.path.join(config.PROCESSED_DATA_DIR, f'Sub{subject_id}_*'))[0]
        except IndexError:
            print(f"  Subject {subject_id} directory not found")
            return
        
        emg_files = glob.glob(os.path.join(subject_dir, 'CNN', '*_emg.csv'))
        timing_files = glob.glob(os.path.join(subject_dir, '*_timing.csv'))
        
        if not timing_files:
            print(f"  No timing file found for Subject {subject_id}")
            return
        
        timing_file = timing_files[0]
        
        print(f"\nProcessing Subject {subject_id}: {len(emg_files)} files")
        
        for emg_file in emg_files:
            try:
                trimmed_actions = self._process_emg_file(emg_file, timing_file)
                self._collect_segment_info(trimmed_actions, subject_id, emg_file)
            except Exception as e:
                print(f"  Error: {os.path.basename(emg_file)}: {e}")
                continue
    
    # EMGファイルを処理してトリミング後のセグメントを取得
    def _process_emg_file(self, emg_file, timing_file):
        emg = pd.read_csv(emg_file).drop(['moving', 'characteristic_num'], axis=1, errors='ignore')
        emg = emg.rename({'timestamp': 'timestamp_Under'}, axis='columns')
        
        first_timestamp = emg['timestamp_Under'].iloc[0]
        emg['timestamp_Under'] -= first_timestamp
        adjusted_data = emg.copy()
        
        timing_df = pd.read_csv(timing_file, sep='\t')
        current_file = os.path.basename(emg_file)
        matching_row = timing_df[timing_df['file_name'] == current_file]
        
        if matching_row.empty:
            return []
        
        timing_info = {}
        for i, col in enumerate(['start_time_1', 'start_time_2', 'start_time_3', 'start_time_4'], 1):
            if col in matching_row.columns:
                timing_info[f'action{i}_start'] = matching_row[col].iloc[0]
        
        actions = self._extract_actions(adjusted_data, timing_info)
        trimmed_actions = self._trim_task_edges(actions)
        
        return trimmed_actions
    
    def _extract_actions(self, adjusted_data, timing_info):
        action_configs = [
            {
                'name': 'action1',
                'start': timing_info.get('action1_start', 0), 
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
                'start': timing_info.get('action2_start', 0), 
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
                'start': timing_info.get('action3_start', 0), 
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
                'start': timing_info.get('action4_start', 0), 
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
        
        actions = []
        for config_item in action_configs:
            start_pos = config_item['start']
            end_pos = start_pos + config_item['duration']
            
            action = adjusted_data.loc[
                (adjusted_data['timestamp_Under'] >= start_pos) & 
                (adjusted_data['timestamp_Under'] <= end_pos), :
            ].copy()
            
            if action.empty:
                actions.append(pd.DataFrame())
                continue
            
            action['repetition'] = 0
            action['task'] = 0
            current_pos = start_pos
            
            if config_item['type'] == 'work':
                for j in range(config_item['total_sets']):
                    work_start = current_pos
                    work_end = work_start + config_item['work_duration']
                    mask = (action['timestamp_Under'] >= work_start) & (action['timestamp_Under'] <= work_end)
                    action.loc[mask, 'repetition'] = j + 1
                    action.loc[mask, 'task'] = config_item['work_task']
                    current_pos = work_end
                    
                    if j < config_item['total_sets'] - 1:
                        rest_start = current_pos
                        rest_end = rest_start + config_item['rest_duration']
                        mask = (action['timestamp_Under'] >= rest_start) & (action['timestamp_Under'] <= rest_end)
                        action.loc[mask, 'repetition'] = 0
                        action.loc[mask, 'task'] = config_item['rest_task']
                        current_pos = rest_end
            
            elif config_item['type'] == 'work_up_down':
                for j in range(config_item['total_sets']):
                    work_up_start = current_pos
                    work_up_end = work_up_start + config_item['work_up_duration']
                    mask = (action['timestamp_Under'] >= work_up_start) & (action['timestamp_Under'] <= work_up_end)
                    action.loc[mask, 'repetition'] = j + 1
                    action.loc[mask, 'task'] = config_item['work_up_task']
                    current_pos = work_up_end
                    
                    mid_rest_start = current_pos
                    mid_rest_end = mid_rest_start + config_item['mid_rest_duration']
                    mask = (action['timestamp_Under'] >= mid_rest_start) & (action['timestamp_Under'] <= mid_rest_end)
                    action.loc[mask, 'repetition'] = j + 1
                    action.loc[mask, 'task'] = config_item['mid_rest_task']
                    current_pos = mid_rest_end
                    
                    work_down_start = current_pos
                    work_down_end = work_down_start + config_item['work_down_duration']
                    mask = (action['timestamp_Under'] >= work_down_start) & (action['timestamp_Under'] <= work_down_end)
                    action.loc[mask, 'repetition'] = j + 1
                    action.loc[mask, 'task'] = config_item['work_down_task']
                    current_pos = work_down_end
                    
                    if j < config_item['total_sets'] - 1:
                        rest_start = current_pos
                        rest_end = rest_start + config_item['rest_duration']
                        mask = (action['timestamp_Under'] >= rest_start) & (action['timestamp_Under'] <= rest_end)
                        action.loc[mask, 'repetition'] = 0
                        action.loc[mask, 'task'] = config_item['rest_task']
                        current_pos = rest_end
            
            actions.append(action)
        
        return actions
    
    def _trim_task_edges(self, actions):
        trimmed_actions = []
        
        for action in actions:
            if action.empty:
                trimmed_actions.append(pd.DataFrame())
                continue
            
            trimmed_data = []
            
            for rep in action['repetition'].unique():
                rep_data = action[action['repetition'] == rep]
                
                for task in rep_data['task'].unique():
                    if task == 0:
                        task_data = rep_data[rep_data['task'] == task].copy()
                        trimmed_data.append(task_data)
                        continue
                    
                    task_data = rep_data[rep_data['task'] == task].copy()
                    
                    if task_data.empty:
                        continue
                    
                    task_start = task_data['timestamp_Under'].min()
                    task_end = task_data['timestamp_Under'].max()
                    
                    if self.trim_mode == 'front':
                        trim_start = task_start + self.trim_duration
                        trim_end = task_end
                    elif self.trim_mode == 'back':
                        trim_start = task_start
                        trim_end = task_end - self.trim_duration
                    else:  # 'both'
                        trim_start = task_start + self.trim_duration
                        trim_end = task_end - self.trim_duration
                    
                    if trim_start >= trim_end:
                        continue
                    
                    trimmed_task = task_data[
                        (task_data['timestamp_Under'] >= trim_start) & 
                        (task_data['timestamp_Under'] <= trim_end)
                    ].copy()
                    
                    trimmed_data.append(trimmed_task)
            
            if trimmed_data:
                trimmed_action = pd.concat(trimmed_data, ignore_index=False).sort_index()
                trimmed_actions.append(trimmed_action)
            else:
                trimmed_actions.append(pd.DataFrame())
        
        return trimmed_actions
    
    # セグメント情報を収集
    def _collect_segment_info(self, trimmed_actions, subject_id, emg_file):
        for action_idx, action in enumerate(trimmed_actions, 1):
            if action.empty:
                continue
            
            for rep in action['repetition'].unique():
                if rep == 0:
                    continue
                
                rep_data = action[action['repetition'] == rep]
                
                for task in rep_data['task'].unique():
                    if task in [0, 7, 8]:
                        continue
                    
                    task_data = rep_data[rep_data['task'] == task]
                    segment_length = len(task_data)
                    
                    self.segment_info.append({
                        'subject': subject_id,
                        'file': os.path.basename(emg_file),
                        'action': action_idx,
                        'repetition': int(rep),
                        'task': int(task),
                        'length': segment_length,
                        'duration_sec': segment_length / self.fs
                    })
    
    # 基本統計情報を表示
    def _print_basic_statistics(self, df):
        lengths = df['length'].values
        
        print(f"\nBasic Segment Statistics")
        print(f"Total segments: {len(lengths)}")
        print(f"Min length:     {lengths.min():4d} samples ({lengths.min()/self.fs:.2f} s)")
        print(f"Max length:     {lengths.max():4d} samples ({lengths.max()/self.fs:.2f} s)")
        print(f"Mean length:    {lengths.mean():.1f} samples ({lengths.mean()/self.fs:.2f} s)")
        print(f"Median length:  {np.median(lengths):.1f} samples ({np.median(lengths)/self.fs:.2f} s)")
    
    # 複数のwindow_size設定で比較
    def _compare_window_configurations(self, df):
        lengths = df['length'].values
        total_segments = len(lengths)
        
        # window_size（秒）
        window_sizes_sec = [0.5, 1.0, 1.5, 2.0, 2.5]
        
        # オーバーラップ率
        overlap_rates = [0, 25, 50, 75]
        
        print(f"\nWindow Size Comparison (Overlap: {overlap_rates}%)")
        
        for window_sec in window_sizes_sec:
            window_size = int(window_sec * self.fs)
            
            print(f"\nWindow Size: {window_sec}s ({window_size} samples)")
            
            for overlap_pct in overlap_rates:
                step_size = int(window_size * (1 - overlap_pct / 100))
                
                result = self._simulate_config(lengths, window_size, step_size, total_segments)
                
                print(f"\n【{overlap_pct}% Overlap】 step_size = {step_size}")
                print(f"  Usable segments:     {result['usable_segments']:4d} / {total_segments} "
                      f"({result['retention_pct']:5.1f}%)")
                print(f"  Lost segments:       {result['lost_segments']:4d} "
                      f"({result['lost_pct']:5.1f}%)")
                print(f"  Total windows:       {result['total_windows']:4d}")
                print(f"  Avg windows/segment: {result['avg_windows']:5.2f}")
                
                # タスクごとの詳細
                task_stats = self._analyze_by_task(df, window_size, step_size)
                print(f"\n  By Task:")
                print(f"  {'Task':>6} {'Segments':>10} {'Usable':>10} {'Lost':>10} {'Windows':>10} {'Avg/Seg':>10}")
                for task_id in sorted(task_stats.keys()):
                    stats = task_stats[task_id]
                    print(f"  {task_id:6d} {stats['total']:10d} {stats['usable']:10d} "
                          f"{stats['lost']:10d} {stats['windows']:10d} {stats['avg_windows']:10.2f}")
    
    # "特定のwindow_sizeとstep_sizeでシミュレーション
    def _simulate_config(self, lengths, window_size, step_size, total_segments):
        usable_segments = 0
        total_windows = 0
        
        for length in lengths:
            if length >= window_size:
                usable_segments += 1
                n_windows = (length - window_size) // step_size + 1
                total_windows += n_windows
        
        lost_segments = total_segments - usable_segments
        retention_pct = (usable_segments / total_segments * 100) if total_segments > 0 else 0
        lost_pct = (lost_segments / total_segments * 100) if total_segments > 0 else 0
        avg_windows = (total_windows / usable_segments) if usable_segments > 0 else 0
        
        return {
            'usable_segments': usable_segments,
            'lost_segments': lost_segments,
            'retention_pct': retention_pct,
            'lost_pct': lost_pct,
            'total_windows': total_windows,
            'avg_windows': avg_windows
        }
    
    # タスクごとに分析
    def _analyze_by_task(self, df, window_size, step_size):
        task_stats = {}
        
        for task_id in sorted(df['task'].unique()):
            task_df = df[df['task'] == task_id]
            task_lengths = task_df['length'].values
            
            total = len(task_lengths)
            usable = sum(1 for l in task_lengths if l >= window_size)
            lost = total - usable
            
            windows = 0
            for length in task_lengths:
                if length >= window_size:
                    n_windows = (length - window_size) // step_size + 1
                    windows += n_windows
            
            avg_windows = (windows / usable) if usable > 0 else 0
            
            task_stats[task_id] = {
                'total': total,
                'usable': usable,
                'lost': lost,
                'windows': windows,
                'avg_windows': avg_windows
            }
        
        return task_stats


# 被験者の全ての組み合わせを生成
def generate_subject_combinations(subjects):
    all_combinations = []
    
    # 1人ずつから全員まで
    for r in range(1, len(subjects) + 1):
        for combo in combinations(subjects, r):
            all_combinations.append(list(combo))
    
    return all_combinations


# 全ての被験者の組み合わせで分析を実行
def run_all_combinations():
   
    base_subjects = config.SUBJECTS
    
    print(f"\nMultiple Subject Combination Analysis")
    print(f"Base subjects: {base_subjects}")
    
    # 全組み合わせを生成
    all_combinations = generate_subject_combinations(base_subjects)
    
    print(f"Total combinations to analyze: {len(all_combinations)}")
    print(f"\nCombinations:")
    for i, combo in enumerate(all_combinations, 1):
        print(f"  {i}. {combo}")
    
    # 各組み合わせで分析を実行
    all_results = []
    
    for combo_idx, subject_combo in enumerate(all_combinations, 1):
        print(f"\nCombination {combo_idx}/{len(all_combinations)}: Subjects {subject_combo}")
        
        try:
            analyzer = WindowSizeComparison(
                subjects=subject_combo,
                trim_duration=config.TRIM_DURATION,
                trim_mode=config.TRIM_MODE
            )
            
            df = analyzer.analyze_all_subjects()
            
            if df is not None:
                result_summary = {
                    'subjects': subject_combo,
                    'subject_str': '_'.join(map(str, subject_combo)),
                    'n_subjects': len(subject_combo),
                    'total_segments': len(df),
                    'dataframe': df
                }
                all_results.append(result_summary)
            
        except Exception as e:
            print(f"\n  Error analyzing subjects {subject_combo}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return all_results


# 特定のパターンの組み合わせのみ実行
def run_specific_combinations(combination_type='all'):
    
    base_subjects = config.SUBJECTS
    
    if combination_type == 'single':
        combinations_to_run = [[s] for s in base_subjects]
        print("Analyzing: Single subjects only")
    
    elif combination_type == 'pairs':
        combinations_to_run = [list(combo) for combo in combinations(base_subjects, 2)]
        print("Analyzing: Pairs only")
    
    elif combination_type == 'all':
        return run_all_combinations()
    
    else:
        raise ValueError(f"Unknown combination_type: {combination_type}")
    
    # 選択された組み合わせを実行
    print(f"\nSpecific Combination Analysis")
    print(f"Base subjects: {base_subjects}")
    print(f"Combinations to analyze: {len(combinations_to_run)}")
    for combo in combinations_to_run:
        print(f"  {combo}")
    
    all_results = []
    
    for combo_idx, subject_combo in enumerate(combinations_to_run, 1):
        print(f"\nCombination {combo_idx}/{len(combinations_to_run)}: Subjects {subject_combo}")
        
        try:
            analyzer = WindowSizeComparison(
                subjects=subject_combo,
                trim_duration=config.TRIM_DURATION,
                trim_mode=config.TRIM_MODE
            )
            
            df = analyzer.analyze_all_subjects()
            
            if df is not None:
                result_summary = {
                    'subjects': subject_combo,
                    'subject_str': '_'.join(map(str, subject_combo)),
                    'n_subjects': len(subject_combo),
                    'total_segments': len(df),
                    'dataframe': df
                }
                all_results.append(result_summary)
        
        except Exception as e:
            print(f"\nError analyzing subjects {subject_combo}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return all_results


if __name__ == '__main__':
    
    # 全ての組み合わせを分析
    results = run_all_combinations()
    
    # 個人のみ分析
    # results = run_specific_combinations('single')
    
    # ペアのみ分析
    # results = run_specific_combinations('pairs')
    
    # config.SUBJECTSのみ
    # analyzer = WindowSizeComparison()
    # df = analyzer.analyze_all_subjects()