import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob


class EMGdataset:
    # 初期設定
    def __init__(self, Dataname):
        self.actions = []
        self.Dataname = Dataname
        self.adjusted_data = pd.DataFrame()
        self.emg = pd.DataFrame()
        self.move_forward = 0.0
        self.previous_end_position = 0
        self.monve_forward_list = []

        self.load_csv()
        self.adjust_timing()

    def load_csv(self):
            self.emg = pd.read_csv(os.path.join(self.Dataname)).drop(['moving', 'characteristic_num'], axis=1)
            self.emg = self.emg.rename({'timestamp': 'timestamp_Under'}, axis='columns')
            print(f'emg {self.Dataname} has been loaded.')

    def adjust_timing(self):
        first_timestamp = self.emg['timestamp_Under'].iloc[0]
        self.emg['timestamp_Under'] -= first_timestamp
        self.adjusted_data = self.emg.copy()
        return self.adjusted_data

    def set_base_time(self, action_num, move_forward):

        self.monve_forward = move_forward

        if action_num == 1:
            base_time = 5 - move_forward
            start_position = 0
            end_position = base_time + 100

            start_time = 5 - move_forward
            end_time = start_time + 95

        if action_num == 2:
            start_position = self.previous_end_position
            end_position = start_position + 155

            start_time = start_position + (5 + move_forward)
            end_time = start_time + 145

        if action_num == 3:
            start_position = self.previous_end_position
            end_position = start_position + 105

            start_time = start_position + (5 + move_forward)
            end_time = start_time + 95

        if action_num == 4:
            start_position = self.previous_end_position
            end_position = self.adjusted_data['timestamp_Under'].max()

            end_time = end_position - move_forward
            start_time = end_time -145

        print(f"data renge: {start_position} s ～ {end_position} s")

        action = self.adjusted_data.loc[(self.adjusted_data['timestamp_Under'] >= start_position) & (self.adjusted_data['timestamp_Under'] <= end_position), :].copy()
        if action.empty:
            print(f"Warning: data ({start_position}～{end_position} s）is empty.")
            return

        print(f"Number of data-points: {len(action)}")
        print(f"Actual time range: {action['timestamp_Under'].min():.1f} s ～ {action['timestamp_Under'].max():.1f} s")

        next_start_posision = action['timestamp_Under'].max() + 5
        idx = (np.abs(self.adjusted_data['timestamp_Under'] - next_start_posision)).idxmin()
        closest_data = self.adjusted_data.loc[idx]
        closest_time = closest_data['timestamp_Under']

        print(f"next_start_position: {closest_time} s")

        self.previous_end_position = closest_time

        start_idx = (action['timestamp_Under'] - start_time).abs().idxmin()
        start_closest_data = self.adjusted_data.loc[start_idx]
        start_closest_time = start_closest_data['timestamp_Under']

        end_idx = (action['timestamp_Under'] - end_time).abs().idxmin()
        end_closest_data = self.adjusted_data.loc[end_idx]
        end_closest_time = end_closest_data['timestamp_Under']

        print(f"start_time: {start_closest_time} s")
        print(f"end_time: {end_closest_time} s")

        self.action = action

        return self.action, self.monve_forward


def visualize_action(emg_dataset, action_num, figsize=(15, 10)):
    if not hasattr(emg_dataset, 'action') or emg_dataset.action.empty:
        print("Error: action data is empty.")
        return

    action = emg_dataset.action
    move_forward = emg_dataset.monve_forward

    # EMG列を特定（timestamp_Under以外の数値列）
    emg_columns = [col for col in action.columns if col != 'timestamp_Under' and action[col].dtype in ['float64', 'int64']]

    if len(emg_columns) == 0:
        print("Error: EMG signal sequence not found.")
        return

    n_points = len(action)
    data_points = range(n_points)
    n_channels = len(emg_columns)

    # プロット作成
    fig, axes = plt.subplots(n_channels, 1, figsize=figsize, sharex=True)
    if n_channels == 1:
        axes = [axes]

    # 各EMGチャンネルをプロット
    for idx, (ax, col) in enumerate(zip(axes, emg_columns)):
        ax.plot(data_points, action[col].values, linewidth=0.5, color='blue')
        ax.set_ylabel(f'{col}', fontsize=10)
        ax.grid(True, alpha=0.3)

        if move_forward is not None:
            min_time = action['timestamp_Under'].min()
            max_time = action['timestamp_Under'].max()

            if action_num == 1:
                start_time = 5 - move_forward
                current_time = start_time
                line_count = 0

                while current_time <= max_time:
                    time_diff = (action['timestamp_Under'] - current_time).abs()
                    closest_idx = time_diff.idxmin()
                    closest_data_point = action.index.get_loc(closest_idx)

                    ax.axvline(x=closest_data_point, color='red', linestyle='-', linewidth=2, alpha=0.7)

                    current_time += 5
                    line_count += 1

                if idx == 0 and line_count > 0:
                    ax.legend(loc='upper right')
                    print(f"start time: {start_time:.1f}s, interval: 5 s")

            if action_num == 2:
                start_time = min_time + (5 + move_forward)
                current_time = start_time
                line_count = 0
                intervals = [2.5, 5.0]

                while current_time <= max_time:
                    time_diff = (action['timestamp_Under'] - current_time).abs()
                    closest_idx = time_diff.idxmin()
                    closest_data_point = action.index.get_loc(closest_idx)

                    ax.axvline(x=closest_data_point, color='red', linestyle='-', linewidth=2, alpha=0.7)

                    interval = intervals[line_count % 2]
                    current_time += interval
                    line_count += 1

            if action_num == 3:
                start_time = min_time + (5 + move_forward)
                current_time = start_time
                line_count = 0

                while current_time <= max_time:
                    time_diff = (action['timestamp_Under'] - current_time).abs()
                    closest_idx = time_diff.idxmin()
                    closest_data_point = action.index.get_loc(closest_idx)

                    ax.axvline(x=closest_data_point, color='red', linestyle='-', linewidth=2, alpha=0.7)

                    current_time += 5
                    line_count += 1

                if idx == 0 and line_count > 0:
                    ax.legend(loc='upper right')
                    print(f"start time: {start_time:.1f}s, interval: 5 s")

            if action_num == 4:
                end_time = max_time - move_forward
                current_time = end_time
                line_count = 0
                intervals = [2.5, 5.0]

                while current_time >= min_time and line_count < 40:
                    time_diff = (action['timestamp_Under'] - current_time).abs()
                    closest_idx = time_diff.idxmin()
                    closest_data_point = action.index.get_loc(closest_idx)

                    ax.axvline(x=closest_data_point, color='red', linestyle='-', linewidth=2, alpha=0.7)

                    interval = intervals[line_count % 2]
                    current_time -= interval
                    line_count += 1

        # タイトル（最初のサブプロットのみ）
        if idx == 0:
            time_range = f"{action['timestamp_Under'].min():.1f}~{action['timestamp_Under'].max():.1f} s"
            title = f'EMG signal (data-point num: {n_points}, time-range: {time_range})'
            if move_forward is not None:
                title += f' [5-move_forward: {5-move_forward:.1f}s]'
            ax.set_title(title, fontsize=12, fontweight='bold')

    # x軸ラベル
    axes[-1].set_xlabel('data point', fontsize=11)
    axes[-1].set_xlim(0, n_points-1)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Load data
    Subject = 2
    path_pattern = os.path.join('../../data/raw', f'Sub{Subject}_*', '1013', '*_emg.csv')
    FileList = sorted(glob.glob(path_pattern))
    print(FileList)
    print(f'file_num: {len(FileList)}')

    section = EMGdataset(FileList[1])
    max_timestamp = section.adjusted_data['timestamp_Under'].max()
    print(f"timestamp_Underの最大値: {max_timestamp}")

    # Action1
    action1 = section.set_base_time(1, 1.75)
    visualize_action(section, 1)

    # Action2
    action2 = section.set_base_time(2, 0.8)
    visualize_action(section, 2)

    # Action3
    action3 = section.set_base_time(3, 1.8)
    visualize_action(section, 3)

    # Action4
    action4 = section.set_base_time(4, 0)
    visualize_action(section, 4)
