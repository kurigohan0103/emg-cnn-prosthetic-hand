import argparse
import tkinter as tk
from tkinter import messagebox
import math
import os
import subprocess
import time
from datetime import datetime


TICK_MS = 100 # タイマー更新間隔(100ms)
PREPARE_SECONDS = 5 * 10 # 準備時間（5秒）
WORK_SECONDS = 5 * 10 # Task1,3
WORK_UP_SECONDS = int(2.5 * 10) # Task2,4
WORK_DOWN_SECONDS = int(2.5 * 10) # Task2,4
WORK_MID_REST_SECONDS = 5 * 10 # WORK-UPとWORK-DOWNの間の休憩
REST_SECONDS = 5 * 10
BREAK_SECONDS = 15 * 10
TOTAL_TASKS = 4
TOTAL_SETS = 10

timer_id = None
is_running = False
current_task = 1
current_set = 1
current_phase = "Idle"
remaining_time = WORK_SECONDS

# 高精度タイマー用の変数
phase_start_time = None  # 各フェーズの開始実時間
phase_duration = None    # 各フェーズの予定実行時間（秒）
session_start_time = None

# EMGセンサー関連の変数
myo_process = None
is_recording = False


class MyoIntegratedTimer:
    def __init__(self, root, args):
        self.root = root
        self.setup_variables(args)
        self.setup_ui()
        self.update_display()
        
    def setup_variables(self, args):
        global timer_id, is_running, current_task, current_set, current_phase, remaining_time
        global myo_process, is_recording, session_start_time

        # プロジェクトルートからの相対パスで出力ディレクトリを構築
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        self.output_dir = os.path.join(project_root, "data", "raw", f"Sub{args.subject}", args.date)
        self.myo_logger_path = os.path.join(script_dir, "myo-logger.py")
        
    def setup_ui(self):
        self.root.title("EMGデータ取得タイマー")
        self.root.geometry("500x500")
        
        # タイマーUI
        self.setup_timer_ui()
    
    # タイマーUIをセットアップ
    def setup_timer_ui(self):
        global phase_label, time_label, phase_canvas, foreground_bar
        global status_label, start_button, reset_button
        
        # フェーズ表示ラベル
        phase_label = tk.Label(self.root, text="待機中", font=("Helvetica", 32, "bold"))
        phase_label.pack(pady=(20,10))

        # 残り時間表示ラベル
        time_label = tk.Label(self.root, text="--", font=("Helvetica", 90))
        time_label.pack(pady=5)

        # プログレスバー
        CANVAS_HEIGHT = 25
        phase_canvas = tk.Canvas(self.root, height=CANVAS_HEIGHT, bg="#D3D3D3", highlightthickness=0)
        foreground_bar = phase_canvas.create_rectangle(0, 0, 0, CANVAS_HEIGHT, fill="#007aff", outline="")
        phase_canvas.pack(pady=5, fill='x', padx=20)

        # 状態表示ラベル
        status_label = tk.Label(self.root, text="STARTボタンを押してください", font=("Helvetica", 14))
        status_label.pack(pady=10)

        # 実行時間表示
        self.runtime_label = tk.Label(self.root, text="実行時間: 0.0秒", font=("Helvetica", 12), fg="blue")
        self.runtime_label.pack(pady=5)

        # ボタン
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=20)
        start_button = tk.Button(button_frame, text="START", font=("Helvetica", 16), command=self.start_timer)
        start_button.pack(side="left", padx=15, ipadx=10, ipady=5)
        reset_button = tk.Button(button_frame, text="RESET", font=("Helvetica", 16), command=self.reset_timer)
        reset_button.pack(side="left", padx=15, ipadx=10, ipady=5)

    # 各フェーズの実行時間（秒）を返す
    def get_phase_duration_seconds(self, phase):
        if phase == "Prepare": return PREPARE_SECONDS / 10
        if phase == "Work-UP": return WORK_UP_SECONDS / 10
        if phase == "Work-Mid-Rest": return WORK_MID_REST_SECONDS / 10
        if phase == "Work-DOWN": return WORK_DOWN_SECONDS / 10
        if phase == "Work": return WORK_SECONDS / 10
        if phase == "Rest": return REST_SECONDS / 10
        if phase == "Break": return BREAK_SECONDS / 10
        return 1

    # 新しいフェーズを実時間ベースで開始
    def start_phase(self, new_phase):
        global phase_start_time, phase_duration, current_phase
        current_phase = new_phase
        phase_start_time = time.time()
        phase_duration = self.get_phase_duration_seconds(new_phase)
        print(f"フェーズ開始: {new_phase} (予定時間: {phase_duration}秒) - Task{current_task} Set{current_set}")

    # EMGセンサー記録開始 
    def start_myo_recording(self):
        global myo_process, is_recording
        
        if is_recording:
            return
            
        try:
            # myo-logger.pyのコマンドを構築
            cmd = ["python", self.myo_logger_path]
            
            # 固定出力ディレクトリを使用
            cmd.extend(["-o", self.output_dir])
            
            # プロセス開始
            myo_process = subprocess.Popen(cmd)
            is_recording = True
            
            print(f"記録開始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"出力ディレクトリ: {self.output_dir}")
            
        except Exception as e:
            messagebox.showerror("エラー", f"記録開始に失敗しました: {e}")

    # 記録停止
    def stop_myo_recording(self):
        global myo_process, is_recording
        
        if not is_recording or not myo_process:
            return
            
        try:
            myo_process.terminate()
            myo_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            myo_process.kill()
            myo_process.wait()
        except:
            pass
            
        is_recording = False
        myo_process = None
        
        print(f"記録停止: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # タイマー更新（実時間ベース） 
    def update_timer(self):
        global current_phase, current_set, current_task, phase_start_time, phase_duration
        
        if not is_running:
            return
            
        # 現在の実時間での経過時間を計算
        current_time = time.time()
        elapsed = current_time - phase_start_time
        remaining_seconds = max(0, phase_duration - elapsed)
        
        # 表示用に10倍
        global remaining_time
        remaining_time = int(remaining_seconds * 10)
        
        if remaining_seconds <= 0:
            # フェーズ完了処理
            old_phase = current_phase
            print(f"フェーズ完了: {old_phase}")
            
            if current_phase == "Prepare":
                # Task 1の最初のみPrepareフェーズ
                if current_task in [2, 4]:
                    self.start_phase("Work-UP")
                else:
                    self.start_phase("Work")
            elif current_phase == "Work-UP":
                self.start_phase("Work-Mid-Rest")
            elif current_phase == "Work-Mid-Rest":
                self.start_phase("Work-DOWN")
            elif current_phase in ["Work", "Work-DOWN"]:
                # セット完了
                if current_set < TOTAL_SETS:
                    current_set += 1
                    self.start_phase("Rest")
                else:
                    # 10セット完了 → 次のタスクまたは終了（Restなし）
                    if current_task < TOTAL_TASKS:
                        current_task += 1
                        current_set = 1
                        self.start_phase("Break")
                    else:
                        current_phase = "Finished"
                        self.finish_app()
                        return
            elif current_phase in ["Rest", "Break"]:
                if current_phase == "Break":
                    print(f"Task {current_task-1} 完了 -> Task {current_task} 開始")
                    # Break後は直接Work/Work-UPに移行（Prepareなし）
                    if current_task in [2, 4]:
                        self.start_phase("Work-UP")
                    else:
                        self.start_phase("Work")
                else:
                    # Rest後は同じタスクの次のセットなので、直接Workに移行
                    if current_task in [2, 4]:
                        self.start_phase("Work-UP")
                    else:
                        self.start_phase("Work")
        
        self.update_display()
        timer_id = self.root.after(50, self.update_timer)  # より高頻度で更新

    def start_timer(self):
        global is_running, session_start_time
        if is_running: 
            return
        
        is_running = True
        session_start_time = time.time()  # 実時間で記録
        start_button.config(state="disabled")
        
        self.start_phase("Prepare")  # 実時間ベースでフェーズ開始
        
        # START時に自動で記録開始
        self.start_myo_recording()
        
        self.update_display()
        self.update_timer()
        
        print(f"タイマー開始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    def reset_timer(self):
        global is_running, timer_id, current_task, current_set, current_phase, remaining_time
        
        # 記録中の場合は停止
        if is_recording:
            self.stop_myo_recording()
            
        if timer_id:
            self.root.after_cancel(timer_id)
            timer_id = None
        is_running = False
        current_task = 1
        current_set = 1
        
        current_phase = "Idle"
        remaining_time = WORK_SECONDS
        
        start_button.config(state="normal")
        self.update_display()
        print("タイマーリセット")

    # 理論的な総実行時間を計算
    def calculate_theoretical_time(self):
        total = 0
        
        # Prepare (最初のタスクのみ)
        total += PREPARE_SECONDS / 10  # 5秒
        
        # 各タスクの時間計算
        for task in range(1, TOTAL_TASKS + 1):
            if task in [2, 4]:  # Task 2, 4
                # Work-UP + Work-Mid-Rest + Work-DOWN のセット
                work_time = (WORK_UP_SECONDS + WORK_MID_REST_SECONDS + WORK_DOWN_SECONDS) / 10
            else:  # Task 1, 3
                # Work のセット
                work_time = WORK_SECONDS / 10
            
            # セット間のRest (9回、最後のセットにはRestなし)
            rest_time = REST_SECONDS / 10 * (TOTAL_SETS - 1)
            
            # このタスクの合計時間
            task_total = work_time * TOTAL_SETS + rest_time
            total += task_total
        
        # タスク間のBreak (3回)
        total += BREAK_SECONDS / 10 * (TOTAL_TASKS - 1)
        
        return total

    def finish_app(self):
        global is_running, timer_id, session_start_time
        
        # 自動で記録停止
        if is_recording:
            self.stop_myo_recording()
            
        is_running = False
        if timer_id:
            self.root.after_cancel(timer_id)
            timer_id = None
        start_button.config(state="normal")
        self.update_display()
        
        # 完了メッセージ
        if session_start_time:
            total_time_real = time.time() - session_start_time
            theoretical_time = self.calculate_theoretical_time()
            error = total_time_real - theoretical_time
            
            print(f"タイマー完了: 実行時間 {total_time_real:.6f}秒")
            print(f"理論値: {theoretical_time:.6f}秒")
            print(f"誤差: {error:.6f}秒")
            
            messagebox.showinfo("完了", 
                               f"全てのタスクが完了しました！\n\n"
                               f"実行時間: {total_time_real:.3f}秒\n"
                               f"理論値: {theoretical_time:.3f}秒\n"
                               f"誤差: {error:.3f}秒 ({error*1000:.1f}ms)\n\n")
 
    def get_max_time(self, phase):
        if phase == "Prepare": return PREPARE_SECONDS
        if phase == "Work-UP": return WORK_UP_SECONDS
        if phase == "Work-Mid-Rest": return WORK_MID_REST_SECONDS
        if phase == "Work-DOWN": return WORK_DOWN_SECONDS
        if phase == "Work": return WORK_SECONDS
        if phase == "Rest": return REST_SECONDS
        if phase == "Break": return BREAK_SECONDS
        return 1

    # 画面の表示を現在の状態に合わせて更新
    def update_display(self):
        # フェーズテキストの決定
        if current_phase == "Idle":
            phase_text = "待機中"
        elif current_phase == "Prepare":
            phase_text = "準備中"
        elif current_phase == "Work":
            phase_text = "WORK"
        elif current_phase == "Work-UP":
            phase_text = "WORK-UP"
        elif current_phase == "Work-Mid-Rest":
            phase_text = "REST"
        elif current_phase == "Work-DOWN":
            phase_text = "WORK-DOWN"
        elif current_phase == "Rest":
            phase_text = "REST"
        elif current_phase == "Break":
            phase_text = "BREAK"
        elif current_phase == "Finished":
            phase_text = "完了！"
        else:
            phase_text = f"不明: {current_phase}"
        
        phase_label.config(text=phase_text)

        if current_phase not in ["Idle", "Finished"]:
            # Work-UP / Work-DOWN の場合 → 分:秒:ミリ秒 表示
            if current_phase in ["Work-UP", "Work-DOWN"]:
                total_seconds = remaining_time // 10
                minutes = total_seconds // 60
                seconds = total_seconds % 60
                tenths = remaining_time % 10
                time_text = f"{minutes:02d}:{seconds:02d}:{tenths}"
            # それ以外のフェーズの場合 → 分:秒 表示
            else:
                total_seconds = math.ceil(remaining_time / 10)
                minutes = total_seconds // 60
                seconds = total_seconds % 60
                time_text = f"{minutes:02d}:{seconds:02d}"
        else:
            # 待機中や完了時は 00:00 形式で表示
            time_text = "00:00"
        
        time_label.config(text=time_text)
        
        status_text = (f"Task: {current_task}/{TOTAL_TASKS}   Set: {current_set}/{TOTAL_SETS}" 
                      if current_phase not in ["Idle", "Finished"] 
                      else "STARTボタンを押してください")
        status_label.config(text=status_text)

        # 実行時間の表示
        if session_start_time and is_running:
            runtime = time.time() - session_start_time
            self.runtime_label.config(text=f"実行時間: {runtime:.1f}秒")
        else:
            self.runtime_label.config(text="実行時間: 0.0秒")

        # プログレスバーの更新ロジック
        progress_percentage = 0

        if is_running and current_phase != "Finished":
            max_time = self.get_max_time(current_phase)
            progress_percentage = (max_time - remaining_time) / max_time * 100
        elif current_phase == "Finished":
            progress_percentage = 100

        canvas_width = phase_canvas.winfo_width()
        bar_width = canvas_width * (progress_percentage / 100)
        phase_canvas.coords(foreground_bar, 0, 0, bar_width, 25)

    # アプリケーション終了時の処理
    def on_closing(self):
        if is_recording:
            self.stop_myo_recording()
        self.root.destroy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EMGデータ取得タイマー")
    parser.add_argument("-s", "--subject", type=int, required=True,
                        help="被験者番号 (ex: 2)")
    parser.add_argument("-d", "--date", type=str, required=True,
                        help="日付 (ex: 1017)")
    args = parser.parse_args()

    root = tk.Tk()
    app = MyoIntegratedTimer(root, args)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()