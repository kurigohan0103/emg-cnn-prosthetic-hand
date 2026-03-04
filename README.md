# EMG-Based Prosthetic Hand Control with CNN

Real-time prosthetic hand control system using CNN classification of surface EMG signals.

CNN（畳み込みニューラルネットワーク）による表面筋電位（EMG）信号の分類と、リアルタイムアシスト装置制御システム。

## 研究概要

Myo Armbandから取得した8チャンネルのEMG信号をCNNで6クラスに分類し、分類結果に基づいてモーターをリアルタイムに制御するシステムを構築した。モデルの学習はPC上で行い、学習済みモデルをRaspberry Pi 5に転送してリアルタイム推論を実行する。

### 分類クラス

| クラス | 動作 |
|--------|------|
| Class 1 | 1kg_Stationary |
| Class 2 | 1kg_Dynamic_up |
| Class 3 | 1kg_Dynamic_down |
| Class 4 | 3kg_Stationary |
| Class 5 | 3kg_Dynamic_up |
| Class 6 | 3kg_Dynamic_down |

## ディレクトリ構成

```
.
├── data_collection/          # EMGデータ収集（Myo Armband）
│   ├── timer-with-myologger.py  # タイマー付きデータ収集GUI
│   ├── myo-logger.py            # Myoデータロガー
│   ├── myoraw.py                # Myo Armband通信プロトコル
│   ├── bled112.py               # BLED112 Bluetoothバックエンド
│   └── consumerpool.py          # データハンドラスレッドプール
│
├── training/                # モデル学習・評価（PC側）
│   ├── main.py              # メインスクリプト（学習・評価実行）
│   ├── config.py            # ハイパーパラメータ・パス設定
│   ├── check_data_structure.py  # データ構造確認ツール
│   ├── data/
│   │   ├── emg_dataset.py   # EMGデータの読み込み・前処理・ウィンドウ分割
│   │   ├── dataloader.py    # PyTorch Dataset クラス
│   │   └── data_split.py    # 層化分割（被験者×タスク）
│   ├── models/
│   │   └── cnn.py           # EMGNet（CNNモデル定義）
│   ├── training/
│   │   ├── trainer.py       # 学習ループ
│   │   ├── evaluator.py     # 評価（精度・混同行列・F1等）
│   │   └── model_saver.py   # モデル保存・早期終了
│   ├── data_processing/
│   │   ├── triming_size_checker.py  # セグメント長の確認ツール
│   │   └── window_size_analyzer.py  # ウィンドウサイズ分析
│   ├── experiments/
│   │   └── multiple_runs.py # 複数回実験の管理
│   └── utils/
│       ├── seed.py          # 乱数シード固定
│       ├── file_utils.py    # ファイル読み込みユーティリティ
│       ├── visualization.py # 混同行列・学習曲線の可視化
│       └── stratified_utils.py  # 層化分割ユーティリティ
│
├── hardware/                # ハードウェア設計（KiCad）
│   └── myoelectric/           # 基板設計データ
│
├── raspberry_pi/            # リアルタイム推論・制御（Raspberry Pi側）
│   ├── main.py              # メインスクリプト（システム起動）
│   ├── config.py            # GPIO・モデル・制御パラメータ設定
│   ├── inference.py         # 推論エンジン（EMGNet + フィルタ処理）
│   ├── inference_thread.py  # 推論スレッド
│   ├── control_thread.py    # モーター制御スレッド
│   ├── myo_thread.py        # Myoデータ受信スレッド（リングバッファ）
│   ├── motor_control.py     # モーター・ポテンショメーター制御
│   ├── led_control.py       # LED制御（状態表示）
│   ├── hardware_test.py     # ハードウェアテスト
│   ├── test_inference.py    # 推論テスト
│   ├── logger_setup.py      # ログ設定
│   ├── myoraw.py            # Myo Armband通信プロトコル
│   ├── bled112.py           # BLED112 Bluetoothバックエンド
│   └── consumerpool.py      # データハンドラスレッドプール
│
├── requirements.txt
└── .gitignore
```

## 使用技術

| カテゴリ | 技術 |
|----------|------|
| 言語 | Python 3.11 |
| 深層学習 | PyTorch |
| 信号処理 | SciPy（バンドパスフィルタ、ノッチフィルタ） |
| データ処理 | NumPy, pandas |
| 可視化 | Matplotlib, Seaborn |
| 評価 | scikit-learn（精度、F1、混同行列） |
| ハードウェア | Raspberry Pi 5, Myo Armband |
| モーター制御 | gpiozero, spidev |
| BLE通信 | pyserial + BLED112 |

## セットアップ

### 学習環境（PC）

```bash
git clone https://github.com/<username>/emg-cnn-prosthetic-hand.git
cd emg-cnn-prosthetic-hand

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### データの準備

EMGデータは以下の形式で `data/` ディレクトリに配置してください:

```
data/
├── raw/
│   └── Sub<N>/
│       └── <date>/
│           ├── <timestamp>_emg.csv
│           └── <timestamp>_imu.csv
└── processed/
    └── Sub<N>/
        └── CNN/
            ├── <timestamp>_emg.csv
            └── <timestamp>_timing.csv
```

### 学習の実行

```bash
cd training
python main.py
```

`config.py` で主要なパラメータを変更できます:

| パラメータ | デフォルト値 | 説明 |
|-----------|-------------|------|
| `SUBJECTS` | `[2]` | 使用する被験者番号 |
| `WINDOW_SIZE` | `434` | ウィンドウサイズ（サンプル数） |
| `BATCH_SIZE` | `8` | バッチサイズ |
| `EPOCHS` | `100` | エポック数 |
| `LEARNING_RATE` | `0.001` | 学習率 |
| `TRAIN_RATIO` | `0.75` | 訓練データの割合 |

### Raspberry Pi でのリアルタイム推論

```bash
# Raspberry Pi上で
cd raspberry_pi
pip install torch numpy scipy gpiozero spidev pyserial

# 学習済みモデルを配置
mkdir -p models
cp <path_to_model>/model_weight.pth models/

# システム起動
python main.py
```

## ハードウェア構成

### Raspberry Pi GPIO割り当て

| GPIO | 機能 | config.py |
|------|------|-----------|
| GPIO 3 | 起動ボタン | BUTTON_START |
| GPIO 4 | 停止ボタン | BUTTON_STOP |
| GPIO 17 | ステータスLED | LED_STATUS |
| GPIO 27 | Task 1-3 LED | LED_TASK_1_3 |
| GPIO 22 | Task 4-6 LED | LED_TASK_4_6 |
| GPIO 13 | モーター PWM | MOTOR_PWM_PIN |
| GPIO 19 | モーター DIR | MOTOR_DIR_PIN |
| SPI0 CH0 | ポテンショメーター | POT_CHANNEL |

## License

The Myo communication libraries (`myoraw.py`, `bled112.py`, `consumerpool.py`) are licensed under the MIT License.
See the header of each file for details.
