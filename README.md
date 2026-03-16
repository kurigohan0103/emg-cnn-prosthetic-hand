# EMG-Based Prosthetic Hand Control with CNN

CNN（畳み込みニューラルネットワーク）による表面筋電位（EMG）信号の分類と、リアルタイムアシスト装置制御システム。

## 概要

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
├── data_collection/   # EMGデータ収集（Myo Armband + タイマーGUI）
├── training/          # モデル学習・評価
│   ├── data/          # データ読み込み・前処理・分割
│   ├── models/        # CNNモデル定義
│   ├── training/      # 学習ループ・評価・モデル保存
│   └── utils/
├── raspberry_pi/      # リアルタイム推論・モーター制御
├── hardware/          # 基板設計データ（KiCad）
└── requirements.txt
```

## 使用技術

| カテゴリ | 技術 |
|----------|------|
| 言語 | Python 3.11 |
| 深層学習 | PyTorch |
| 信号処理 | SciPy（バンドパスフィルタ、ノッチフィルタ） |
| データ処理 | NumPy, pandas |
| 可視化 | Matplotlib, Seaborn |
| 評価 | scikit-learn |
| ハードウェア | Raspberry Pi 5, Myo Armband |

## セットアップ

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

### 学習パラメータ

`config.py` で主要なパラメータを変更できます:

| パラメータ | デフォルト値 | 説明 |
|-----------|-------------|------|
| `SUBJECTS` | `[2]` | 使用する被験者番号 |
| `WINDOW_SIZE` | `434` | ウィンドウサイズ（サンプル数） |
| `BATCH_SIZE` | `8` | バッチサイズ |
| `EPOCHS` | `100` | エポック数 |
| `LEARNING_RATE` | `0.001` | 学習率 |
| `TRAIN_RATIO` | `0.75` | 訓練データの割合 |

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
