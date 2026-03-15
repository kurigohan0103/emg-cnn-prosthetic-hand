# GPIOピン
BUTTON_START = 3
BUTTON_STOP = 4
LED_STATUS = 17
LED_TASK_1_3 = 27
LED_TASK_4_6 = 22
MOTOR_DIR_PIN = 19
MOTOR_PWM_PIN = 13

# # ポテンショメーター
POT_CHANNEL = 0

# タイミング設定
DEBOUNCE_TIME = 0.3       # ハードウェアデバウンス
MIN_PRESS_INTERVAL = 0.5  # ソフトウェアデバウンス
LONG_PRESS_TIME = 2.0     # 長押し判定時間

# LED点滅設定
LED_BLINK_ON_TIME = 0.5     # 点灯時間
LED_BLINK_OFF_TIME = 0.5    # 消灯時間
ERROR_BLINK_INTERVAL = 0.5  # エラー時点滅間隔

# 位置設定
POS_REST = 490   # 休止位置
POS_LIFT = 695   # 持ち上げ位置

# モーター
SPEED_FAST = 60       # 高速
SPEED_SLOW = 40       # 低速
TOLERANCE = 15        # 許容誤差
MOTOR_TIMEOUT = 15.0  # タイムアウト時間

# モデル設定
MODEL_PATH = "models/model_waite.pth"
MODEL_PATH_BENCH = "models_bench/model_waite.pth"
NUM_TASKS = 6
MYO_PORT = '/dev/tty.usbmodem11'
SAMPLING_FREQ = 200.0
WINDOW_SIZE = 990
HIGHCUT = 80.0
LOWCUT = 20.0 

# 信頼度閾値設定
USE_CONFIDENCE_THRESHOLD = True
CONFIDENCE_THRESHOLD = 0.5  # 信頼度の最低ライン
MAX_RETRIES = 3             # 最大再試行回数
RETRY_INTERVAL = 0.5        # 再試行間隔

# ログ設定
LOG_FILE = "logs/system.log"
LOG_LEVEL = "DEBUG"

# デバック設定
DEBUG_INFERENCE_DELAY = 2.0
