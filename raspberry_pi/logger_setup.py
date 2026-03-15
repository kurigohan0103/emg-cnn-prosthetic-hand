import logging
from logging.handlers import RotatingFileHandler
import config


def setup_logger():
    logger = logging.getLogger('Myoelectric_Prosthesis')
    logger.setLevel(getattr(logging, config.LOG_LEVEL))
    
    file_handler = RotatingFileHandler(
        config.LOG_FILE,
        maxBytes=1000000,
        backupCount=3
    )
    file_handler.setLevel(logging.DEBUG)
    
    # コンソールハンドラ
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # フォーマット設定
    formatter = logging.Formatter(
        '%(asctime)s [%(threadName)s] %(levelname)s: %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger