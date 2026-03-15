import random
import numpy as np
import torch
import os

# すべての乱数を固定
def set_all_seeds(seed):
    # Python標準の乱数
    random.seed(seed)
    
    # NumPyの乱数
    np.random.seed(seed)
    
    # PyTorchの乱数（CPU）
    torch.manual_seed(seed)
    
    # PyTorchの乱数（GPU）
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # CuDNN（GPU演算の最適化）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Python標準のhash
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"All random seeds fixed: seed={seed}")