import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class EMGNet(nn.Module):
    def __init__(self, sig_size, num_classes):
        super().__init__()
        self.sig_size = sig_size
        self.num_classes = num_classes
        
        # 畳み込み層
        self.conv1 = nn.Conv2d(1, 16, (64, 1))
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 3, (1, 8), padding='same')
        self.ln = nn.BatchNorm2d(3)
        
        # 全結合層
        self.fc1 = nn.Linear(self._get_fc_input_size(), 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
    
    # 全結合層の入力サイズを計算
    def _get_fc_input_size(self):
        with torch.no_grad():
            x = torch.zeros(self.sig_size)
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = torch.flatten(x, 1)
        return x.shape[1]
    
    # 順伝播
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.ln(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x