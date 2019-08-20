#   Developed by LIU Tianyi on 2019-08-11
#   Copyright(c) 2019. All Rights Reserved.

import torch.nn as nn
import torch.nn.functional as F
from cfg import cls_name

class network(nn.Module):
    def __init__(self):
        super(network, self).__init__()
        self.layer_1 = nn.Sequential(
            nn.Linear(58051, 800),
            nn.ReLU(),
            nn.Dropout(p=0.4))
        self.layer_2 = nn.Sequential(
            nn.Linear(800, 400),
            nn.ReLU(),
            nn.Dropout(p=0.4))
        self.layer_3 = nn.Sequential(
            nn.Linear(400, 100),
            nn.ReLU(),
            nn.Dropout(p=0.4))
        self.clf = nn.Sequential(
            nn.Linear(100, len(cls_name)),
            nn.ReLU())

    def forward(self, x):
        out = self.layer_1(x)
        out = self.layer_2(out)
        out = self.layer_3(out)
        out = self.clf(out)
        return out, F.softmax(out, dim=1)