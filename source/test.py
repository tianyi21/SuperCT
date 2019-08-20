#   Developed by LIU Tianyi on 2019-08-12
#   Copyright(c) 2019. All Rights Reserved.

import pickle as pkl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_recall_curve, average_precision_score

with open('./test_cache/y_test.pkl', 'rb') as f:
    label = pkl.load(f)
print(label)