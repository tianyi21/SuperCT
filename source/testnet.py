#   Developed by LIU Tianyi on 2019-08-14
#   Copyright(c) 2019. All Rights Reserved.

import os
import argparse
import pickle as pkl
import numpy as np
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn

from net import network
from trainvalnet import read_cache
from dataset import *
from eval import *
from cfg import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', dest='model', default="SuperCT_30_16.pth")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    params_val()

    args = parse_args()
    print('Called with Args:')
    print("CLI Args: \t\t{}".format(args))
    print("Net Params: \t{}".format(params))
    print("Class Names: \t{}".format(cls_name))
    print("Class Range: \t{}".format(cls_range))

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists('./test_cache/'):
        os.mkdir('./test_cache/')
        if not os.path.exists('./data_cache/'):
            raise Exception("Cache Not Found. STOP")

        print('Cache Found at ./data_cache/')
        x_read, y_read = read_from()
        x, y = proc_data(x_read, y_read)

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=777)

        with open('./test_cache/x_test.pkl', 'wb') as f:
            pkl.dump(x_test, f)
            print('Cache Written to ./test_cache/x_test.pkl')
        with open('./test_cache/y_test.pkl', 'wb') as f:
            pkl.dump(y_test, f)
            print('Cache Written to ./test_cache/y_test.pkl')
    else:
        print('Cache Found at ./test_cache/')
        with open('./test_cache/x_test.pkl', 'rb') as f:
            x_test = pkl.load(f)
            print('Data Loaded from Cache: ./test_cache/x_test.pkl')
        with open('./test_cache/y_test.pkl', 'rb') as f:
            y_test = pkl.load(f)
            print('Data Loaded from Cache: ./test_cache/y_test.pkl')

    model = network()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])

    checkpoint = torch.load('./save/SuperCT_30_16.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    print('Checkpoint Loaded from: {}'.format('./save/' + args.model))

    fig, ax = plt.subplots()

    model.eval()
    with torch.no_grad():
        test_cells, test_labels = torch.from_numpy(x_test).to(device).float(), torch.from_numpy(y_test).to(
            device).long()
        net_predict(model, test_cells, test_labels, ax)




