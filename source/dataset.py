#   Developed by LIU Tianyi on 2019-08-11
#   Copyright(c) 2019. All Rights Reserved.

import pandas as pd
import numpy as np
import pickle as pkl
from cfg import *

label_idx = -1


class exp_matrix():
    def __init__(self, file_path):
        self.file_path = file_path
        self.exp = self._read_data()
        self.label = self._create_label()
        assert self.exp.shape[0] == self.label.shape[0]
        self._write_cache()

    def _read_data(self):
        print('Reading: {}'.format(self.file_path.split('/')[-1]))
        if self.file_path.split('.')[-1] in ['xlsx', 'xls']:
            df = pd.read_excel(self.file_path).T
        elif self.file_path.split('.')[-1] == 'csv':
            df = pd.read_csv(self.file_path).T
        else:
            raise Exception('File type not supported. Only .xlsx, .xls, and .csv are supported.')
        df = df[1:][:]
        print('Size: {}'.format(df.shape))
        return df

    def _create_label(self):
        global label_idx
        label_idx = label_idx + 1
        if label_idx == 0:
            return np.zeros(self.exp.shape[0])
        else:
            return np.ones(self.exp.shape[0]) * label_idx

    def _write_cache(self):
        with open('./data_cache/cache_x_' + self.file_path.split('/')[-1].split('_')[0] + '.pkl', 'wb') as f:
            pkl.dump(self.exp, f)
            print('Cache Written to ./data_cache/cache_x_' + self.file_path.split('/')[-1].split('_')[0] + '.pkl')
        with open('./data_cache/cache_y_' + self.file_path.split('/')[-1].split('_')[0] + '.pkl', 'wb') as f:
            pkl.dump(self.label, f)
            print('Cache written to ./data_cache/cache_y_' + self.file_path.split('/')[-1].split('_')[0] + '.pkl')


def read_cache(data_sel):
    data_sel = data_sel.split('_')[0]
    with open('./data_cache/cache_x_' + str(data_sel) + '.pkl', 'rb') as f:
        exp = pkl.load(f)
        print('Data Loaded from Cache: ./data_cache/cache_x_' + str(data_sel) + '.pkl')

    with open('./data_cache/cache_y_' + str(data_sel) + '.pkl', 'rb') as f:
        label = pkl.load(f)
        print('Data Loaded from Cache: ./data_cache/cache_y_' + str(data_sel) + '.pkl')

    exp = np.array(exp.values.tolist())
    print('Count: {}; Dim: {}'.format(exp.shape[0], exp.shape[1]))
    assert exp.shape[0] == label.shape[0], "Data Dim {} and Label Dim {} not agree.".format(exp.shape[0], label.shape[0])
    return exp, label

def read_from(*argc):
    x_read, y_read = [], []
    if len(argc) == 1:
        for i in range(len(argc)):
            a, b = argc[0][i].exp, argc[0][i].label
            x_read.append(a)
            y_read.append(b)
    elif len(argc) == 0:
        for i in range(len(cls_name)):
            a, b = read_cache(cls_name[i])
            x_read.append(a)
            y_read.append(b)
    else:
        raise Exception('Bad Function Calling')
    return x_read, y_read


def proc_data(x_read, y_read):
    x, y = [], []
    for i in range(len(cls_range)):
        if cls_range[i] == 'all':
            x.extend(x_read[i][1:][:])
            y.extend(y_read[i][1:][:])
        else:
            x.extend(x_read[i][1:cls_range[i]][:])
            y.extend(y_read[i][1:cls_range[i]][:])

    x = np.array(x, dtype=np.uint8)
    y = np.array(y, dtype=np.uint8)

    print('Cell: {}'.format(x.shape))
    print('Label: {}'.format(y.shape))

    return x, y


