#   Developed by LIU Tianyi on 2019-08-09
#   Copyright(c) 2019. All Rights Reserved.

import os
import argparse
import pickle as pkl
import numpy as np
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.utils.data as Data

from net import network
from dataset import *
from cfg import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', dest='base_path', default="~/Dropbox/Year\ 4\ Semester\ 3/BGI/scRNA-seq/")
    args = parser.parse_args()
    return args


def update_lr(optimizer, decay_rate):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay_rate


if __name__ == '__main__':
    params_val()

    args = parse_args()
    print('Called with Args:')
    print("CLI Args: \t\t{}".format(args))
    print("Net Params: \t{}".format(params))
    print("Class Names: \t{}".format(cls_name))
    print("Class Range: \t{}".format(cls_range))

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists('./data_cache/'):
        os.mkdir('./data_cache')

        cell_container = []
        for i in range(len(cls_name)):
            cell_container.append(exp_matrix(os.path.join(args.base_path, cls_name[i] + '.xlsx')))
        x_read, y_read = read_from(cell_container)
        x, y = proc_data(x_read, y_read)

    else:
        print('Cache Found at ./data_cache/')
        x_read, y_read = read_from()
        x, y = proc_data(x_read, y_read)

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=777)

    training_set = Data.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    val_set = Data.TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))

    model = network()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])

    train_loader = Data.DataLoader(
        dataset=training_set,
        batch_size=params['batch_size'],
        shuffle=True,
        num_workers=2)

    val_loader = Data.DataLoader(
        dataset=val_set,
        batch_size=params['batch_size'],
        shuffle=True,
        num_workers=2)

    for epoch in range(params['n_epoch']):
        print("Epoch: {}\tlr: {}".format(epoch + 1, optimizer.param_groups[0]['lr']))

        for step, (cells, labels) in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()

            cells = cells.to(device).float()
            labels = labels.to(device).long()

            outputs, _ = model(cells)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if (step + 1) % params['print_step'] == 0:
                print('Epoch: {} \tStep: {}/{:.0f}\tLoss: {:.6f}'.format(
                    epoch + 1, step + 1, np.ceil(x_train.shape[0] / params['batch_size']), loss.item()))

                model.eval()
                correct = 0
                with torch.no_grad():
                    for val_cells, val_labels in val_loader:
                        val_cells, val_labels = val_cells.to(device).float(), val_labels.to(device).long()
                        val_outputs, _ = model(val_cells)
                        _, predicted = torch.max(val_outputs.data, 1)
                        correct += (predicted == val_labels).sum().item()

                    print('\t\tValidation Accuracy: {:.4f} ({}/{})'.format(
                        100. * correct / len(val_loader.dataset), correct, len(val_loader.dataset)))

            if (step + 1 == np.ceil(x_train.shape[0] / params['batch_size'])) & ((epoch + 1) % params['save_step'] == 0):

                if not os.path.exists('./save/'):
                    os.mkdir('./save/')
                save_name = os.path.join('./save/', 'SuperCT_{}_{}.pth'.format(epoch + 1, step + 1))
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                }, save_name)
                print('save model: {}'.format(save_name))

            if (step + 1 == np.ceil(x_train.shape[0] / params['batch_size'])) & ((epoch + 1) % params['lr_decay_epoch'] == 0):
                update_lr(optimizer, params['lr_decay_rate'])
                print("LR is decayed to {}".format(optimizer.param_groups[0]['lr']))
