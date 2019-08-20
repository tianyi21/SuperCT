#   Developed by LIU Tianyi on 2019-08-14
#   Copyright(c) 2019. All Rights Reserved.

import os
import numpy as np

import torch
import torch.utils.data as Data
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from testnet import params
from cfg import *


class test_cell_class():
    def __init__(self, x_test, y_test, cls_no):
        self.x_test = np.array(x_test)
        self.y_test = np.array(y_test)
        self.cls_no = cls_no
        self._prepare_test_set()

    def _prepare_test_set(self):
        self.idx = [i for i, item in enumerate(self.y_test) if item in [self.cls_no]]
        self.label = torch.from_numpy(np.ones(len(self.idx), dtype=np.uint8) * self.cls_no).long()

    def prepare_eval_set(self, conf):
        y_eval = np.ones(self.x_test.shape[0], dtype=np.uint8)
        idx_eval = [i for i, item in enumerate(self.y_test) if item not in [self.cls_no]]
        y_eval[idx_eval] = 0
        conf_eval = conf[:, self.cls_no]
        return y_eval, conf_eval


def eval_pipeline(test_cells, test_labels, predicted, conf, cls_no, ax):
    cell = test_cell_class(test_cells, test_labels, cls_no)
    predicted_cell = predicted[cell.idx]
    correct = (predicted_cell == cell.label).sum().item()

    y_eval, conf_eval = cell.prepare_eval_set(conf)

    p, r, thres = precision_recall_curve(y_eval, conf_eval)
    ap = average_precision_score(y_eval, conf_eval)
    cell_type = (cls_no == 0) * 'B' + (cls_no == 1) * 'M' + (cls_no == 2) * 'T'

    print(str(cell_type) + ' Cell Accuracy: {:.4f}% ({}/{})\tAP: {:.4f}'.format(
        100. * correct / (predicted_cell.size(0)), correct, predicted_cell.size(0), ap))
    ax.plot(r, p, color=cls_color[cls_no], label=cls_name[cls_no] + ' AP={:.4f}'.format(ap))
    return ax, ap


# def net_predict(model, test_cells, test_labels, ax, plot_dict):
#     test_outputs, conf = model(test_cells)
#     _, predicted = torch.max(test_outputs.data, 1)
#     correct = (predicted == test_labels).sum().item()
#     print('Overall Accuracy: {:.4f}% ({}/{})'.format(
#         100. * correct / (test_labels.size(0)), correct, test_labels.size(0)))
#
#     ax = predict_pipeline(test_cells, test_labels, predicted, conf, 0, ax, plot_dict)
#     ax = predict_pipeline(test_cells, test_labels, predicted, conf, 1, ax, plot_dict)
#     ax = predict_pipeline(test_cells, test_labels, predicted, conf, 2, ax, plot_dict)
#
#     return ax

def net_predict(model, test_cells, test_labels, ax):
    testing_set = Data.TensorDataset(test_cells, test_labels)
    test_loader = Data.DataLoader(
        dataset=testing_set,
        batch_size=params['batch_size'],
        shuffle=False,
        num_workers=2)

    correct = 0
    conf = torch.Tensor()
    predicted = torch.Tensor().long()
    for step, (test_cells_batch, test_labels_batch) in enumerate(test_loader):
        print("Test Batch: {}/{}\tSize: {}".format(step + 1, len(test_loader), test_cells_batch.size(0)))
        test_outputs, test_conf = model(test_cells_batch)
        _, test_predicted = torch.max(test_outputs.data, 1)
        correct += (test_predicted == test_labels_batch).sum().item()
        conf = torch.cat((conf, test_conf))
        predicted = torch.cat((predicted, test_predicted))

    map = []
    for i in range(len(cls_name)):
        ax, ap = eval_pipeline(test_cells, test_labels, predicted, conf, i, ax)
        map.append(ap)

    print('Overall Accuracy: {:.4f} ({}/{})\tmAP: {:.4f}'.format(
        100. * correct / len(test_loader.dataset), correct, len(test_loader.dataset), np.average(map)))

    ax.legend(loc='best')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('P-R Curve')

    try:
        os.mkdir('./img/')
    except FileExistsError:
        pass
    plt.savefig('./img/prcurve.png', dpi=200)
    print('P-R Plot Saved to ./img/prcurve.png')
    plt.show()

    vis(test_cells, test_labels, predicted)

    return None


def vis(test_cells, test_labels, predicted):
    def prepare_plot_idx(test_labels, predicted, cls_no):
        return [i for i, item in enumerate(predicted) if item == cls_no], \
               [i for i, item in enumerate(test_labels) if item == cls_no]

    print("Processing with t-SNE for Visualization")
    cells_embedded = TSNE(n_components=2).fit_transform(test_cells.tolist())

    idx_all, idx_g_all = [], []
    for i in range(len(cls_name)):
        idx, idx_g = prepare_plot_idx(test_labels, predicted, i)
        idx_all.append(idx)
        idx_g_all.append(idx_g)

    fig = plt.figure(figsize=(10, 5))
    plt.subplot(121)
    for i in range(len(cls_name)):
        plt.scatter(cells_embedded[idx_all[i], 0], cells_embedded[idx_all[i], 1], s=1, c=cls_color[i], label=cls_name[i])
    plt.legend(loc='best')
    plt.title("Prediction")
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.subplot(122)
    for i in range(len(cls_name)):
        plt.scatter(cells_embedded[idx_g_all[i], 0], cells_embedded[idx_g_all[i], 1], s=1, c=cls_color[i], label=cls_name[i])
    plt.legend(loc='best')
    plt.title("Ground Truth")
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.savefig('./img/tsne.png', dpi=200)
    print('t-SNE Visualization Saved to ./img/tsne.png')
    plt.show()
