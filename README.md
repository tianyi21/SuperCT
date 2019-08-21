# SuperCT
*A PyTorch Implementation of SuperCT for Single-Cell Clustering*

*Ⓒ Copyright Tianyi Liu 2019, All Rights Reserved.*

Latest Update: 15:50 Aug 20th, 2019 CST

💥**If you use or partially use this repo, you shall formally - in report or presentation - acknowledge this repo.**

## Introduction
SuperCT was initially published by [Xie et al.](https://academic.oup.com/nar/article/47/8/e48/5364134), in Nucleic Acids Research Vol. 47, 2019. This repo presents a modified version of SuperCT for supervised single-cell clustering. We have tested B, M, and T cells clustering with the implemented model on an industrial dataset and achieved 99.28% accuracy during testing.

### Structure of this repo
`source` contains all the source files

`img`    contains the images used in this README.md

### Usage
Input: *Binary single-cells expression matrices*

`trainvalnet.py` for training and validation

`testnet.py` for testing after training

`cfg.py` contains all hyperparameters, **Make sure to modify this before you use this repo**

### Package Requirements
matplotlib

pickle

PyTorch

scikit-learn

Numpy

## Description
<div align="center"><img src="https://github.com/evanliuty/SuperCT/blob/master/img/net.png" width="80%"></div>

The model was modified as shown above and implemented in PyTorch v1.0.
<div align="center">
  <img src="https://github.com/evanliuty/SuperCT/blob/master/img/pr.png" width="38%">
  <img src="https://github.com/evanliuty/SuperCT/blob/master/img/tsne.png" width="58%">
</div>
With ~4k8 cells of three categories, this repo achieved 99.28% accuracy and 0.9965 mAP when testing.
