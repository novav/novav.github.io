---
title: Paper-CV-cnn_geometric
date: 2019-11-27 17:48:34
tags:
- Geomtric

---

# CNNGeometric pytorch 中文介绍

### 论文：

I. Rocco, R. Arandjelović and J. Sivic. Convolutional neural network architecture for geometric matching. CVPR 2017 [[website](http://www.di.ens.fr/willow/research/cnngeometric/)][[arXiv](https://arxiv.org/abs/1703.05593)]

### Started:

- demo.py demonstrates the results on the ProposalFlow dataset (Proposal Flow Dataset 的示范结果)
- train.py is the main training script (训练入口)
- eval_pf.py evaluates on the ProposalFlow dataset (用于评估dataset)

### Trained models

Using Streetview-synth dataset + VGG

Using Pascal-synth dataset + VGG

Using Pascal-synth dataset + ResNet-101



Streetview: 是通过对来自东京时间机器数据集[4]的图像应用合成变换生成的，该数据集包含了东京的谷歌街景图像

Pascal: created from the training set of Pascal VOC 2011 [16]  