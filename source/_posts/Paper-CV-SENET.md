---
title: Paper_CV_SENET
date: 2019-08-29 11:45:58
tags:
---

我们可以看到，已经有很多工作在空间维度上来提升网络的性能。那么很自然想到，网络是否可以从其他层面来考虑去提升性能，比如考虑特征通道之间的关系？我们的工作就是基于这一点并提出了 **Squeeze-and-Excitation Networks（简称 SENet）。在我们提出的结构中，Squeeze 和 Excitation 是两个非常关键的操作**，所以我们以此来命名。我们的动机是希望显式地建模特征通道之间的相互依赖关系。另外，我们并不打算引入一个新的空间维度来进行特征通道间的融合，而是采用了一种全新的「特征重标定」策略。具体来说，**就是通过学习的方式来自动获取到每个特征通道的重要程度，然后依照这个重要程度去提升有用的特征并抑制对当前任务用处不大的特征**。





![img](Paper-CV-SENET/247d198e8ef64a7fa040887b6f0ee0e0_th.jpg)

Squeeze：挤，榨

Excitation：激发

<!-- more -->

这里我们使用 global average pooling 作为 Squeeze 操作。紧接着两个 Fully Connected 层组成一个 Bottleneck 结构去建模通道间的相关性，并输出和输入特征同样数目的权重。我们首先将特征维度降低到输入的 1/16，然后经过 ReLu 激活后再通过一个 Fully Connected 层升回到原来的维度。这样做比直接用一个 Fully Connected 层的好处在于：

1）具有更多的非线性，可以更好地拟合通道间复杂的相关性；

2）极大地减少了参数量和计算量。然后通过一个 Sigmoid 的门获得 0~1 之间归一化的权重，最后通过一个 Scale 的操作来将归一化后的权重加权到每个通道的特征上。





实验于思考：

- 1、从核心思路上来说，抑制特征，从实验过程中，不适合ResNet-DDZ的模型收敛。
- 2、global average pooling 不适合DDZ/GO的Squeeze