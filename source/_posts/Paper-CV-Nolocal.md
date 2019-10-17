---
title: Paper_CV_Nolocal
date: 2019-08-29 11:49:51
tags:
- CV
- attention
---

![1567051217525](Paper-CV-Nolocal/1567051217525.png)

[official github_facebook](https://github.com/facebookresearch/video-nonlocal-net?utm_source=mybridge&utm_medium=blog&utm_campaign=read_more)

本质就是输出的每个位置值都是其他所有位置的加权平均值，通过softmax操作可以进一步突出共性。

<!-- more -->

[nonlocal_helper.py](https://github.com/facebookresearch/video-nonlocal-net/blob/master/lib/models/nonlocal_helper.py)

[resnet_help.py](https://github.com/facebookresearch/video-nonlocal-net/blob/master/lib/models/resnet_helper.py)



$y = \text{softmax}(X^T W_θ^T W_φ X)g(X)$  公式来源：

49《Attention is all you need 》NIPS 2017 [arxiv](https://arxiv.org/abs/1706.03762)



### Related Work：

IN（Interaction Networks）互动网络

《Interaction networks for learning about objects, relations and physics 》[arxiv](https://arxiv.org/abs/1612.00222)



VIN 视觉互动网络 DeepMind

​	https://arxiv.org/pdf/1706.01433.pdf

​	VIN 由两大机制组成：一个视觉模块和一个现实推理模块（physical reasoning module）。结合在一起，VIN 的两大模块能够处理一段视觉场景，并且预测其中每个不同物体在现实物理规律下会发生的情况。



CV_BASE:

《A non-local algorithm for image denoising.》2005 CVPR

