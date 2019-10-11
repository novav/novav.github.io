---
title: Paper-MLL-NN
date: 2019-10-11 16:06:33
tags:
- Multi-Label
- Neural Network
- Functional Genomics 
- Text Categorization
---

## BP_MLL

2006 《Multi-Label Neural Networks with Applications to Functional Genomics and Text Categorization》Min-Ling Zhang and Zhi-Hua Zhou. IEEE Transactions on Knowledge and Data Engineering 18, 10 (2006), 1338–1351.

**Architecture**

1、方差损失

global error
$$
E = \sum_{i=1}^{m}E_i
$$
m multi-label instances .

Q lables
$$
E_i = \sum_{j=1}^{Q}(c_j^i - d_j^i)
$$
$c_j^i = c_j(X_i)$ is the actual output of the network on xi on the j-th class.

$d^i_j$ is the desired output of $X_i$ on the j-th class. 取值为1，-1



=》 只关注单个标签的识别，没有考虑类别之间的相关性，真的标签要大于假的标签的值。



本文通过重写全局错误函数，适当地解决了多标记学习的这些特点：

2、
$$
E = \sum_{i=1}^{m}E_i =  \sum_{i=1}^{m} \frac{1}{|Y_i||\overline{Y_i}|} 
\sum_{(k,l) \in Y_i \times \overline{Y_i}} \exp(-(c_k^i - c_l^i))
$$
在第i个误差项中的求和考虑了任意一对标签的输出与另一对不属于xi的标签的输出之间的累积差，然后根据可能的对的总数进行归一化，





















