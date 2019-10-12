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

### **Architecture**

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

2、论文自定义指数损失函数
$$
E = \sum_{i=1}^{m}E_i =  \sum_{i=1}^{m} \frac{1}{|Y_i||\overline{Y_i}|} 
\sum_{(k,l) \in Y_i \times \overline{Y_i}} \exp(-(c_k^i - c_l^i))
$$
在第i个误差项中的求和考虑了任意一对标签的输出与另一对不属于xi的标签的输出之间的累积差，然后根据可能的对的总数进行归一化，



3、思考我的对数损失函数。

​	采用对数损失函数sigmoid的交叉熵的形式。

softmax_cross_entropy_with_logits： labels 中每一维只能包含一个 **1**

sigmoid_cross_entropy_with_logits： labels 中每一维只能可以含多个 **1**



-- 所以此论文采用指数损失函数的原因是：

1、out值取值范围【-1，1】 tanh

2、对数损失函数不成熟当时？作者不熟

3、其它



### 评估

#### hamming loss 

错误标签的比例 [hamming_loss]，属于某个样本的标签没有被预测，不属于
该样本的标签被预测属于该样本。 
$$
\text{hloss}_S(h) = \frac{1}{p}\sum_{i=1}^{p}\frac{1}{Q} |h(X_i\Delta Y_i|
$$


#### one-errors

预测概率值最大的标签不在真实标签集中的次数。  [zero_one_loss]
$$
\text{one-errors}_S(f) = \frac{1}{p} \sum_{i=1}^{p} [[argmax_{y \in \mathcal{Y}} f(X_i, y)] \notin Y_i]
$$




#### coverage 覆盖误差

[coverage_error] 表示所有样本中排序最靠后的真实标签的排序均值。 
$$
\text{coverage}_S(f) = \frac{1}{p} \sum_{i=1}^{p} \text{max}_{y\in Y_i}
rank_f(X_i, y) - 1
$$


#### rank loss 排名损失

[label_ranking_loss] 表示相关标签的预测概率值比不相关标签预测概率值小的次数。 
$$
\text{rloss}_S(f) = \frac{1}{p} \sum_{i=1}^{p} \frac{|D_i|}{|Y_i||\overline{Y_i}|}
\\
D_i = \{(y1, y2)| f(x_i,y1) \leq f(x_i, y2), (y1, y2) \in Y_i \overline{Y_i} \}
$$


#### average precision 平均精度损失

[average_precision_score] 表示标签高于某个特定标签 y∈ Y 的统计概率。 
$$
avgprec_S(f)= \frac{1}{p} \sum^{p}_{i=1} 
\frac{1}{\overline{Y_i}} \sum_{y \in Y_i} 
\frac{ L_i}{ rank_f(x_i, y) }
$$














