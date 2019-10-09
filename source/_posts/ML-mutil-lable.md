---
title: ML_mutil-lable
date: 2019-10-09 16:41:00
tags: 
- Multilabel learning
- ranking
- classification
- machine learning
- data mining
---

《A Tutorial on Multilabel Learning 》 [download pdf](http://dx.doi.org/10.1145/2716262 )

EVA GIBAJA， SEBASTIAN VENTURA 

西班牙 科多巴大学

this article presents an up-to-date tutorial about multilabel learning that introduces the paradigm and describes the main contributions developed. evaluation measures, fields of application, trending topics, and resources are also presented.

本文介绍了一个关于多标签学习的最新教程，介绍了该范例，并描述了已开发的主要贡献。还介绍了评估措施、应用领域、趋势主题和资源。

categories and subject descriptors: h.2.8 [database management]: database applications—data mining; h.3.3 [information storage and retrieval]: information search and retrieval—retrieval models; i.2.6 [artificial intelligence]: learning—concept learning, connectionism and neural nets, induction; i.7.5 [document and text processing]: document capture—document analysis; i.4.8 [image processing and computer vision]: scene analysis—object recognition; i.5.2 [pattern recognition]: design methodology—classifier design and evaluation; i.5.4 [pattern recognition]: applications—computer vision, text processing

类别和主题描述符：

```
H.2.8[数据库管理]：数据库应用-数据挖掘；
H.3.3[信息存储和检索]：信息搜索和检索-检索模型；
i.2.6[人工智能]：学习-概念学习、连接论和神经网络，归纳；
I.7.5[文件和文本处理]：文件捕获-文档分析；
I.4.8[图像处理和计算机视觉]：场景分析-目标识别；
I.5.2[模式识别]：设计方法-分类器设计和评价；
I.5.4[模式识别]：应用-计算机视觉、文本处理
```

## 1 introduction

分类是数据挖掘的主要任务之一。

一组特征，一个相关类的识别。

如今，人们正在考虑越来越多的分类问题，例如文本和声音分类、语义场景分类、医学诊断或基因和蛋白质功能分类，其中一个模式可以同时具有多个标签。

single-label (classical supervised learning （经典监督学习）)

multilable ( Multilabel Learning (MML))

除其他问题外，本文的目的是包括问题的正式定义、适用MLL的领域、最近几年提出的主要建议的最新摘要、评估措施和资源。



outline

Section 2, the MLL problem is formally defined. 

Section 3, some aspects related to the development and **evaluation** of MLL models are described. 

Section 4, The main approaches developed in the literature are presented

Section 5 describes findings on empirical comparisons between MLL algorithms. 

Section 6 describes the maindomains where MLL has been applied, 

finally, new trends in MLL (Section 7) and a set of conclusions are presented. 

The article also includes an Appendix with resources
(software, datasets, etc.) for MLL. 

## 2.MML

MLL includes two main tasks: 

​	Multilabel Classification (MLC) 

​	Label Ranking (LR). 

​	Multilabel Ranking  

### 3. EVALUATION OF MULTILABEL MODELS 

3.1 evaluation metrics

​	3.1.1 metrics to evalute bipartitions

​		two approches:

​			lebel-based (macro, micro)

​			example-based (*0/1 subset accuracy*=*classifification accuracy* or *exact match ratio* )
$$
B_{macro}=1/q \sum_{i=1}^{q}B(tp_i, fp_i, tn_i, fn_i)
$$

$$
B_{micro} = B(\sum_{i=1}^{q}tp_i, \sum_{i=1}^{q}fp_i, \sum_{i=1}^{q}tn_i, \sum_{i=1}^{q}fn_i)
$$

$$
0/1 -subset- accuracy = 1/t \sum_{i=1}^{t}[Z_i= Y_i]
$$

$$
Hamming loss = 1/t \sum^{t}_{i=1} 1/q|Z_i /\Y_i|.
$$



​	3.1.2 metrics to evalute rankings

​		One-error

how to prepare the dataset,

 statistical tests, 

complexity 