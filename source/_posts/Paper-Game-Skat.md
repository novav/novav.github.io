---
title: Game_Skat
date: 2019-09-19 11:39:58
tags:
---



## Skat:

《2011 [Skat] Policy Based Inference in Trick-Taking Card Games 》 【博士论文】Jeffrey Richard Long 
```
三个贡献

【4】专家级计算机SKAT-AI (组合游戏树搜索、状态评估和隐藏信息推理三个核心方面的组合来实现这一性能)
《M. Buro, J. Long, T. Furtak, and N. R. Sturtevant. Improving state evaluation, inference, and
search in trick-based card games. In Proceedings of the 21st International Joint Conference on Artificial Intelligence (IJCAI2009), 2009. 》

【26】次优解决方案方法的框架
《 J. Long, N. R. Sturtevant, M. Buro, and T. Furtak. Understanding the success of perfect information monte carlo sampling in game tree search. In Proceedings of the 24th AAAI Conference on Artificial Intelligence (AAAI2010), 2010.》

【27】一种简单的自适应对手实时建模技术。
《 J. R. Long and M. Buro. Real-time opponent modelling in trick-taking card games. In Proceedings of the 22nd International Joint Conference on Artificial Intelligence (IJCAI2011), page To appear, 2011. 》
```


- 介绍Best Defense Model的构建（三个假设）【Contract Bridge】


  - 对手(Miner)是完全信息的，我们(Maxer)是非完全
  - 对手在Maxer选择之后play
  - Maxer采用纯策略，不适用融合策略

  this results in an algorithm for solving the best defense form of a game which frank and basin term exhaustive strategy minimisation（穷举策略最小化）.

DDS(Double-Dummy Solver ):	PIMC + ab

- (对alpha-beta搜索组件的两个主要增强):
  - 准对称缩小(quasi-symmetry reduction )，是Ginsberg的分区搜索的一种改编，它将搜索状态组合在一起，这些状态“几乎”等价
  - 启发式对抗（adversarial heuristics ）




- 最强AI-Skat ‘Kermit ‘ 设计：
- Perfect Information Monte Carlo Search for Cardplay 
  
- State Evaluation for Bidding 
  
- Inference 
  
- Kermit Versus the World: Experimental Results 
  



引用

- 【9】对不完美信息博弈的完美信息蒙特卡罗搜索(或称为重复最小化)方法的广泛批评。
- 【29】描述使用神经网络预测完美信息(或double-dummy )桥接问题的结果，并报告与某些类别交易的人类专家相当的准确性。
- 【35】最强桥牌AI-Ginsberg’s GIB  {加权MC样本weighted Monte Carlo samples }

---

《2009 Improving State Evaluation, Inference, and Search in Trick-Based Card Games》

https://www.cs.du.edu/~sturtevant/papers/skat.pdf

- 贝叶斯公式

- KI Kermit infrence



《AAAI2010 Understanding the success of perfect information monte carlo sampling in game tree search》



《Real-time opponent modelling in trick-taking card games (IJCAI2011) 》



---



《2019 Policy Based Inference in Trick-Taking Card Games》

http://ieee-cog.org/papers/paper_123.pdf

- 真实状态采样比率
- 对手模型
- 信息集状态分布评估



《2019 Improving Search with Supervised Learning in Trick-Based Card Games》

https://arxiv.org/pdf/1903.09604.pdf

- 



《2019 Learning policies form human data for skat》

https://arxiv.org/pdf/1905.10907.pdf

- 预测隐藏牌

- NN 训练pre-cardplay 和cardplay

  



### Scope

四人游戏（意大利游戏）

《1807.06813 [Scopone] Traditional Wisdom and Monte Carlo Tree Search Face-to-Face in the Card Game Scopone.pdf》

