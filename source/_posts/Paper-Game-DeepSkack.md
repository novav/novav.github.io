---
title: DeepSkack to Texas Hold'em 
date: 2019-10-24 17:32:31
tags:
- Game
- CFR
- DeepSkack
---



## Deep Counterfactual Value Networks

### Architecture & Train:

两个NN；

Flop Network: 1 million randomly generated flop games. 

Turn Network: 10 million randomly generated poker turn games. 

一个辅助网络；在处理任何公共卡之前，使用一个辅助的值网络来加速早期动作的重新求解

![1571911214323](Paper-Game-DeepSkack/1571911214323.png)



## Evaluating DeepStack 





















