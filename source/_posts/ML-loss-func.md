---
title: Summary of loss function in Machine Learning
date: 2019-10-12 11:17:17
tags:
- loss function
- ML
---

## ML损失函数

#### 0-1损失函数

$$
L(Y, f(X)) = \begin{cases}
1, Y \neq f(X) \\ 
0, Y = f(X)
\end{cases}
$$


$$
L(Y, f(X)) = \begin{cases}
1 , |Y - f(X)| \geq T \\
0 , |Y = f(X)| < T
\end{cases}
$$


#### 绝对值损失函数

$$
L(Y, f(X)) = |Y - f(X)|
$$



#### 平方损失函数

实际结果和观测结果之间差距的平方和，一般用在线性回归中，(与最小二乘法应用场景类似)
$$
L(Y, f(X)) = \sum_{i=1}^{N} (y_i-f(x_i))^2
$$




#### 对数损失函数

	主要在逻辑回归中使用，样本预测值和实际值的误差符合高斯分布，使用极大似然估计的方法，取对数得到损失函数：
$$
L(Y, P(Y|X)) = -logP(Y|X)
$$
对数损失函数包括entropy和softmax，一般在做分类问题的时候使用（而回归时多用绝对值损失（拉普拉斯分布时，μ值为中位数）和平方损失（高斯分布时，μ值为均值））



#### 指数损失函数

$$
L(Y|f(X))  = \exp(-yf(x))
$$



#### 铰链损失函数

	主要用在SVM中，Hinge Loss的标准形式为：
$$
L(y) = max(0, 1-ty)
$$



## Keras / TensorFlow 中常用 Cost Function 总结

- mean_squared_error或mse
- mean_absolute_error或mae
- mean_absolute_percentage_error或mape
- mean_squared_logarithmic_error或msle
- squared_hinge
- hinge
- categorical_hinge
- binary_crossentropy（亦称作对数损失，logloss）
- sigmoid_cross_entropy
- softmax_cross_entropy
- sparse_softmax_cross_entropy
- logcosh
- categorical_crossentropy：亦称作多类的对数损失，注意使用该目标函数时，需要将标签转化为形如`(nb_samples, nb_classes)`的二值序列
- sparse_categorical_crossentrop：如上，但接受稀疏标签。注意，使用该函数时仍然需要你的标签与输出值的维度相同，你可能需要在标签数据上增加一个维度：`np.expand_dims(y,-1)`
- kullback_leibler_divergence:从预测值概率分布Q到真值概率分布P的信息增益,用以度量两个分布的差异.
- poisson：即`(predictions - targets * log(predictions))`的均值
- cosine_proximity：即预测值与真实标签的余弦距离平均值的相反数