# 从零开始的机器学习

## 数据预处理
### 特征缩放
为了使范围不同的数据都能相对平均的影响到拟合结果，而不会因其本身的数值大小而被过度关注或忽视。我们引入特征缩放，可以将数据范围重新放缩到[0, 1]或[-1, 1]。

### 缺失数据处理
我们对于缺失数据，如果将其空置或给予一个随意的值都有可能较大的影响拟合结果。因而，我们往往会使用平均数或中位数来填充它。

## 可视化
对于数据本身，可视化是可以让我们直观的看到低维数据的方法。

而如果把坐标轴换成某给变量（如loss）和迭代次数，那么我们绘制的图像就能显示该模型下的训练过程。也能指示我们的模型是高方差、高偏差，还是合适的。

## 数据集处理
### k折交叉验证
若样本量相对较小，则可以使用k折交叉验证来训练模型，这样能够更充分的利用数据、使模型更稳健。
方法是将数据集分成k个包，取其中的k-1个作为训练集，剩下的一个作为测试集。


## 线性回归
### 代价函数
无论使用哪种方式去寻找我们需要的线性方程，总要有一个评价指标。我们称之为代价函数，


### 如何寻找拟合的系数
#### 不好的寻找方式：枚举
假设我们使用一个2层、每层10个节点的全连接网络，除bias外的参数就有100个，我们自然也可以找到一个「恰当」的参数，但其复杂度非常高。

但我们还是尝试去实现它，但是作为一个反面教材出现。

#### 数学方法：最小二乘法
使用数学方法来拟合数据，结果可能不是最优，但其复杂度极低。

#### 梯度下降
我们可以不断计算残差，然后找到更新每个参数，使其向最优方向再进化一点。反复迭代，即接近最优接解（正常拟合时）。

而对于批量梯度下降，我们是使用所有的样本来计算残差，因而在大样本的时候会出现运行缓慢的问题。因而改进算法，先随机打乱数据，然后从中选取任一来优化参数。这样虽然可能在局部出现非最优优化、甚至是劣化的现象，但整体上是向最优方向靠近的。

为了使进化方向更稳定，再优化算法如下。在打乱后，随机抽取指定数量的样本，以避免单个样本可能出现的离群值问题。


## 逻辑回归
在二分类逻辑回归中，我们通常使用Sigmoid函数来将预测结果缩小到(0,1)的范围内。

## 距离函数
### 欧几里德距离（欧氏距离）
与初中学的距离概念相同，即直接使用两点间线段的长度代表距离。

### 曼哈顿距离
使用坐标系下，与坐标轴平行的线段长度之和表示距离。所有线段首尾相连，从一点出发，在平行与坐标轴的前提下取得最短距离，连接到另一点。

### 切比雪夫距离
为其各座标数值差的最大值，字面意思。

### 闵可夫斯基距离（闵氏距离）
可以看作以上几种距离的集合，其维度等于1时，为曼哈顿距离；为2时，为欧几里德距离；大于2时，为切比雪夫距离。


# Machine Learning From Scratch
## Data Preprocessing
### Feature Scaling
There is an aim that we want the result could show the difference evenly and will not be significant affected by the data in larger range. Therefore, we use scaling to force the different dimension of data have a similar range, for instance, [0, 1] or [-1, 1].

### Process of Missing Data
As for the missing data, it will product a wrong result if it be replaced into NAN or random value. Thus, we usually fill it with the average or median.

## Visualization
Visualization is the considerable method for us to catch the sight of low-dimension data.

//For instance, if we set axis as a variable
