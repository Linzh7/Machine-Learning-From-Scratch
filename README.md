# Machine-Learning-From-Scratch

## 前言

个人的身体状况也不知道能不能支持写完这么多东西，但总会尽力去写。

因为是非科班出身，因而确实踩了很多坑，我尽力在完善这个项目的过程中将之一一说明（如果还记得的话）

## 线性回归

我使用了随机生成的数据，大概是 y=3x+3
如果不想用默认的这个，可以传参进去，使用方法：

DataReader.PrebuiltData.MyData(rangeStart, rangeEnd, rangeStep, k, b)



### 代价函数

无论使用哪种方式去寻找我们需要的线性方程，总要有一个评价指标。我们称之为代价函数，

### 不好的寻找方式：枚举

假设我们使用一个2层、每层10个节点的全连接网络，除bias外的参数就有100个，我们自然也可以找到一个「恰当」的参数，但其精细度与复杂度非常高。

但我们还是尝试去实现它，但是作为一个反面教材出现。
