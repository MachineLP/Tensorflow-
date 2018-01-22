# coding=utf-8
from numpy import *  
import operator
import os


# 原始形式感知机算法
# 下面是一个简单的例子， 帮助理解感知机
def createDataSet():  
    # 创建三组数据，及其对应的标签 
    group = array([[3,3], [4,3], [1,1]])  
    labels = [1, 1, -1] # 共分为两类
    return group, labels

# 下面是感知机参数的更新过程， 其实就是按照上一篇的原理实现的：
def perceptronClassify(trainGroup,trainLabels):
    global w, b
    # 用来终止程序 和 保留最好的w, b;
    isFind = False
    # 计算样本数量， 用于参数训练更新；
    numSamples = trainGroup.shape[0]
    # 计算特征的维度， 用于初始化权重;
    mLenth = trainGroup.shape[1]
    # 开始将权重都初始化为0;
    w = [0]*mLenth
    b = 0
    # 用来标识是否样本都分类成正确，用停止 和 保存最好的权重;
    while(not isFind):
        # 去每一个样本， 进行判断， 预测模型时候是否将其错误分类;
        for i in range(numSamples):
            # 按照上一节，误分类点相乘 < 0
            if cal(trainGroup[i],trainLabels[i]) <= 0:
                print (w, b)
                # 更新参数， 这里要注意一下，每次有误分类点， 更新完以后，此for训练结束；
                # 重新进入while循环，表明一点发现误分类点后，更新模型，然后再从第一个点开始;
                # 这里就好比我们再学习数据结构算法时，各种时间和空间的优化，像动态规划自顶向下 和 被备忘的自底向上;
                # 还好比数组中连续累加和最大的序列，可以从O(n^2)优化到O(n);
                # 所以个人感觉无论你学什么，只要变成请好好学习数据结构和算法。
                update(trainGroup[i],trainLabels[i])
                break    #end for loop
            # 这地方更巧妙，判断最后一个点， 由上面的逻辑，最后一个点对了，前面的点肯定分对了;
            elif i == numSamples-1:
                print (w, b)
                isFind = True   #end while loop

# 用来判断是不是模型的误分类点的;
def cal(row,trainLabel):
    # 设置为全局变量， 这里也可以都写到一类中;
    global w, b
    res = 0
    #  样本每个属性乘以权重 + 偏置， 计算预测值; 
    # 这里可以理解用样本矩阵的每一行， 也就是样本中一个样本；
    for i in range(len(row)):
        res += row[i] * w[i]
    res += b
    # 预测值和真实值相乘， 看是大于0; 还是小于0, 用来判断是不是误分类点;
    res *= trainLabel
    return res
# 如果是误分类点， 就更新权重;
def update(row,trainLabel):
    global w, b
    # w[i] = w[i] + 一个样本中的每一个值 * 标签 * 学习率; 这地方学习率可以默认为1了
    for i in range(len(row)):
        w[i] += trainLabel * row[i]
    # 下面是偏置的更新
    b += trainLabel

#import perceptron
g,l = createDataSet()
perceptronClassify(g,l)
print("w, b", w, b)

# 对上面的模型训练结果进行显示;
import matplotlib.pyplot as plt
from pylab import *
import numpy as np

group = array([[3,3], [4,3], [1,1]])

plt.figure(1)
plt.subplot(111)
x1 = [3, 4]
y1 = [3, 3]
x2 = [1]
y2 = [1]
plt.plot(x1, y1, 'bs')
plt.plot(x2, y2, 'g^')
x = np.arange(-5.0, 5.0, 0.02)
y = - w[0]/w[1] * x - (b)
plt.plot(x, y)
plt.grid(True)
plt.show()

####################################################################################################                                 

from numpy import *  
import operator
import os

# 对偶形式的感知机算法

def createDataSet():  
    
    group = array([[3,3], [4,3], [1,1]])  
    labels = [1, 1, -1] 
    return group, labels

#classify using DualPerception
def dualPerceptionClassify(trainGroup,trainLabels):
    global a,b
    isFind = False
    numSamples = trainGroup.shape[0]
    #mLenth = trainGroup.shape[1]
    a = [0]*numSamples
    b = 0
    gMatrix = cal_gram(trainGroup)
    while(not isFind):
        for i in range(numSamples):
            if cal(gMatrix,trainLabels,i) <= 0:
                cal_wb(trainGroup,trainLabels)
                update(i,trainLabels[i])
                break
            elif i == numSamples-1:
                cal_wb(trainGroup,trainLabels)
                isFind = True
    

# caculate the Gram matrix
def cal_gram(group):
    # 样本数量
    mLenth = group.shape[0]
    gMatrix = zeros((mLenth,mLenth))
    for i in range(mLenth):
        for j in range(mLenth):
            gMatrix[i][j] =  dot(group[i],group[j])
    return gMatrix

def update(i,trainLabel):
    global a,b
    a[i] += 1
    b += trainLabel

def cal(gMatrix,trainLabels,key):
    global a,b
    res = 0
    for i in range(len(trainLabels)):
        res += a[i]*trainLabels[i]*gMatrix[key][i]
    res = (res + b)*trainLabels[key]
    return res

# 下面代码通过a计算出 权重(w)和偏置(b)
def cal_wb(group,labels):
    global a,b
    w=[0]*(group.shape[1])
    h = 0
    for i in range(len(labels)):
        h +=a[i]*labels[i]
        w +=a[i]*labels[i]*group[i]
    print (w,h)

# import DualPerception
g,l = createDataSet()
dualPerceptionClassify(g,l)

