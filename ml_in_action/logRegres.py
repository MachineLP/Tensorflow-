'''
Created on Oct 27, 2010
Logistic Regression Working Module
@author: Peter
'''
from numpy import *

# 加载样本数据和标签
def loadDataSet():
    # 定义存放样本和标签的列表;
    dataMat = []; labelMat = []
    fr = open('testSet.txt')
    # 读取文件中的每一行，样本和标签
    for line in fr.readlines():
        # 分隔每行的数据
        lineArr = line.strip().split()
        # 前两个为样本数据，又添加了一维， 目的是将w和b融合一起，计算更加方便;
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        # 样本的标签;
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

# 这个应该很熟悉了吧，sigmoid函数;
def sigmoid(inX):
    return 1.0/(1+exp(-inX))

# 梯度上升算法用来求函数（这里的函数就是损失函数）的最大值; 梯度下算法用来求函数（损失函数）的最小值;
# 其实这两个损失函数就一个负号之差;
def gradAscent(dataMatIn, classLabels):
    # 将样本定义为矩阵，方便参数更新时候的矩阵运算;
    dataMatrix = mat(dataMatIn)             #convert to NumPy matriax
    # 将标签也转为矩阵的形式，方便后边的 预测值和真实标检做差值;
    labelMat = mat(classLabels).transpose() #convert to NumPy matrix
    # 得到矩阵的行数和列数， 也就是样本的数量和维度，维度用来定义权重参数 w 和 b;
    m,n = shape(dataMatrix)
    # 权重更新时候的学习率
    alpha = 0.001
    # 选择更新多少次权重;
    maxCycles = 500
    # 初始化权重都为1;
    weights = ones((n,1)
    # 循环更新权重值
    for k in range(maxCycles):              #heavy on matrix operations
        # 根据更新后的权重预测样本的类别值
        h = sigmoid(dataMatrix*weights)     #matrix mult
        # 真实标签和样本值做差值
        error = (labelMat - h)              #vector subtraction
        # 下面就是梯度上升的迭代公式;
        weights = weights + alpha * dataMatrix.transpose()* error #matrix mult
    return weights

# 下面就是画不同类别的样本点 和 分类线;
def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat,labelMat=loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0] 
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i])== 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2');
    plt.show()

# 梯度上升每次更新权重都要遍历整个数据集，该方法在少量数据集上尚可，但是数据集大了，计算复杂度就太高了；
# 下面是改进的方法一次用一个样本点更新权重；
def stocGradAscent0(dataMatrix, classLabels):
    # 获取样本的行数和列数， 标识样本数量和维度：
    m,n = shape(dataMatrix)
    # 定义随机梯度上升的学习率
    alpha = 0.01
    # 初始化权重
    weights = ones(n)   #initialize to all ones
    # 遍历所有的样本， 用每个样本来更新权重
    for i in range(m):
        # 预测一个样本的类别
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = classLabels[i] - h
        # 权重更新迭代式
        weights = weights + alpha * error * dataMatrix[i]
    return weights

# 由于上面的随机梯度下降的学习率是肯定的， 并且样本的顺序也是固定的
# 下面对其进行了优化， 学习率随着迭代次数逐渐减少，并且将样本引入了随机性，即随机选取样本进行权重更新
def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m,n = shape(dataMatrix)
    weights = ones(n)   #initialize to all ones
    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            # 学习率不再是唯一的
            alpha = 4/(1.0+j+i)+0.0001    #apha decreases with iteration, does not 
            # 随机选取样本；这种方法可以减少周期性的波动; 
            randIndex = int(random.uniform(0,len(dataIndex)))#go to 0 because of the constant
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            # 记得将随机选取的值在数组中删除掉！
            del(dataIndex[randIndex])
    return weights

# 下面就是分出类别，根据sigmoid的值，>0判断为1，小于0判断为0；
def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5: return 1.0
    else: return 0.0

# 下面进行训练和测试阶段
def colicTest():
    # 分别打开训练机和测试集文件
    frTrain = open('horseColicTraining.txt'); frTest = open('horseColicTest.txt')
    # 定义保存训练样本和训练标签的列表
    trainingSet = []; trainingLabels = []
    # 读取每一行， 也就是一个样本和标签；
    for line in frTrain.readlines():
        # 用以将样本和标签分割开
        currLine = line.strip().split('\t')
        # 用与保存每一个样本中的所有值
        lineArr =[]
        # 取每行前21个为样本，剩下的一个为标签值；
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    # 用随机上升迭代1000次来求权重值；
    trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 1000)
    # 根据上面求得的权重值，测试 在测试结果上的错误率；
    errorCount = 0; numTestVec = 0.0
    # 读取测试文件中的每一行测试
    for line in frTest.readlines():
        # 用来统计测试样本的数量
        numTestVec += 1.0
        # 和上边一样， 用于分割样本数据和标签数据
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        # 下面来计算预测值和真实值是否相等， 如果不相等加一， 统计错误样本的数量
        if int(classifyVector(array(lineArr), trainWeights))!= int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print "the error rate of this test is: %f" % errorRate
    return errorRate

# 相当于计算十次的权重，得平均值；
def multiTest():
    numTests = 10; errorSum=0.0
    for k in range(numTests):
        errorSum += colicTest()
    print "after %d iterations the average error rate is: %f" % (numTests, errorSum/float(numTests))
        