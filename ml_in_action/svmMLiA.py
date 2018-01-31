'''
Created on Nov 4, 2010
Chapter 5 source file for Machine Learing in Action
@author: Peter
'''
from numpy import *
from time import sleep

# 依旧是数据的准备
def loadDataSet(fileName):
    # 定义保存样本和标签的列表;
    dataMat = []; labelMat = [a]
    # 打开数据文件
    fr = open(fileName)
    # 读取文件中的每一行;
    for line in fr.readlines():
        # 将每行的数据通过制表符分开;
        lineArr = line.strip().split('\t')
        # 前两个为样本数据数据，第二个为标签数据;
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    # 返回样本和标签，用于SVM的训练;
    return dataMat,labelMat

# 用于产生一个随机数，用于下面随机获取一个样本;
def selectJrand(i,m):
    j=i #we want to select any J not equal to i
    while (j==i):
        j = int(random.uniform(0,m))
    return j

# 相当于给定alpha一个范围，大于最大值的话，赋值为最大值，小于最小值的话，就赋值最小值; 
def clipAlpha(aj,H,L):
    if aj > H: 
        aj = H
    if L > aj:
        aj = L
    return aj

# 简化版的smo算法求alpha和b;
# 下面是smo算法的流程;
# 此简化版的smo是严格按照上一节MachineLN之SVM（2）的手撕smo来的；
def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    # 将样本集转化为矩阵格式， 将样本的标签也转化为矩阵格式和，用于后面的矩阵运算， 主要两个地方用到：预测和eta。 
    dataMatrix = mat(dataMatIn); labelMat = mat(classLabels).transpose()
    # 初始化偏置，然后获取矩阵的行数和列数， 行数用来输出初始化下面的alpha. 
    b = 0; m,n = shape(dataMatrix)
    # 初始化alphas为为m行1列；
    alphas = mat(zeros((m,1)))
    iter = 0
    # 定义迭代次数
    while (iter < maxIter):
        alphaPairsChanged = 0
        # 遍历每个样本;
        for i in range(m):
            # 计算第i个样本的预测标签; 用于计算差值； 
            fXi = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[i,:].T)) + b
            # 计算两个的差值用于 KKT 条件的判断
            Ei = fXi - float(labelMat[i])#if checks if an example violates KKT conditions
            # 正间隔 和 负间隔 都会被测试; 并且还要保证 alpha的值在 [0, C]之间
            if ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):
                # 从 i到m中随机选择一个样本
                j = selectJrand(i,m)
                # 计算此样本的预测值
                fXj = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[j,:].T)) + b
                # 预测值和真实值的差值： 用于后面计算alpha. 
                Ej = fXj - float(labelMat[j])
                # 用于保存未更新的alpha，方便b的计算;
                alphaIold = alphas[i].copy(); alphaJold = alphas[j].copy();
                # 计算不同的情况下 aphpa 的最小值和最大值， 这里可以参考手撕smo;
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L==H: print "L==H"; continue
                # 下面就是计算alpha2 和 进行剪枝后，求alpha1;
                eta = 2.0 * dataMatrix[i,:]*dataMatrix[j,:].T - dataMatrix[i,:]*dataMatrix[i,:].T - dataMatrix[j,:]*dataMatrix[j,:].T
                if eta >= 0: print "eta>=0"; continue
                alphas[j] -= labelMat[j]*(Ei - Ej)/eta
                alphas[j] = clipAlpha(alphas[j],H,L)
                if (abs(alphas[j] - alphaJold) < 0.00001): print "j not moving enough"; continue
                alphas[i] += labelMat[j]*labelMat[i]*(alphaJold - alphas[j])#update i by the same amount as j
                # 更新参数b1， b2， 和手撕smo算法流程一样;
                b1 = b - Ei- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
                b2 = b - Ej- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T
                # 根据参数b1， b2得到b;
                if (0 < alphas[i]) and (C > alphas[i]): b = b1
                elif (0 < alphas[j]) and (C > alphas[j]): b = b2
                else: b = (b1 + b2)/2.0
                alphaPairsChanged += 1
                print "iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged)
        if (alphaPairsChanged == 0): iter += 1
        else: iter = 0
        print "iteration number: %d" % iter
    return b,alphas

# 下面就是核函数：线性核函数 和 rbf核函数
def kernelTrans(X, A, kTup): #calc the kernel or transform data to a higher dimensional space
    m,n = shape(X)
    K = mat(zeros((m,1)))
    if kTup[0]=='lin': K = X * A.T   # 线性核
    elif kTup[0]=='rbf':
        for j in range(m):
            deltaRow = X[j,:] - A
            K[j] = deltaRow*deltaRow.T
        K = exp(K/(-1*kTup[1]**2))  # rbf核
    else: raise NameError('Houston We Have a Problem -- \
    That Kernel is not recognized')
    return K

# 利用完整的 Platt SMO算法加速运算;
# 与简化版相比：实现alpha的更改和代数运算的优化环节一摸一样，在优化过程中唯一不同的就是选择alpha的方式。
# 用于设置模型中的数据和参数： 训练样本、标签、学习率、KKT条件的参数设置值、alpha、b、核函数的参数；
class optStruct:
    def __init__(self,dataMatIn, classLabels, C, toler, kTup):  # Initialize the structure with the parameters 
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m,1)))
        self.b = 0
        self.eCache = mat(zeros((self.m,2)))  # 用于误差缓存
        self.K = mat(zeros((self.m,self.m)))
        for i in range(self.m):
            self.K[:,i] = kernelTrans(self.X, self.X[i,:], kTup)

# 计算预测值和真实值的标签的差值;        
def calcEk(oS, k):
    fXk = float(multiply(oS.alphas,oS.labelMat).T*oS.K[:,k] + oS.b)
    Ek = fXk - float(oS.labelMat[k])
    return Ek

# 在选择第一个alpha值后，算法会通过内循环来选择第二个alpha值，在优化过程中，会通过最大步长的方式来获得第二个alpha值
# 选择合适的第二个样本; 计算Ej 
def selectJ(i, oS, Ei):         
    maxK = -1; maxDeltaE = 0; Ej = 0
    # 将其放在Ei缓存区。
    oS.eCache[i] = [1,Ei]  #set valid #choose the alpha that gives the maximum delta E
    # 返回的是非零E值所对应的alpha值，而不是E本身，程序会在所有的值上进行循环并选择其中使得改变最大的那个值；
    # else中， 在第一次的循环的话， 那么就随机选择一个alpha值。
    validEcacheList = nonzero(oS.eCache[:,0].A)[0]
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:   
            if k == i: continue 
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):
                maxK = k; maxDeltaE = deltaE; Ej = Ek
        return maxK, Ej
    else:   
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej

# 更新选取新样本后的E值
def updateEk(oS, k):#after any alpha has changed update the new value in the cache
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1,Ek]
        
# 下面的算法流程和简化版的smo流程差不多
def innerL(i, oS):
    Ei = calcEk(oS, i)
    if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)):
        j,Ej = selectJ(i, oS, Ei) #this has been changed from selectJrand
        alphaIold = oS.alphas[i].copy(); alphaJold = oS.alphas[j].copy();
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L==H: print "L==H"; return 0
        eta = 2.0 * oS.K[i,j] - oS.K[i,i] - oS.K[j,j] #changed for kernel
        if eta >= 0: print "eta>=0"; return 0
        oS.alphas[j] -= oS.labelMat[j]*(Ei - Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)
        updateEk(oS, j) #added this for the Ecache
        if (abs(oS.alphas[j] - alphaJold) < 0.00001): print "j not moving enough"; return 0
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])#update i by the same amount as j
        updateEk(oS, i) #added this for the Ecache                    #the update is in the oppostie direction
        b1 = oS.b - Ei- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,i] - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[i,j]
        b2 = oS.b - Ej- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,j]- oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[j,j]
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]): oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]): oS.b = b2
        else: oS.b = (b1 + b2)/2.0
        return 1
    else: return 0

# Platt AMO算法
def smoP(dataMatIn, classLabels, C, toler, maxIter,kTup=('lin', 0)):    #full Platt SMO
    oS = optStruct(mat(dataMatIn),mat(classLabels).transpose(),C,toler, kTup)
    iter = 0
    entireSet = True; alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:   #go over all
            for i in range(oS.m):        
                alphaPairsChanged += innerL(i,oS)
                print "fullSet, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged)
            iter += 1
        else:#go over non-bound (railed) alphas
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i,oS)
                print "non-bound, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged)
            iter += 1
        if entireSet: entireSet = False #toggle entire set loop
        elif (alphaPairsChanged == 0): entireSet = True  
        print "iteration number: %d" % iter
    return oS.b,oS.alphas

# 通过计算的alpha计算权重w值
def calcWs(alphas,dataArr,classLabels):
    X = mat(dataArr); labelMat = mat(classLabels).transpose()
    m,n = shape(X)
    w = zeros((n,1))
    # 计算 w
    for i in range(m):
        w += multiply(alphas[i]*labelMat[i],X[i,:].T)
    return w

# 使用rbf核的svm，进行测试
def testRbf(k1=1.3):
    dataArr,labelArr = loadDataSet('testSetRBF.txt')
    # 通过Platt AMO算法计算alpha和b的值
    b,alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, ('rbf', k1)) #C=200 important
    # 
    datMat=mat(dataArr); labelMat = mat(labelArr).transpose()
    # 构建支持向量矩阵， 选择 0<alpha<C
    svInd=nonzero(alphas.A>0)[0]
    # 仅选支持向量用于kernel相乘
    sVs=datMat[svInd] #get matrix of only support vectors
    labelSV = labelMat[svInd];
    print "there are %d Support Vectors" % shape(sVs)[0]
    m,n = shape(datMat)
    errorCount = 0
    # 下面是计算在训练集的错误率
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],('rbf', k1))
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!=sign(labelArr[i]): errorCount += 1
    print "the training error rate is: %f" % (float(errorCount)/m)
    # 下面是求在测试集的错误率
    dataArr,labelArr = loadDataSet('testSetRBF2.txt')
    errorCount = 0
    datMat=mat(dataArr); labelMat = mat(labelArr).transpose()
    m,n = shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],('rbf', k1))
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!=sign(labelArr[i]): errorCount += 1    
    print "the test error rate is: %f" % (float(errorCount)/m)    

# 下面是手写体识别的svm测试
def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def loadImages(dirName):
    from os import listdir
    hwLabels = []
    trainingFileList = listdir(dirName)           #load the training set
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        if classNumStr == 9: hwLabels.append(-1)
        else: hwLabels.append(1)
        trainingMat[i,:] = img2vector('%s/%s' % (dirName, fileNameStr))
    return trainingMat, hwLabels    

def testDigits(kTup=('rbf', 10)):
    dataArr,labelArr = loadImages('trainingDigits')
    b,alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, kTup)
    datMat=mat(dataArr); labelMat = mat(labelArr).transpose()
    svInd=nonzero(alphas.A>0)[0]
    sVs=datMat[svInd] 
    labelSV = labelMat[svInd];
    print "there are %d Support Vectors" % shape(sVs)[0]
    m,n = shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],kTup)
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!=sign(labelArr[i]): errorCount += 1
    print "the training error rate is: %f" % (float(errorCount)/m)
    dataArr,labelArr = loadImages('testDigits')
    errorCount = 0
    datMat=mat(dataArr); labelMat = mat(labelArr).transpose()
    m,n = shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],kTup)
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!=sign(labelArr[i]): errorCount += 1    
    print "the test error rate is: %f" % (float(errorCount)/m) 


'''#######********************************
Non-Kernel VErsions below
'''#######********************************

class optStructK:
    def __init__(self,dataMatIn, classLabels, C, toler):  # Initialize the structure with the parameters 
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m,1)))
        self.b = 0
        self.eCache = mat(zeros((self.m,2))) #first column is valid flag
        
def calcEkK(oS, k):
    fXk = float(multiply(oS.alphas,oS.labelMat).T*(oS.X*oS.X[k,:].T)) + oS.b
    Ek = fXk - float(oS.labelMat[k])
    return Ek
        
def selectJK(i, oS, Ei):         #this is the second choice -heurstic, and calcs Ej
    maxK = -1; maxDeltaE = 0; Ej = 0
    oS.eCache[i] = [1,Ei]  #set valid #choose the alpha that gives the maximum delta E
    validEcacheList = nonzero(oS.eCache[:,0].A)[0]
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:   #loop through valid Ecache values and find the one that maximizes delta E
            if k == i: continue #don't calc for i, waste of time
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):
                maxK = k; maxDeltaE = deltaE; Ej = Ek
        return maxK, Ej
    else:   #in this case (first time around) we don't have any valid eCache values
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej

def updateEkK(oS, k):#after any alpha has changed update the new value in the cache
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1,Ek]
        
def innerLK(i, oS):
    Ei = calcEk(oS, i)
    if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)):
        j,Ej = selectJ(i, oS, Ei) #this has been changed from selectJrand
        alphaIold = oS.alphas[i].copy(); alphaJold = oS.alphas[j].copy();
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L==H: print "L==H"; return 0
        eta = 2.0 * oS.X[i,:]*oS.X[j,:].T - oS.X[i,:]*oS.X[i,:].T - oS.X[j,:]*oS.X[j,:].T
        if eta >= 0: print "eta>=0"; return 0
        oS.alphas[j] -= oS.labelMat[j]*(Ei - Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)
        updateEk(oS, j) #added this for the Ecache
        if (abs(oS.alphas[j] - alphaJold) < 0.00001): print "j not moving enough"; return 0
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])#update i by the same amount as j
        updateEk(oS, i) #added this for the Ecache                    #the update is in the oppostie direction
        b1 = oS.b - Ei- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.X[i,:]*oS.X[i,:].T - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.X[i,:]*oS.X[j,:].T
        b2 = oS.b - Ej- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.X[i,:]*oS.X[j,:].T - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.X[j,:]*oS.X[j,:].T
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]): oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]): oS.b = b2
        else: oS.b = (b1 + b2)/2.0
        return 1
    else: return 0

def smoPK(dataMatIn, classLabels, C, toler, maxIter):    #full Platt SMO
    oS = optStruct(mat(dataMatIn),mat(classLabels).transpose(),C,toler)
    iter = 0
    entireSet = True; alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:   #go over all
            for i in range(oS.m):        
                alphaPairsChanged += innerL(i,oS)
                print "fullSet, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged)
            iter += 1
        else:#go over non-bound (railed) alphas
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i,oS)
                print "non-bound, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged)
            iter += 1
        if entireSet: entireSet = False #toggle entire set loop
        elif (alphaPairsChanged == 0): entireSet = True  
        print "iteration number: %d" % iter
    return oS.b,oS.alphas


# SVM梯度下降进参数求解
import matplotlib.pyplot as plt  
import numpy as np  
import tensorflow as tf  
from sklearn import datasets  
from tensorflow.python.framework import ops  
ops.reset_default_graph()  
  
# Create graph  
sess = tf.Session()  
  
# Load the data  
# iris.data = [(Sepal Length, Sepal Width, Petal Length, Petal Width)]  
iris = datasets.load_iris()  
x_vals = np.array([[x[0], x[3]] for x in iris.data])  
y_vals = np.array([1 if y==0 else -1 for y in iris.target])  
  
# Split data into train/test sets  
train_indices = np.random.choice(len(x_vals), round(len(x_vals)*0.8), replace=False)  
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))  
x_vals_train = x_vals[train_indices]  
x_vals_test = x_vals[test_indices]  
y_vals_train = y_vals[train_indices]  
y_vals_test = y_vals[test_indices]  
  
# Declare batch size  
batch_size = 100  
  
# Initialize placeholders  
x_data = tf.placeholder(shape=[None, 2], dtype=tf.float32)  
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)  
  
# Create variables for linear regression  
A = tf.Variable(tf.random_normal(shape=[2,1]))  
b = tf.Variable(tf.random_normal(shape=[1,1]))  
  
# Declare model operations  
model_output = tf.subtract(tf.matmul(x_data, A), b)  

# 定义 hinge loss function
# Declare vector L2 'norm' function squared  
l2_norm = tf.reduce_sum(tf.square(A))  

# Declare loss function  
# Loss = max(0, 1-pred*actual) + alpha * L2_norm(A)^2  
# L2 regularization parameter, alpha  
alpha = tf.constant([0.01])  
# Margin term in loss  
classification_term = tf.reduce_mean(tf.maximum(0., tf.subtract(1., tf.multiply(model_output, y_target))))  
# Put terms together  
loss = tf.add(classification_term, tf.multiply(alpha, l2_norm))  
  
# Declare prediction function  
prediction = tf.sign(model_output)  
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, y_target), tf.float32))  
  
# Declare optimizer  
my_opt = tf.train.GradientDescentOptimizer(0.01)  
train_step = my_opt.minimize(loss)  
  
# Initialize variables  
init = tf.global_variables_initializer()  
sess.run(init)  
  
# Training loop  
loss_vec = []  
train_accuracy = []  
test_accuracy = []  
for i in range(500):  
    rand_index = np.random.choice(len(x_vals_train), size=batch_size)  
    rand_x = x_vals_train[rand_index]  
    rand_y = np.transpose([y_vals_train[rand_index]])  
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})  
      
    temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})  
    loss_vec.append(temp_loss)  
      
    train_acc_temp = sess.run(accuracy, feed_dict={x_data: x_vals_train, y_target: np.transpose([y_vals_train])})  
    train_accuracy.append(train_acc_temp)  
      
    test_acc_temp = sess.run(accuracy, feed_dict={x_data: x_vals_test, y_target: np.transpose([y_vals_test])})  
    test_accuracy.append(test_acc_temp)  
      
    if (i+1)%100==0:  
        print('Step #' + str(i+1) + ' A = ' + str(sess.run(A)) + ' b = ' + str(sess.run(b)))  
        print('Loss = ' + str(temp_loss))  
  
# Extract coefficients  
[[a1], [a2]] = sess.run(A)  
[[b]] = sess.run(b)  
slope = -a2/a1  
y_intercept = b/a1  
  
# Extract x1 and x2 vals  
x1_vals = [d[1] for d in x_vals]  
  
# Get best fit line  
best_fit = []  
for i in x1_vals:  
  best_fit.append(slope*i+y_intercept)  
  
# Separate I. setosa  
setosa_x = [d[1] for i,d in enumerate(x_vals) if y_vals[i]==1]  
setosa_y = [d[0] for i,d in enumerate(x_vals) if y_vals[i]==1]  
not_setosa_x = [d[1] for i,d in enumerate(x_vals) if y_vals[i]==-1]  
not_setosa_y = [d[0] for i,d in enumerate(x_vals) if y_vals[i]==-1]  
  
# Plot data and line  
plt.plot(setosa_x, setosa_y, 'o', label='I. setosa')  
plt.plot(not_setosa_x, not_setosa_y, 'x', label='Non-setosa')  
plt.plot(x1_vals, best_fit, 'r-', label='Linear Separator', linewidth=3)  
plt.ylim([0, 10])  
plt.legend(loc='lower right')  
plt.title('Sepal Length vs Pedal Width')  
plt.xlabel('Pedal Width')  
plt.ylabel('Sepal Length')  
plt.show()  
  
# Plot train/test accuracies  
plt.plot(train_accuracy, 'k-', label='Training Accuracy')  
plt.plot(test_accuracy, 'r--', label='Test Accuracy')  
plt.title('Train and Test Set Accuracies')  
plt.xlabel('Generation')  
plt.ylabel('Accuracy')  
plt.legend(loc='lower right')  
plt.show()  
  
# Plot loss over time  
plt.plot(loss_vec, 'k-')  
plt.title('Loss per Generation')  
plt.xlabel('Generation')  
plt.ylabel('Loss')  
plt.show()  