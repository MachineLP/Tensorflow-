# coding=utf-8
'''
Created on Oct 12, 2010
Decision Tree Source Code for Machine Learning in Action Ch. 3
@author: Peter Harrington
'''
from math import log
import operator

# 创建数据集
def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    #change to discrete values
    return dataSet, labels

# 计算信息熵，
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    # 统计样本集种的每个类别的数量。
    for featVec in dataSet: #the the number of unique elements and their occurance
        # 每个样本的最后一位是类别信息。获取得到类别信息，用于字典中各类别的数量统计。
        currentLabel = featVec[-1]
        # 判断字典中的已有keys，是否含有该类别， 如果不在字典中， 则进行添加， 并且设置values为0。
        # 这里就是指类别的数量信息。
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    # 对个类别的信息熵求和用
    shannonEnt = 0.0
    # 计算信息熵。
    for key in labelCounts:
        # 每个类别数量占总样本数量的比例。
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob,2) 
    return shannonEnt

# 划分数据集： 第几个特征， 特征值。 
# 根据某个特征的特征值划分数据集。
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    # 遍历数据集中的每一个样本：
    for featVec in dataSet:
        # 判断该样本的第 axis为的特征值 是否 等于 value。
        # 如果相等，则进行保存， 很可能就是此节点value的子节点。
        # 保存下来的用于计算条件熵。
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]    
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

# 选取特征计算信息增益，选取信息增益最大的特征作为此时最好的分类特征。
def chooseBestFeatureToSplit(dataSet):
    # 计算总的特征数量， 一个样本的维度 = 特征+标签， 所以要减1。
    numFeatures = len(dataSet[0]) - 1     
    # 计算信息熵， 用于下面计算信息增益。
    baseEntropy = calcShannonEnt(dataSet)
    # 用于保存最大的信息增益； 和信息增益最大的时候的特征。
    bestInfoGain = 0.0; bestFeature = -1
    # 遍历所有特征
    for i in range(numFeatures):        #iterate over all the features
        # 所有样本的一个特征， 用于计算该特征的所有特征值 的信息增益。
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)    
        newEntropy = 0.0
        # 计算条件熵
        # 遍历该特征的所有特征值。
        for value in uniqueVals:
            # 已i特征的value特征值进行划分数据， 计算含有该特征值的样本数量占总数的比例。
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            # 累加计算条件熵
            newEntropy += prob * calcShannonEnt(subDataSet)     
        # 计算信息增益：熵 - 条件熵
        infoGain = baseEntropy - newEntropy     
        # 记录最大的信息熵， 和最大信息熵时候的特征。
        if (infoGain > bestInfoGain):       
            bestInfoGain = infoGain        
            bestFeature = i
    return bestFeature                 


# 统计个类别的数量， 进行排序， 返回数量最多的类别。
def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

# 递归创建决策树
def createTree(dataSet,labels):
    # 获取所有样本的类别
    classList = [example[-1] for example in dataSet]
    # 此时所有的样本属于同一类
    if classList.count(classList[0]) == len(classList): 
        return classList[0]#stop splitting when all of the classes are equal
    # 如果特征已为空， 将类别数量最多的作为该节点的类别。
    if len(dataSet[0]) == 1: #stop splitting when there are no more features in dataSet
        return majorityCnt(classList)
    # 选取最好的特征索引作为此时的节点。
    bestFeat = chooseBestFeatureToSplit(dataSet)
    # 通过索引获取到最好的特征标识。用于下面字典记录书的节点。
    bestFeatLabel = labels[bestFeat]
    # 决策树
    myTree = {bestFeatLabel:{}}
    # 将最好的特征删除掉， 该特征后边已经用不到了， 将递归的重新选取除此之外的下一个特征，最为树的节点。
    del(labels[bestFeat])
    # 基于最好的特征继续构建数。
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    # 遍历最好特征的每一个特征值继续构建树。
    for value in uniqueVals:
        subLabels = labels[:]      
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),subLabels)
    return myTree                            

# 通过构建的决策树对新数据进行分类；
def classify(inputTree,featLabels,testVec):
    # 获取树的第一个keys
    firstStr = inputTree.keys()[0]
    # 通过keys获取values
    secondDict = inputTree[firstStr]
    # 通过keys， 获取特征列表中的特征的索引
    featIndex = featLabels.index(firstStr)
    # 通过特征索引获取测试数据中的特征值
    key = testVec[featIndex]
    # 根据上面获取到的特征值， 确定下一步往树的哪个分支走， 相当于又重新进入到下一个树， 递归上面的步骤。
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict): 
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else: classLabel = valueOfFeat
    return classLabel

# 保存树
def storeTree(inputTree,filename):
    import pickle
    fw = open(filename,'w')
    pickle.dump(inputTree,fw)
    fw.close()

# 加载树    
def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)
    

# 测试代码
myData, labels = createDataSet()
myTree = createTree(myData, labels)
print (myTree)
storeTree(myTree, 'myTree.pkl')


myTree = grabTree('myTree.pkl')
predict = classify(myTree, ['no surfacing','flippers'], [1,1])
print (predict)
