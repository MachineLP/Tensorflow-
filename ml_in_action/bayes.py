'''
Created on Oct 19, 2010

@author: Peter
'''
from numpy import *

# 构建一个简单的文本数据集， 包含两个类别。
def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return postingList,classVec

# 通过所有样本， 创建词典列表； 用于后面的词转向量。                 
def createVocabList(dataSet):
    # 构建一个空的集合
    vocabSet = set([])  #create empty set
    # 遍历样本集中的所有样本。
    for document in dataSet:
        # 取集合的并集。
        vocabSet = vocabSet | set(document) #union of the two sets
    # 集合转列表。
    return list(vocabSet)

# 词转向量。 此处将每一个样本转成一个向量。
def setOfWords2Vec(vocabList, inputSet):
    # 创建一个和词典长度相同 全为0的向量， 用于样本的词再词典中出现，则词典中的相应位置值为1。
    returnVec = [0]*len(vocabList)
    # 遍历一个样本中的所有词。
    for word in inputSet:
        # 如果该词出现再字典中，对应位置1。
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else: print "the word: %s is not in my Vocabulary!" % word
    return returnVec

# 进行训练， 这里就是计算： 条件概率 和 先验概率
def trainNB0(trainMatrix,trainCategory):
    # 计算总的样本数量
    numTrainDocs = len(trainMatrix)
    # 计算样本向量化后的长度， 这里等于词典长度。
    numWords = len(trainMatrix[0])
    # 计算先验概率
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    # 进行初始化， 用于向量化后的样本 累加， 为什么初始化1不是全0， 防止概率值为0.  
    p0Num = ones(numWords); p1Num = ones(numWords)      #change to ones() 
    # 初始化求条件概率的分母为2， 防止出现0，无法计算的情况。
    p0Denom = 2.0; p1Denom = 2.0                        #change to 2.0
    # 遍历所有向量化后的样本， 并且每个向量化后的长度相等， 等于词典长度。
    for i in range(numTrainDocs):
        # 统计标签为1的样本： 向量化后的样本的累加， 样本中1总数的求和， 最后相除取log就是条件概率。 
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        # 统计标签为0的样本： 向量化后的样本累加， 样本中1总数的求和， 最后相除取log就是条件概率。 
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # 求条件概率。
    p1Vect = log(p1Num/p1Denom)          #change to log()
    p0Vect = log(p0Num/p0Denom)          #change to log()
    # 返回条件概率 和 先验概率
    return p0Vect,p1Vect,pAbusive

# 通过条件概率 和 先验概率 对新的样本进行向量化后分类。
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    # 向量化后的样本 分别 与 各类别的条件概率相乘 加上 先验概率取log， 之后进行大小比较， 输出类别。
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)    #element-wise mult
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else: 
        return 0

# 通过所有样本， 创建词典列表； 用于后面的词转向量。  
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

# 测试程序
def testingNB():
    # 生成训练样本 和 标签
    listOPosts,listClasses = loadDataSet()
    # 创建词典
    myVocabList = createVocabList(listOPosts)
    # 用于保存样本转向量之后的
    trainMat=[]
    # 遍历每一个样本， 转向量后， 保存到列表中。
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    # 计算 条件概率 和 先验概率
    p0V,p1V,pAb = trainNB0(array(trainMat),array(listClasses))
    # 给定测试样本 进行测试
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb)
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb)

# 将字符串中的词 转成列表
def textParse(bigString):    #input is big string, #output is word list
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2] 

# 垃圾邮件分类的代码    
def spamTest():
    # 存放转列表后的样本，用于创建词典。存放给定的标签值。
    docList=[]; classList = []; fullText =[]
    # 从26个文件中读取邮件内容。
    for i in range(1,26):
        # 从每个文件中读取内容，将字符转为列表
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        # 设定标签
        classList.append(1)
        # 从每个文件中读取内容，将字符转为列表
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        # 设定标签
        classList.append(0)
    # 根据处理后的样本数据创建词典
    vocabList = createVocabList(docList)#create vocabulary
    # 设定训练样本 和 测试样本的 索引
    trainingSet = range(50); testSet=[]           #create test set
    for i in range(10):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])  
    trainMat=[]; trainClasses = []
    # 获取训练样本 和 标签， 将样本通过词典转为向量。
    for docIndex in trainingSet:#train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    # 计算条件概率 和 先验概率
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
    errorCount = 0
    # 获取测试样本， 并将测试样本转为向量。
    for docIndex in testSet:        #classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        # 预测类别， 并且计算预测错误的数量
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
            print "classification error",docList[docIndex]
    print 'the error rate is: ',float(errorCount)/len(testSet)
    #return vocabList,fullText

# 统计词典中的各词在样本中的数量， 进行排序，只取前30个。
def calcMostFreq(vocabList,fullText):
    import operator
    freqDict = {}
    # 遍历字典中的所有词
    for token in vocabList:
        # 获取每个词出现的次数
        freqDict[token]=fullText.count(token)
    # 进行排序， 只取出现次数最多的前30个
    sortedFreq = sorted(freqDict.iteritems(), key=operator.itemgetter(1), reverse=True) 
    return sortedFreq[:30]       

# 下面代码根本和上面重复。 
def localWords(feed1,feed0):
    import feedparser
    docList=[]; classList = []; fullText =[]
    minLen = min(len(feed1['entries']),len(feed0['entries']))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1) #NY is class 1
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)#create vocabulary
    top30Words = calcMostFreq(vocabList,fullText)   #remove top 30 words
    for pairW in top30Words:
        if pairW[0] in vocabList: vocabList.remove(pairW[0])
    trainingSet = range(2*minLen); testSet=[]           #create test set
    for i in range(20):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])  
    trainMat=[]; trainClasses = []
    for docIndex in trainingSet:#train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
    errorCount = 0
    for docIndex in testSet:        #classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
    print 'the error rate is: ',float(errorCount)/len(testSet)
    return vocabList,p0V,p1V

# 显示每个类别的常用词 
def getTopWords(ny,sf):
    import operator
    vocabList,p0V,p1V=localWords(ny,sf)
    topNY=[]; topSF=[]
    # 选择条件概率大于 -6.0 的，保存下来， 排序后打印。
    for i in range(len(p0V)):
        if p0V[i] > -6.0 : topSF.append((vocabList[i],p0V[i]))
        if p1V[i] > -6.0 : topNY.append((vocabList[i],p1V[i]))
    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
    print "SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**"
    for item in sortedSF:
        print item[0]
    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print "NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**"
    for item in sortedNY:
        print item[0]
