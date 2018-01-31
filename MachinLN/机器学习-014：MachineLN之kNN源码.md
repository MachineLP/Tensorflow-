我想说： 

其实你坚持的东西都是有意义的，就好比给代码加注释，你真去做了就知道了；另外建议大家建立一个自己的工具箱，就像我建立一个MachineLP_tools，里面放可以复用的代码，然后支持大家开源，你手上现有的代码可能很快就会过时，希望能够在有效期内发挥最大的功效；最后建议自己搭建一个自己的机器学习训练框架，用着顺手、用着舒服、用着放心，比起相信别人，相信大家更相信自己。

下面是kNN代码和详细注释， 又是截图，哈哈，虽然好代码都是敲出来的但是需要文本的可以私聊我；kNN原理参考：[MachineLN之kNN](http://mp.weixin.qq.com/s?__biz=MzU3MTM3MTIxOQ==&mid=2247484116&idx=1&sn=b26ee03d2e520aa429e424950e59943d&chksm=fce07968cb97f07e6414d3dad8f1744c39cf99bc4f0f68ead7d153f33bf39b46a513d6223ee2&scene=21#wechat_redirect)：

```
# 下面就是kNN的核心公式，每次给代码加注释都能想起李皓宇老师  
def classify0(inX, dataSet, labels, k):  
    # 在矩阵中我们一般说行和列，而在图像中我们说的是宽和高，但是宽对应的是列，高对应的是行;  
    # 首先获取有样本集的数量，为什么？ 这下一句就能看出来，为了给新的输入样本做广播，进行向量与矩阵的减法;  
    dataSetSize = dataSet.shape[0]  
    # dataSetSize行，1列； 进行矩阵减法;  
    diffMat = tile(inX, (dataSetSize,1)) - dataSet  
    # 下面三行就是平方求和取根号， 计算欧式距离; 也就是计算样本间的相似度;  
    sqDiffMat = diffMat**2  
    # 求和  
    sqDistances = sqDiffMat.sum(axis=1)  
    # 取根a号  
    distances = sqDistances**0.5  
    # 对新样本与各类别样本计算距离后排序，然后返回排序后的索引; （默认的是升序排列）  
    sortedDistIndicies = distances.argsort()  
    # 定义一个字典， 排序后的前k个值中含有类别的个数;  
    classCount={}  
    # 下面就是kNN中的k;            
    for i in range(k):  
        # 下面就是根据排序后的索引值，取出对应样本的标签值;  
        voteIlabel = labels[sortedDistIndicies[i]]  
        # 统计前排序后k个中个类别的数量; 这里的get用的很妙，前几天刚听同事提过，学习了;  
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1  
    # 对字典按照value进行排序; python2中可以用下面的方式，貌似python3貌似用不了;  
    # 如果没有好的方法; 那么可以用我这种比较笨拙的方法：但是返回的是列表;  
    ''''' 
    def dict2list(dic:dict):   
        #将字典转化为列表  
        keys = dic.keys()   
        vals = dic.values()   
        lst = [(key, val) for key, val in zip(keys, vals)]   
        return lst   
    # lambda生成一个临时函数   
    # d表示字典的每一对键值对，d[0]为key，d[1]为value   
    # reverse为True表示降序排序   
    stat = sorted(dict2list(stat), key=lambda d:d[1], reverse=True) '''  
    # 降序排列;   
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)  
    # 取类别中数量最多的作为新样本的类别;  
    return sortedClassCount[0][0]  
  
# 创建一个简单的数据集  
def createDataSet():  
    # 定义一个二维数组; 作为已知标签样本集  
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])  
    # 下面就是标签  
    labels = ['A','A','B','B']  
    return group, labels  
  
# 将文件中的样本数据生成矩阵; 或者说是二维数组;  
def file2matrix(filename):  
    fr = open(filename)  
    # 计算样本的数量; 用于初始化保存样本集的二维数组;  
    numberOfLines = len(fr.readlines())         #get the number of lines in the file  
    # 定义返回样本的集二维数组; 只取了3个数据所以定义为3列;  
    returnMat = zeros((numberOfLines,3))        #prepare matrix to return  
    # 存放对应的标签;  
    classLabelVector = []                       #prepare labels return     
    fr = open(filenamea)  
    index = 0  
    # 读取文件中的每一行数据;  
    for line in fr.readlines():  
        # 去掉开头和结尾的符号的  
        line = line.strip()  
        # 由于文件中各数据是已制表符分开的，所以已制表符进行切割;  
        listFromLine = line.split('\t')  
        # 前三个是样本中的数据; 最后一个是标签;  
        returnMat[index,:] = listFromLine[0:3]  
        classLabelVector.append(int(listFromLine[-1]))  
        # 记录数组中行的索引  
        index += 1  
    return returnMat,classLabelVector  
  
  
# 对数据做归一化处理:  
def autoNorm(dataSet):  
    # 计算样本中的最小值和最大值，用于进行归一化操作;  
    minVals = dataSet.min(0)  
    maxVals = dataSet.max(0)  
    ranges = maxVals - minVals  
    # 定义一个接收样本归一化后数据的数组; 和样本集的行数和列数是相同的;  
    normDataSet = zeros(shape(dataSet))  
    # 获取样本的集的数量; 用于对最小值 和 ranges进行广播; 用于后面矩阵的减法 和 除法;  
    m = dataSet.shape[0]  
    # 归一化的操作; 此处有问题; 可以查看 MachineLN之样本归一化;  
    normDataSet = dataSet - tile(minVals, (m,1))  
    normDataSet = normDataSet/tile(ranges, (m,1))   #element wise divide  
    return normDataSet, ranges, minVals  
  
  
# kNN 分类   
def datingClassTest():  
    # 设置一个比率，在0到1之间，用于将样本集设置多少训练集和多少为测试集;   
    hoRatio = 0.50      #hold out 10%  
    # 这个上面已经解释过了， 将文件中的样本数据生成矩阵; 或者说是二维数组;  
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')       #load data setfrom file  
    # 进行样本归一化，切记用的时候不要忘记;  
    normMat, ranges, minVals = autoNorm(datingDataMat)  
    # 获取样本集的行数; 即样本集的数量;  
    m = normMat.shape[0]  
    # 根据设置的比例，计算用于测试的样本的数量，同时用于训练的样本的数量就有了;  
    numTestVecs = int(m*hoRatio)  
    # 用书输出错误的数量;  
    errorCount = 0.0  
    # 训练计算每个测试样本的标签值;  
    for i in range(numTestVecs):  
        # 取第i个测试样本，通过训练集计算新样本的类别;  
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)  
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i])  
        if (classifierResult != datingLabels[i]): errorCount += 1.0  
    print "the total error rate is: %f" % (errorCount/float(numTestVecs))  
    print errorCount  
  
  
# 下面内容是处理图像的数据;  
def img2vector(filename):  
    # 用来存储每个图片拉成向量后的结果，这里的图片是32*32所以是1024;  
    returnVect = zeros((1,1024))  
    #   
    fr = open(filename)  
    # 二维矩阵的取值，或者是二维数组的取值;  
    for i in range(32):  
        # 在文件中每行存储的是图像的一行像素;  
        lineStr = fr.readline()  
        for j in range(32):  
            # 取每一个像素;  
            returnVect[0,32*i+j] = int(lineStr[j])  
    return returnVect  
  
  
# 下面是手写体数字通过kNN进行识别;  
def handwritingClassTest():  
    # 用于存放样本类别  
    hwLabels = [a]  
    # 由于每张手写体的图是放在一个文件中；要将所有文件生成样本集；  
    # 获取文件夹中的所有文件;  
    trainingFileList = listdir('trainingDigits')           #load the training set  
    # 获取文件数量;  
    m = len(trainingFileList)  
    # 定义一个数组， m行（手写体数量），1024像素数;  
    trainingMat = zeros((m,1024))  
    # 训练从文件中读图像;  
    for i in range(m):  
        # 取每一个文件， 根据文件名字解析出来标签;  
        fileNameStr = trainingFileList[i]  
        fileStr = fileNameStr.split('.')[0]     #take off .txt  
        classNumStr = int(fileStr.split('_')[0])  
        hwLabels.append(classNumStr)  
        # 将每个文件中存放的手写体数字，转化为向量;  
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)  
    # 对测试数据进行相同的操作;  
    testFileList = listdir('testDigits')        #iterate through the test set  
    errorCount = 0.0  
    mTest = len(testFileList)  
    for i in range(mTest):  
        fileNameStr = testFileList[i]  
        fileStr = fileNameStr.split('.')[0]     #take off .txt  
        classNumStr = int(fileStr.split('_')[0])  
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)  
        # 取第i个测试样本，通过训练集计算新样本的类别; k值仍为3  
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)  
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr)  
        if (classifierResult != classNumStr): errorCount += 1.0  
    print "\nthe total number of errors is: %d" % errorCount  
    print "\nthe total error rate is: %f" % (errorCount/float(mTest))  
```


推荐阅读：

1. [MachineLN之三要素](http://mp.weixin.qq.com/s?__biz=MzU3MTM3MTIxOQ==&mid=2247483841&idx=2&sn=e4a3cff7b12c48af237c577c487ba3a1&chksm=fce07a7dcb97f36be5003c3018b3a391070bdc4e56839cb461d226113db4c5f24032e0bf5809&scene=21#wechat_redirect)

2. [MachineLN之模型评估](http://mp.weixin.qq.com/s?__biz=MzU3MTM3MTIxOQ==&mid=2247483872&idx=2&sn=8436e1eb9055d3a372278ee8688cd703&chksm=fce07a5ccb97f34a4490f60304b206c741d2395149c2c2e68bddb3faf7daf9121ca27a5d6a97&scene=21#wechat_redirect)

3. [MachinLN之dl](http://mp.weixin.qq.com/s?__biz=MzU3MTM3MTIxOQ==&mid=2247483894&idx=2&sn=63333c02674e15e84159e064073fe563&chksm=fce07a4acb97f35cc38f75dc891a19129e2406270d04b739cfa9b8a28f9780b4e2a65a7cd39b&scene=21#wechat_redirect)

4. [DeepLN之CNN解析](http://mp.weixin.qq.com/s?__biz=MzU3MTM3MTIxOQ==&mid=2247483906&idx=1&sn=2eceda7d9703d5315638739e04d5b6e7&chksm=fce079becb97f0a8b8dd2e34a9e757f23757cf2699c397707bfaa677c9f8204c91508840d8f7&scene=21#wechat_redirect)

[5\. DeepLN之手撕CNN权值更新（笔记）](http://mp.weixin.qq.com/s?__biz=MzU3MTM3MTIxOQ==&mid=2247483927&idx=1&sn=132b90eb009022d2d792303e6d824df7&chksm=fce079abcb97f0bd9b893889a7c8232c0254e447277b930ab8e35ea1f9c49e57c0fc66bab239&scene=21#wechat_redirect)

6. [DeepLN之CNN源码](http://mp.weixin.qq.com/s?__biz=MzU3MTM3MTIxOQ==&mid=2247483948&idx=1&sn=d1edce6e99dac0437797404d15714876&chksm=fce07990cb97f086f2f24ec8b40bb64b588ce657e6ae8d352d4d20e856b5f0e126eaffc5ec99&scene=21#wechat_redirect)

7. [MachineLN之激活函数](http://mp.weixin.qq.com/s?__biz=MzU3MTM3MTIxOQ==&mid=2247483968&idx=1&sn=dc2e52c68cd8ea9037b114625c9b1a33&chksm=fce079fccb97f0ea4f3b8f8c74cb779613e06c54a718378c9d174651c16cabb28a1c283b9083&scene=21#wechat_redirect) 

8. [MachineLN之BN](http://mp.weixin.qq.com/s?__biz=MzU3MTM3MTIxOQ==&mid=2247483979&idx=1&sn=3b5cfedc2d475f69e52656e50ff44f36&chksm=fce079f7cb97f0e1217a6b7222cef60930f79f3870692315935dab7abefff32af6b4201c07ab&scene=21#wechat_redirect)

9. [MachineLN之数据归一化](http://mp.weixin.qq.com/s?__biz=MzU3MTM3MTIxOQ==&mid=2247483996&idx=1&sn=bc692328396199f3e192836916c10cd0&chksm=fce079e0cb97f0f68fc3f2a2be81430a3d0a27f68a6f6cbe07fdbd39ca48d15ead942367e013&scene=21#wechat_redirect)

10. [MachineLN之样本不均衡](http://mp.weixin.qq.com/s?__biz=MzU3MTM3MTIxOQ==&mid=2247484011&idx=1&sn=c7ee568af41c28793162e3fa2084c5b6&chksm=fce079d7cb97f0c175e3561b4c999101268984574fea5adc04b1e2ada99bee4277817f969484&scene=21#wechat_redirect)

11. [MachineLN之过拟合](http://mp.weixin.qq.com/s?__biz=MzU3MTM3MTIxOQ==&mid=2247484026&idx=1&sn=36329f7ee26fb675627d64884ac02d0f&chksm=fce079c6cb97f0d05bb1d67218203bfc4b1234a47404fc9e38399c2f0aca9fe21455c0a97b7c&scene=21#wechat_redirect)

[12\. MachineLN之优化算法](http://mp.weixin.qq.com/s?__biz=MzU3MTM3MTIxOQ==&mid=2247484064&idx=1&sn=f834585e2e77299f28148084e6fddcbc&chksm=fce0791ccb97f00a1c168fc05725c59f9e5be5fd22249e3c49e949d5daf812a9ca9da365b50e&scene=21#wechat_redirect) ReinforcementNL之图像分类

13. [ReinforcementLN之图像分类](http://mp.weixin.qq.com/s?__biz=MzU3MTM3MTIxOQ==&mid=2247484102&idx=1&sn=a44c0efaa3d939d148cc0d3dd2dc06b9&chksm=fce0797acb97f06c83cf33f711165dcef005dd54478503098f904e41800fe3beb855a579ed78&scene=21#wechat_redirect)

14. [MachineLN之kNN](http://mp.weixin.qq.com/s?__biz=MzU3MTM3MTIxOQ==&mid=2247484116&idx=1&sn=b26ee03d2e520aa429e424950e59943d&chksm=fce07968cb97f07e6414d3dad8f1744c39cf99bc4f0f68ead7d153f33bf39b46a513d6223ee2&scene=21#wechat_redirect)

![image](http://upload-images.jianshu.io/upload_images/4618424-198fded0b528089d?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


版权声明：本文为博主原创文章，未经博主允许不得转载。有问题可以加微信：lp9628(注明CSDN)。

