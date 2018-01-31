

我想说：

学习dl不去深层的扒扒，没有理论的支撑是不行的，今天分享一篇笔记，另加读者的心得，很棒。

读者分享数据挖掘心得：

我跟你讲一下在实际项目中我们是怎么做数据挖掘的。1:定义业务问题，很多人认为机器学习越高大上的算法越厉害，其实不是这样的，每类算法都有特定的业务场景。机器学习主要分为有监督无监督和半监督，当拿到业务问题时，要看业务场景下哪类算法比较好。比如做风控我们会用决策树，做点击率预估我们会用LR。这里你要清楚每个算法的优缺点，比如为什么我要用决策树不用随机森林，为什么用LR不用SVM 2:根据模型做数据的收集和整合(比如爬虫，建立数据仓库，用户画像，使用spark做数据统计和清洗等等) 3:拿到数据以后，怎么建立有效的特征 因为数据不可能都是完整的，会有缺失值和异常值 这个时候需要根据业务做一些业务场景下的替代，比如用平均值代替缺失值，用中值代替异常值  4:数据特征的向量化表示 比如LR,LR这个模型要求输入的数据必须是0到1之间的，但是我们的数据不可能都是0到1之间的，这个时候就需要对数据进行向量化表示(比如离散化也叫做one hot encoding，归一化)文本数据使用(tf-idf word2vec)等等  5:建立有效的损失函数 把数据跑到LR中，需要一种方法来迭代数据的误差，比如Logloss function 我们的目的就是不断迭代求出误差的最小值 6:怎么快速求出模型 这里比如离线数据下我们会使用梯度下降算法迭代模型 实时数据下我们会使用ftrl算法迭代模型 7:模型的评估 比如使用AUC  8:模型的调整 比如过拟合我们会使用正则项，pca降维 这里比如会用交叉验证算出正则向的系数 其实大部分数据挖掘场景下都是这个套路。

下面是cnn权重更新详解：

![image](http://upload-images.jianshu.io/upload_images/4618424-297fefaff2cafacf?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![image](http://upload-images.jianshu.io/upload_images/4618424-f90621471d133cf8?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![image](http://upload-images.jianshu.io/upload_images/4618424-f14605a48d867181?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![image.png](http://upload-images.jianshu.io/upload_images/4618424-28d59074a5c792d6.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


![image.png](http://upload-images.jianshu.io/upload_images/4618424-e2bd8d2db6baee8b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


还可以参考：

1. 反向传播和它的直观理解：http://blog.csdn.net/u014365862/article/details/54728707

2\. UFLDL教程：http://deeplearning.stanford.edu/wiki/index.php/UFLDL教程

3. http://www.moonshile.com/post/juan-ji-shen-jing-wang-luo-quan-mian-jie-xi#toc_11

推荐阅读：

1. [MachineLN之三要素](http://mp.weixin.qq.com/s?__biz=MzU3MTM3MTIxOQ==&mid=2247483841&idx=2&sn=e4a3cff7b12c48af237c577c487ba3a1&chksm=fce07a7dcb97f36be5003c3018b3a391070bdc4e56839cb461d226113db4c5f24032e0bf5809&scene=21#wechat_redirect)

2. [MachineLN之模型评估](http://mp.weixin.qq.com/s?__biz=MzU3MTM3MTIxOQ==&mid=2247483872&idx=2&sn=8436e1eb9055d3a372278ee8688cd703&chksm=fce07a5ccb97f34a4490f60304b206c741d2395149c2c2e68bddb3faf7daf9121ca27a5d6a97&scene=21#wechat_redirect)

3. [MachinLN之dl](http://mp.weixin.qq.com/s?__biz=MzU3MTM3MTIxOQ==&mid=2247483894&idx=2&sn=63333c02674e15e84159e064073fe563&chksm=fce07a4acb97f35cc38f75dc891a19129e2406270d04b739cfa9b8a28f9780b4e2a65a7cd39b&scene=21#wechat_redirect)

4. [DeepLN之CNN解析](http://mp.weixin.qq.com/s?__biz=MzU3MTM3MTIxOQ==&mid=2247483906&idx=1&sn=2eceda7d9703d5315638739e04d5b6e7&chksm=fce079becb97f0a8b8dd2e34a9e757f23757cf2699c397707bfaa677c9f8204c91508840d8f7&scene=21#wechat_redirect)

![image](http://upload-images.jianshu.io/upload_images/4618424-3428c5287d46ebb9?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)



版权声明：本文为博主原创文章，未经博主允许不得转载。有问题可以加微信：lp9628(注明CSDN)。
