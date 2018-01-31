我想说：

其实训练模型是个力气活，有人说训练模型很简单，把数据塞进去，然后跑完就好了，哦，这样的话谁都会，关键的也就在这里，同样的数据同样的模型，有些人训练的模型在测试集上99%，有些人的则只有95%，甚至90%，其实学习最关键的也在这里，大家同时学一个知识，也都学了，但是理解的程度会大相径庭，注意trick不可不学，并且坑不得不踩。唉，前几天训练好的一个模型，再让自己复现感觉也很难搞定了，天时地利人和！！！今天开始搞传统机器学习的理论和实践，突然发现这是自己的短板，其实也不是啦：李航老师统计学看了4遍，周志华老师机器学习看了一遍，模式分类那本大厚书粗略看了一遍，经典的数据挖掘看了一遍，还看了一本机器学习的忘记名字了，吴恩达的课看了一遍，还看了一些英文资料，机器学习实践照着敲了一遍，在就是一些零零碎碎的.....，虽然做过一些实践，但是缺乏工程上的磨练。

1\. kNN介绍

kNN（K Nearest Neighbor）：存在一个样本数据集合，也称作训练样本集，并且样本集中每个数据都存在标签，即我们知道样本集中每一个数据与所属分类的对应关系。输入没有标签的数据后，将新数据的每个特征与样本集中数据对应的特征进行比较，然后算法提取样本集中特征相似数据（最近邻）的分类标签，一般来说我们只选择样本数据集中前k个最相似的数据，这就是k-近邻算法中的k的出处，通常k是不大于20的整数。最后，选择k个最相似数据中出现次数最多的分类，作为新数据的分类。

看一下下图：x是未知类别的，计算与w1，w2，w3相似度（距离），下图是取5个（k个）最相似的数据，然后从5个中选择出现次数最多的类别，作为x的类别。

![image](http://upload-images.jianshu.io/upload_images/4618424-2c7bcb9b5cae18e4?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

其实，k值的选择至关重要，看下图，不宜太小不宜太大：

![image](http://upload-images.jianshu.io/upload_images/4618424-1d166b3e87d5a7bc?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

2\. kNN中相似度量方法：

上面提到的相似度（还有推荐中的相似度），很多时候都是用距离来衡量，计算距离的方法有： 

*   闵氏距离

    两观测点x和y间的闵氏距离是指两观测点p个变量值绝对差k次方总和的k次方根：

![image](http://upload-images.jianshu.io/upload_images/4618424-ababe087255c5a18?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

*   欧式距离

    两观测点x和y间的欧式距离是指两观测点p个变量值绝对差平方总和的平方根：

![image.png](http://upload-images.jianshu.io/upload_images/4618424-92e79ae405fca97b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

        可以看出，欧式距离是闵氏距离在k=2时的特例。

*   绝对(曼哈顿)距离

    两观测点x和y间的绝对(曼哈顿)距离是指两观测点p个变量值绝对之差的总和：

![image](http://upload-images.jianshu.io/upload_images/4618424-3679c42e3320a7b2?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

       可以看出，绝对(曼哈顿)距离是闵氏距离在k=1时的特例。

*   切比雪夫距离

    两观测点x和y间的切比雪夫距离是指两观测点p个变量值绝对之差的最大值：

![image](http://upload-images.jianshu.io/upload_images/4618424-4b35b52b0a8cc775?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

        可以看出，切比雪夫距离是闵氏距离在k=无穷大时的特例

*   夹角余弦距离

![image](http://upload-images.jianshu.io/upload_images/4618424-72defa7e472899c4?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

       可以看出夹角余弦距离是从两观测的变量整体结构相似性角度测度其距离的。夹角余弦值越大，其结构相似度越高。

当然除了以上的相似度量方法还有很多，马氏距离、交叉熵、KL变换等，都是可以衡量相似度的方法，但是要注意在什么情境用什么方法；

3\. 注意的问题：

实际应用中，p个维度（特征）之间可能存在数量级的差异（这里也体现了数据归一化的重要性），数量级较大的维度对距离大小的影响会大于数量级小的变量。为了消除这种影响，统计学中常见的方法有标准分数法和极差法(有的称为极大-极小值法)。

*   标准分数法：

![image](http://upload-images.jianshu.io/upload_images/4618424-54bda8bb3340669b?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

*   极差(极大-极小值法)法：

![image](http://upload-images.jianshu.io/upload_images/4618424-c6812f9469aaefc8?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

另外，很多时候是这么做的，例如在DL中我们用CNN提取的特征作为kNN的样本；或者更传统一点，可以通过PCA降维后的结果作为kNN的样本；可以减少维度灾难；鄙人缺少此方便实战经验，写起来比较晦涩；

4\. kNN的优缺点

KNN的优缺点：

*   优点：

    1、思想简单，理论成熟，既可以用来做分类也可以用来做回归； 

    2、可用于非线性分类； 

    3、训练（计算时间）时间复杂度为O(n)； 

    4、准确度高，对数据没有假设，对outlier不敏感；

*   缺点： 
    1、计算量大（样本量大）； 
    2、样本不平衡问题（即有些类别的样本数量很多，而其它样本的数量很少）； 
    3、需要大量的内存；

5\. 一些思考：

*   一个是机器学习，算法基本上都比较简单，最难的是数学建模，把那些业务中的特性抽象成向量的过程，另一个是选取适合模型的数据样本。这两个事都不是简单的事。算法反而是比较简单的事。

*   对于KNN算法中找到离自己最近的K个点，是一个很经典的算法面试题，需要使用到的数据结构是“较大堆——Max Heap”，一种二叉树。你可以看看相关的算法。

下一遍： kNN实践 python代码+详细注释

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

[12\. MachineLN之优化算法](http://mp.weixin.qq.com/s?__biz=MzU3MTM3MTIxOQ==&mid=2247484064&idx=1&sn=f834585e2e77299f28148084e6fddcbc&chksm=fce0791ccb97f00a1c168fc05725c59f9e5be5fd22249e3c49e949d5daf812a9ca9da365b50e&scene=21#wechat_redirect)

13. [ReinforcementLN之图像分类](http://mp.weixin.qq.com/s?__biz=MzU3MTM3MTIxOQ==&mid=2247484102&idx=1&sn=a44c0efaa3d939d148cc0d3dd2dc06b9&chksm=fce0797acb97f06c83cf33f711165dcef005dd54478503098f904e41800fe3beb855a579ed78&scene=21#wechat_redirect)

![image](http://upload-images.jianshu.io/upload_images/4618424-d62b84c025c6a9f7?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
版权声明：本文为博主原创文章，未经博主允许不得转载。有问题可以加微信：lp9628(注明CSDN)。
