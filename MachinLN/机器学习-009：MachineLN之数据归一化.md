我想说：

有时候很羡慕一些人，从开始的无到有，譬如一些人平常工资三四千，但是由于很长时间的积累和习惯，他们的睡后收入是上班工资的四倍甚至五倍，但是我感觉他们可以，我也一定可以，所以这半年我就拿出更多的时间睡觉，但是我这半年的睡后收入可能只在五千左右；难道我做错了嘛？那么我就从每天的睡觉十个小时缩减到六个小时试试吧，测试一下睡眠时间是不是和睡后收入成反比的（测试结果明年公布），真是奇怪哈，不应该睡的越久睡后收入越多嘛！！！（哈哈哈，真实幽默哈，但是不一定有人欣赏你哈）

另外今天又背了一首诗《黄鹤楼》：（后面教女儿用的）

昔人已乘黄鹤去，此地空余黄鹤楼。

黄鹤一去不复返，白云千载空悠悠。

晴川历历汉阳树，芳草萋萋鹦鹉洲。

日暮乡关何处是？烟波江上使人愁。

**说到数据归一化，那么我的问题是：**

1\. 什么是数据归一化？

2\. 为什么要进行数据归一化？

3\. 数据归一化的方法有哪些？

看到这里你的答案是什么？下面是我的答案：（但是在使用的时候你要知道使用场景和哪些需要归一化，例如SVM、线性回归等需要归一化，决策树就不需要归一化；[**DeepLN之CNN权重更新（笔记）**](http://mp.weixin.qq.com/s?__biz=MzU3MTM3MTIxOQ==&mid=2247483927&idx=1&sn=132b90eb009022d2d792303e6d824df7&chksm=fce079abcb97f0bd9b893889a7c8232c0254e447277b930ab8e35ea1f9c49e57c0fc66bab239&scene=21#wechat_redirect)**数据挖掘中的问题很犀利，很实用；）**

1\. 什么是数据归一化？

归一化可以定义为：归一化就是要把你需要处理的数据经过处理后（通过某种算法）限制在你需要的一定范围内。首先归一化是为了后面数据处理的方便，其次是保证模型运行时收敛加快。 

下面以二维数据举个例子，下图是未归一化的数据：

![image](http://upload-images.jianshu.io/upload_images/4618424-7e16d322a3e64bd7?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

下面两个图是通过不同算法进行归一化：

![image.png](http://upload-images.jianshu.io/upload_images/4618424-e2c0222db05c5b3a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


是的什么事样本归一化解释完了。

2\. 为什么要进行数据归一化？

在我们模型训练的时候，数据在不同的取值范围内,例如x1取值[1:1000],特征x2的取值范[1:10],那么权重更新的时候，w1和w2值的范围或者比率会完全不同,下图中w和b对应w1和w2，可以看出函数的轮廓十分狭窄。

![image.png](http://upload-images.jianshu.io/upload_images/4618424-aaf4301add25f3c6.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


那么再看一下，在这样的数据的损失函数使用梯度下降,必须使用一个非常小的学习比率,因为如果是在这个位置,梯度下降法可能需要更多次迭代过程。

![image.png](http://upload-images.jianshu.io/upload_images/4618424-f28c22c99d3e79bb.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


数据进行归一化后的图是这个样子的：

![image](http://upload-images.jianshu.io/upload_images/4618424-81d12275fd06820b?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

梯度下降的图事这样子的： 

![image](http://upload-images.jianshu.io/upload_images/4618424-663c8dd427176f27?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

到这里，大家知道了为什么进行数据归一化训练的重要性：可以使用较大的学习率，并且加速收敛。

3\. 数据归一化的方法有哪些？ 适应的场景是什么？

（1） 线性归一化

![image](http://upload-images.jianshu.io/upload_images/4618424-d21a8de6dfef4059?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

在图像中可以简化为：x = ( (x / 255) -0.5 ) * 2, 归一化到[-1,1]; 

这种归一化方法比较适用在数值比较集中的情况。但是，如果max和min不稳定，很容易使得归一化结果不稳定，使得后续使用效果也不稳定，实际使用中可以用经验常量值来替代max和min。而且当有新数据加入时，可能导致max和min的变化，需要重新定义。

在不涉及距离度量、协方差计算、数据不符合正太分布的时候，可以使用第一种方法或其他归一化方法。比如图像处理中，将RGB图像转换为灰度图像后将其值限定在[0 255]的范围。

（2）标准差归一化

处理后的数据符合标准正态分布，即均值为0，标准差为1，其转化函数为：

![image](http://upload-images.jianshu.io/upload_images/4618424-e89996ad57c0b2b5?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

u: 所有数据的均值， σ: 所有数据的标准差。

该种归一化方式要求原始数据的分布可以近似为高斯分布，否则归一化的效果会变得很糟糕。在分类、聚类算法中，需要使用距离来度量相似性的时候、或者使用PCA技术进行降维的时候，这种方法表现更好。

（3）**非线性归一化**

 经常用在数据分化比较大的场景，有些数值很大，有些很小。通过一些数学函数，将原始值进行映射。该方法包括 log、指数，正切等。需要根据数据分布的情况，决定非线性函数的曲线，比如log(V, 2)还是log(V, 10)等。

例如：

通过以10为底的log函数转换的方法同样可以实现归一下：

![image](http://upload-images.jianshu.io/upload_images/4618424-5d08278cd5462977?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

用反正切函数也可以实现数据的归一化：

![image](http://upload-images.jianshu.io/upload_images/4618424-f7c86bc75404cbd9?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

我们正在打造一个系列，一起系统化学习知识，欢迎关注我们，文章可能会有错误，欢迎交流指正。并且发现一处错误可以联系machinelp，10红包，红包虽小，意义重大！让我们一起学习一起进步，这是一个共赢的时代，任何事情从我这里开始了都不会从我这里停下来，一起加油！

推荐阅读：

1. [MachineLN之三要素](http://mp.weixin.qq.com/s?__biz=MzU3MTM3MTIxOQ==&mid=2247483841&idx=2&sn=e4a3cff7b12c48af237c577c487ba3a1&chksm=fce07a7dcb97f36be5003c3018b3a391070bdc4e56839cb461d226113db4c5f24032e0bf5809&scene=21#wechat_redirect)

2. [MachineLN之模型评估](http://mp.weixin.qq.com/s?__biz=MzU3MTM3MTIxOQ==&mid=2247483872&idx=2&sn=8436e1eb9055d3a372278ee8688cd703&chksm=fce07a5ccb97f34a4490f60304b206c741d2395149c2c2e68bddb3faf7daf9121ca27a5d6a97&scene=21#wechat_redirect)

3. [MachinLN之dl](http://mp.weixin.qq.com/s?__biz=MzU3MTM3MTIxOQ==&mid=2247483894&idx=2&sn=63333c02674e15e84159e064073fe563&chksm=fce07a4acb97f35cc38f75dc891a19129e2406270d04b739cfa9b8a28f9780b4e2a65a7cd39b&scene=21#wechat_redirect)

4. [DeepLN之CNN解析](http://mp.weixin.qq.com/s?__biz=MzU3MTM3MTIxOQ==&mid=2247483906&idx=1&sn=2eceda7d9703d5315638739e04d5b6e7&chksm=fce079becb97f0a8b8dd2e34a9e757f23757cf2699c397707bfaa677c9f8204c91508840d8f7&scene=21#wechat_redirect)

[5\. DeepLN之手撕CNN权值更新（笔记）](http://mp.weixin.qq.com/s?__biz=MzU3MTM3MTIxOQ==&mid=2247483927&idx=1&sn=132b90eb009022d2d792303e6d824df7&chksm=fce079abcb97f0bd9b893889a7c8232c0254e447277b930ab8e35ea1f9c49e57c0fc66bab239&scene=21#wechat_redirect)

6. [DeepLN之CNN源码](http://mp.weixin.qq.com/s?__biz=MzU3MTM3MTIxOQ==&mid=2247483948&idx=1&sn=d1edce6e99dac0437797404d15714876&chksm=fce07990cb97f086f2f24ec8b40bb64b588ce657e6ae8d352d4d20e856b5f0e126eaffc5ec99&scene=21#wechat_redirect)

7. [MachineLN之激活函数](http://mp.weixin.qq.com/s?__biz=MzU3MTM3MTIxOQ==&mid=2247483968&idx=1&sn=dc2e52c68cd8ea9037b114625c9b1a33&chksm=fce079fccb97f0ea4f3b8f8c74cb779613e06c54a718378c9d174651c16cabb28a1c283b9083&scene=21#wechat_redirect) 

8. [MachineLN之BN](http://mp.weixin.qq.com/s?__biz=MzU3MTM3MTIxOQ==&mid=2247483979&idx=1&sn=3b5cfedc2d475f69e52656e50ff44f36&chksm=fce079f7cb97f0e1217a6b7222cef60930f79f3870692315935dab7abefff32af6b4201c07ab&scene=21#wechat_redirect)

![image](http://upload-images.jianshu.io/upload_images/4618424-230cee2954f472dc?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
