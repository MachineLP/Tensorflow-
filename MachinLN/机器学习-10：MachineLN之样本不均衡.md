
我想说：

其实很多时候，有竞争是好的事情，可以促进你的成长，可以磨练你的耐性，可以提升你的魅力，可以表现你的豁达，可以体验成功的喜悦，可以感受失败其实并不可怕，可怕的是你没有面对失败的勇气；而今天的社会达尔文的进化论其实从来没有变过，唯一不变的事情想必就是变了，做慈善的是慈善机构，做教育的是学校，百依百顺的是父母，只要踏上社会，那么对不起，优胜劣汰，适者生存，你必须面对，并且你面对的都是高手，是多个依依东望的诸葛亮，你要脱颖而出除了变的更优秀没有出路！ 那么你打算怎么做呢？

说到样本不均衡，感觉平时大家不太重视，下面来一起讨论一下！

那么我的问题是：

1\. 什么是样本不均衡？

2\. 为什么要解决样本不均衡？

3\. 解决样本不均衡有哪些方法？

看到这里你的答案是什么？下面是我的答案：

1\. 什么是样本不均衡？

样本不均衡：在准备训练样本的时候，各类别样本比例不等，有的差距可能比较小，有的差距则会比较大，以CIFAR-10为例：

CIFAR-10是一个简单的图像分类数据集。共有10类（airplane，automobile，bird，cat，deer，dog， frog，horse，ship，truck），每一类含有5000张训练图片，1000张测试图片。如下图：Dist. 1：类别平衡，每一类都占用10%的数据。Dist. 2、Dist. 3：一部分类别的数据比另一部分多。Dist. 4、Dist 5：只有一类数据比较多。Dist. 6、Dist 7：只有一类数据比较少。Dist. 8： 数据个数呈线性分布。Dist. 9：数据个数呈指数级分布。Dist. 10、Dist. 11：交通工具对应的类别中的样本数都比动物的多。

![image](http://upload-images.jianshu.io/upload_images/4618424-154cb678fcd2c728?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

2\. 为什么要解决样本不均衡？

训练网络使用的是CIFAR-10的结构，下面是测试结果：可以看出总的准确率表现不错的几组1,2,6,7,10,11都是大部分类别平衡，一两类差别较大；而表现很差的，像5,9可以说是训练失败了，他们的不平衡性也比前面的要强。

![image.png](http://upload-images.jianshu.io/upload_images/4618424-6e0d809660c7552f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

那么再看一下，对样本少的数据进行过采样之后，测试结果：可以看到经过过采样将类别数量平衡以后，总的表现基本相当。（过采样虽然是一个很简单的想法，但是很OK，3中还将介绍海康威视ImageNet2016竞赛经验）

![image](http://upload-images.jianshu.io/upload_images/4618424-5cd309e83b1c9659?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

想必到这里可以看到样本均衡的重要性了吧。

3\. 解决样本不均衡有哪些方法？

解决不均衡问题的方式有很多：

（1）可以将数据进行扩增： （这些方法有时候也可以勉强做为数据不均衡的增强方法，如果训练时候各类样本都已经用了以下的方法进行data augmentation，那么样本不均衡就选其他方法来做吧）

*   原图：

![image](http://upload-images.jianshu.io/upload_images/4618424-3383c00860f21bb6?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

*   图像旋转；

![image](http://upload-images.jianshu.io/upload_images/4618424-adedd9f1582c3e34?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

*   图像crop；

![image.png](http://upload-images.jianshu.io/upload_images/4618424-f0f7eb0e87830d1a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


*   图像平移；

![image](http://upload-images.jianshu.io/upload_images/4618424-ada7b6b120b27f42?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

*   图像flip；（左右镜像，有的可以上下）

![image.png](http://upload-images.jianshu.io/upload_images/4618424-75f97cb11d6261b1.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

*   （5）图像光照；

![image](http://upload-images.jianshu.io/upload_images/4618424-50d741490d77d9c0?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

*   还有一些像添加噪声； 透视变换等；

（2） 可以借鉴一下海康威视的经验：

![image.png](http://upload-images.jianshu.io/upload_images/4618424-39ca3ba733254393.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

以图中的例子来说，步骤如下：首先对原始的图像列表，按照标签顺序进行排序；然后计算每个类别的样本数量，并得到样本最多的那个类别的样本数**。根据这个最多的样本数，对每类随机都产生一个随机排列的列表；然后用每个类别的列表中的数对各自类别的样本数求余，得到一个索引值，从该类的图像中提取图像，生成该类的图像随机列表；**然后把所有类别的随机列表连在一起，做个Random Shuffling，得到最后的图像列表，用这个列表进行训练。每个列表，到达最后一张图像的时候，然后再重新做一遍这些步骤，得到一个新的列表，接着训练。Label Shuffling方法的优点在于，只需要原始图像列表，所有操作都是在内存中在线完成，非常易于实现。

另外也可以按照同样的方式对多的样本进行欠采样；

（3）还可以用Weighted samples，给每一个样本加权重，样本多的类别每个的权重就小些，样本少的类别每个的权重就大些，这样无论样本是否均衡，在Loss Function中每类的影响力都一样的。

（4）还可以：再过采样之后使用K-fold交叉验证，来弥补一些特殊样本造成的过拟合问题，（K-fold交叉验证就是把原始数据随机分成K个部分，在这K个部分中选择一个作为测试数据，剩余的K-1个作为训练数据。交叉验证的过程实际上是将实验重复做K次，每次实验都从K个部分中选择一个不同的部分作为测试数据，剩余的数据作为训练数据进行实验，最后可以把得到的K个实验结果平均。）

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

![image.png](http://upload-images.jianshu.io/upload_images/4618424-106ee818075ec935.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


版权声明：本文为博主原创文章，未经博主允许不得转载。有问题可以加微信：lp9628(注明CSDN)。
