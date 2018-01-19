
我想说：

又到了每天写东西的时间了，这时候最兴奋，这种兴奋可以延续到后半夜，两点甚至三点；以前写博客都是杂乱无章的，现在写公众号决定按照一个框架，按照一个系列来写；

1\. 前言：  

先看一个概念：

Covariance shift 
——when the input distribution to a learning system changes, it is said to experience covariance shift.

在模型训练的时候我们一般都会做样本归一化（样本归一化作用会在下面文章介绍），在往多层神经网络传播时，前面层参数的改变，使得后面层的输入分布发生改变时，就叫Internal covariance shift。这会导致：其一，增加模型训练时间，因为样本分布变了，要调整 参数适应这种分布；其二：在[MachineLN之激活函数](http://mp.weixin.qq.com/s?__biz=MzU3MTM3MTIxOQ==&mid=2247483968&idx=1&sn=dc2e52c68cd8ea9037b114625c9b1a33&chksm=fce079fccb97f0ea4f3b8f8c74cb779613e06c54a718378c9d174651c16cabb28a1c283b9083&scene=21#wechat_redirect)文章中提到的使用sigmoid函数，梯度消失的问题；

2\. BN （Batch Normalization）

BN：批量规范化：使得均值为0，方差为1；scale and shift：引入两个参数，从而使得BN操作可以代表一个恒等变换，为了训练所需加入到BN有可能还原最初的输入；看一下这个公式：

![image](http://upload-images.jianshu.io/upload_images/4618424-73c092b03d8456d6?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240) 

再看下面BN的两个公式，将上面公式带入，你会发现输入=输出，好尴尬啊！

![image](http://upload-images.jianshu.io/upload_images/4618424-aa95d3b73c35cf1e?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

BN的引入就是为了解决 样本分布改变训练训练慢、梯度消失、过拟合（可以使用较低的dropout和L2系数）等问题； 

BN的具体推导，就不得不提到google的Batch Normalization Accelerating Deep Network Training by Reducing Internal Covariate Shift论文，看下图：

![image](http://upload-images.jianshu.io/upload_images/4618424-720cf3e611a58f92?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

输入：m个样本x{1,...,m}，一般时卷积后输入激活函数前的数据；

输出：BN的处理结果；

上图中前向传播的公式应该很好理解；

下图是后向传播的公式：

![image](http://upload-images.jianshu.io/upload_images/4618424-17bfcbbe31aec2a1?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

直接看起来比较费劲还是用手撕一下吧： 

![image](http://upload-images.jianshu.io/upload_images/4618424-55a31bb5b7e219bd?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

再看一下训练过程：

![image](http://upload-images.jianshu.io/upload_images/4618424-40a6c817ecd527d5?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

可以解释为：（参考大神）

*   1.对于K维（通道数）的输入，假设每一维包含m个变量（这里可以理解为cnn的feature map），所以需要K个循环。每个循环中按照上面所介绍的方法计算γ与β。这里的K维，在卷积网络中可以看作是卷积核个数（卷积后的通道数），如网络中第n层有64个卷积核，就需要计算64次。 

*   *需要注意，在正向传播时，会使用γ与β使得BN层输出与输入一样。*

*   2.在反向传播时利用γ与β求得梯度从而改变训练权值（变量）。 

*   3.通过不断迭代直到训练结束，求得关于不同层的γ与β。如网络有n个BN层，每层根据batch_size决定有多少个变量，设定为m，这里的mini-batcherB指的是***特征图大小*batch_size***，即***m=特征图大小*batch_size***，因此，对于batch_size为1，这里的m就是每层特征图的大小。 

*   4.不断遍历训练集中的图片，取出每个batch_size中的γ与β，最后统计每层BN的γ与β各自的和除以图片数量得到平均直，并对其做无偏估计直作为每一层的E[x]与Var[x]。 

*   5.在预测的正向传播时，对测试数据求取γ与β，并使用该层的E[x]与Var[x]，通过图中11:所表示的公式计算BN层输出。 

*   ***注意，在预测时，BN层的输出已经被改变***，所以BN层在预测的作用体现在此处。

3\. 总结

上面两本部分回答了BN的由来、BN的计算、BN的前后向传播。对自己今后的工作有什么启发？

还可以参考：

1\. tf的BN代码：http://blog.csdn.net/u014365862/article/details/77188011

2. resnet、inception、inception_resnet等网络的BN使用：http://blog.csdn.net/u014365862/article/details/78272811

推荐阅读：

1. [MachineLN之三要素](http://mp.weixin.qq.com/s?__biz=MzU3MTM3MTIxOQ==&mid=2247483841&idx=2&sn=e4a3cff7b12c48af237c577c487ba3a1&chksm=fce07a7dcb97f36be5003c3018b3a391070bdc4e56839cb461d226113db4c5f24032e0bf5809&scene=21#wechat_redirect)

2. [MachineLN之模型评估](http://mp.weixin.qq.com/s?__biz=MzU3MTM3MTIxOQ==&mid=2247483872&idx=2&sn=8436e1eb9055d3a372278ee8688cd703&chksm=fce07a5ccb97f34a4490f60304b206c741d2395149c2c2e68bddb3faf7daf9121ca27a5d6a97&scene=21#wechat_redirect)

3. [MachinLN之dl](http://mp.weixin.qq.com/s?__biz=MzU3MTM3MTIxOQ==&mid=2247483894&idx=2&sn=63333c02674e15e84159e064073fe563&chksm=fce07a4acb97f35cc38f75dc891a19129e2406270d04b739cfa9b8a28f9780b4e2a65a7cd39b&scene=21#wechat_redirect)

4. [DeepLN之CNN解析](http://mp.weixin.qq.com/s?__biz=MzU3MTM3MTIxOQ==&mid=2247483906&idx=1&sn=2eceda7d9703d5315638739e04d5b6e7&chksm=fce079becb97f0a8b8dd2e34a9e757f23757cf2699c397707bfaa677c9f8204c91508840d8f7&scene=21#wechat_redirect)

[5\. DeepLN之手撕CNN权值更新（笔记）](http://mp.weixin.qq.com/s?__biz=MzU3MTM3MTIxOQ==&mid=2247483927&idx=1&sn=132b90eb009022d2d792303e6d824df7&chksm=fce079abcb97f0bd9b893889a7c8232c0254e447277b930ab8e35ea1f9c49e57c0fc66bab239&scene=21#wechat_redirect)

6. [DeepLN之CNN源码](http://mp.weixin.qq.com/s?__biz=MzU3MTM3MTIxOQ==&mid=2247483948&idx=1&sn=d1edce6e99dac0437797404d15714876&chksm=fce07990cb97f086f2f24ec8b40bb64b588ce657e6ae8d352d4d20e856b5f0e126eaffc5ec99&scene=21#wechat_redirect)

7. [MachineLN之激活函数](http://mp.weixin.qq.com/s?__biz=MzU3MTM3MTIxOQ==&mid=2247483968&idx=1&sn=dc2e52c68cd8ea9037b114625c9b1a33&chksm=fce079fccb97f0ea4f3b8f8c74cb779613e06c54a718378c9d174651c16cabb28a1c283b9083&scene=21#wechat_redirect)

![image](http://upload-images.jianshu.io/upload_images/4618424-8da263c6b391e791?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

我想说：

又到了每天写东西的时间了，这时候最兴奋，这种兴奋可以延续到后半夜，两点甚至三点；以前写博客都是杂乱无章的，现在写公众号决定按照一个框架，按照一个系列来写；

1\. 前言：  

先看一个概念：

Covariance shift 
——when the input distribution to a learning system changes, it is said to experience covariance shift.

在模型训练的时候我们一般都会做样本归一化（样本归一化作用会在下面文章介绍），在往多层神经网络传播时，前面层参数的改变，使得后面层的输入分布发生改变时，就叫Internal covariance shift。这会导致：其一，增加模型训练时间，因为样本分布变了，要调整 参数适应这种分布；其二：在[MachineLN之激活函数](http://mp.weixin.qq.com/s?__biz=MzU3MTM3MTIxOQ==&mid=2247483968&idx=1&sn=dc2e52c68cd8ea9037b114625c9b1a33&chksm=fce079fccb97f0ea4f3b8f8c74cb779613e06c54a718378c9d174651c16cabb28a1c283b9083&scene=21#wechat_redirect)文章中提到的使用sigmoid函数，梯度消失的问题；

2\. BN （Batch Normalization）

BN：批量规范化：使得均值为0，方差为1；scale and shift：引入两个参数，从而使得BN操作可以代表一个恒等变换，为了训练所需加入到BN有可能还原最初的输入；看一下这个公式：![image](http://upload-images.jianshu.io/upload_images/4618424-1b1818ddf14fab23?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

再看下面BN的两个公式，将上面公式带入，你会发现输入=输出，好尴尬啊！

![image](http://upload-images.jianshu.io/upload_images/4618424-970f0fbf6eab48ac?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240) 

BN的引入就是为了解决 样本分布改变训练训练慢、梯度消失、过拟合（可以使用较低的dropout和L2系数）等问题； 

BN的具体推导，就不得不提到google的Batch Normalization Accelerating Deep Network Training by Reducing Internal Covariate Shift论文，看下图：

![image](http://upload-images.jianshu.io/upload_images/4618424-9a5278840e851ee5?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240) 

输入：m个样本x{1,...,m}，一般时卷积后输入激活函数前的数据；

输出：BN的处理结果；

上图中前向传播的公式应该很好理解；

下图是后向传播的公式：

![image](http://upload-images.jianshu.io/upload_images/4618424-7d89efdb720f5e36?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

直接看起来比较费劲还是用手撕一下吧： 

![image](http://upload-images.jianshu.io/upload_images/4618424-c0816c6971a8287a?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

再看一下训练过程：

![image](http://upload-images.jianshu.io/upload_images/4618424-534920aa8147edcc?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

可以解释为：（参考大神）

*   1.对于K维（通道数）的输入，假设每一维包含m个变量（这里可以理解为cnn的feature map），所以需要K个循环。每个循环中按照上面所介绍的方法计算γ与β。这里的K维，在卷积网络中可以看作是卷积核个数（卷积后的通道数），如网络中第n层有64个卷积核，就需要计算64次。 

*   *需要注意，在正向传播时，会使用γ与β使得BN层输出与输入一样。*

*   2.在反向传播时利用γ与β求得梯度从而改变训练权值（变量）。 

*   3.通过不断迭代直到训练结束，求得关于不同层的γ与β。如网络有n个BN层，每层根据batch_size决定有多少个变量，设定为m，这里的mini-batcherB指的是***特征图大小*batch_size***，即***m=特征图大小*batch_size***，因此，对于batch_size为1，这里的m就是每层特征图的大小。 

*   4.不断遍历训练集中的图片，取出每个batch_size中的γ与β，最后统计每层BN的γ与β各自的和除以图片数量得到平均直，并对其做无偏估计直作为每一层的E[x]与Var[x]。 

*   5.在预测的正向传播时，对测试数据求取γ与β，并使用该层的E[x]与Var[x]，通过图中11:所表示的公式计算BN层输出。 

*   ***注意，在预测时，BN层的输出已经被改变***，所以BN层在预测的作用体现在此处。

3\. 总结

上面两本部分回答了BN的由来、BN的计算、BN的前后向传播。对自己今后的工作有什么启发？

还可以参考：

1\. tf的BN代码：http://blog.csdn.net/u014365862/article/details/77188011

2. resnet、inception、inception_resnet等网络的BN使用：http://blog.csdn.net/u014365862/article/details/78272811

推荐阅读：

1. [MachineLN之三要素](http://mp.weixin.qq.com/s?__biz=MzU3MTM3MTIxOQ==&mid=2247483841&idx=2&sn=e4a3cff7b12c48af237c577c487ba3a1&chksm=fce07a7dcb97f36be5003c3018b3a391070bdc4e56839cb461d226113db4c5f24032e0bf5809&scene=21#wechat_redirect)

2. [MachineLN之模型评估](http://mp.weixin.qq.com/s?__biz=MzU3MTM3MTIxOQ==&mid=2247483872&idx=2&sn=8436e1eb9055d3a372278ee8688cd703&chksm=fce07a5ccb97f34a4490f60304b206c741d2395149c2c2e68bddb3faf7daf9121ca27a5d6a97&scene=21#wechat_redirect)

3. [MachinLN之dl](http://mp.weixin.qq.com/s?__biz=MzU3MTM3MTIxOQ==&mid=2247483894&idx=2&sn=63333c02674e15e84159e064073fe563&chksm=fce07a4acb97f35cc38f75dc891a19129e2406270d04b739cfa9b8a28f9780b4e2a65a7cd39b&scene=21#wechat_redirect)

4. [DeepLN之CNN解析](http://mp.weixin.qq.com/s?__biz=MzU3MTM3MTIxOQ==&mid=2247483906&idx=1&sn=2eceda7d9703d5315638739e04d5b6e7&chksm=fce079becb97f0a8b8dd2e34a9e757f23757cf2699c397707bfaa677c9f8204c91508840d8f7&scene=21#wechat_redirect)

[5\. DeepLN之手撕CNN权值更新（笔记）](http://mp.weixin.qq.com/s?__biz=MzU3MTM3MTIxOQ==&mid=2247483927&idx=1&sn=132b90eb009022d2d792303e6d824df7&chksm=fce079abcb97f0bd9b893889a7c8232c0254e447277b930ab8e35ea1f9c49e57c0fc66bab239&scene=21#wechat_redirect)

6. [DeepLN之CNN源码](http://mp.weixin.qq.com/s?__biz=MzU3MTM3MTIxOQ==&mid=2247483948&idx=1&sn=d1edce6e99dac0437797404d15714876&chksm=fce07990cb97f086f2f24ec8b40bb64b588ce657e6ae8d352d4d20e856b5f0e126eaffc5ec99&scene=21#wechat_redirect)

7. [MachineLN之激活函数](http://mp.weixin.qq.com/s?__biz=MzU3MTM3MTIxOQ==&mid=2247483968&idx=1&sn=dc2e52c68cd8ea9037b114625c9b1a33&chksm=fce079fccb97f0ea4f3b8f8c74cb779613e06c54a718378c9d174651c16cabb28a1c283b9083&scene=21#wechat_redirect)

![image](http://upload-images.jianshu.io/upload_images/4618424-8a071ff2f655c3da?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![image](http://upload-images.jianshu.io/upload_images/4618424-f2519a71d7f77812?imageMogr2/auto-orient/strip)

![image](http://upload-images.jianshu.io/upload_images/4618424-2ee50cc3d5d46299?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

MachineLN 交流群请扫码加machinelp为好友：

![image](http://upload-images.jianshu.io/upload_images/4618424-a7683c6d8637e8a6?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)



版权声明：本文为博主原创文章，未经博主允许不得转载。有问题可以加微信：lp9628(注明CSDN)。
