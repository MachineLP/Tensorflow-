
我想说：

其实很多时候大家都想自己做一些事情，但是很多也都是想想而已，其实有了想法自己感觉可行，就可以去行动起来，去尝试，即使最后败了，也无怨无悔，有句话说的很好：成功收获成果，失败收获智慧，投入收获快乐！反而有时候顾及的太多，本应该做的事情错过了，怪谁呢？我跟大家不同的是无论什么事情，先做了再说吧！ 

说起过拟合，那么我的问题是：

（1）什么是过拟合？

（2）为什么要解决过拟合问题？

（3）解决过拟合有哪些方法？

1\. 什么是过拟合？

不同的人提到过拟合时会有不同的含义：

（1） 看最终的loss，训练集的loss比验证集的loss小的多；

（2）训练的loss还在降，而验证集的loss已经开始升了；

（3）另外要提一下本人更注重loss，你过你看的是准确率，那么也OK，适合自己的才是最好的，正所谓学习再多tricks，不如踩一遍坑；

*   在第一种（1）中验证集的loss还在降，是不用太在意的。（2）中的overfitting如下图，在任何情况下都是不行滴！

*   和过拟合相对应的是欠拟合，如下图刚开始的时候；可以参考[MachineLN之模型评估](http://mp.weixin.qq.com/s?__biz=MzU3MTM3MTIxOQ==&mid=2247483872&idx=2&sn=8436e1eb9055d3a372278ee8688cd703&chksm=fce07a5ccb97f34a4490f60304b206c741d2395149c2c2e68bddb3faf7daf9121ca27a5d6a97&scene=21#wechat_redirect)。

![image.png](http://upload-images.jianshu.io/upload_images/4618424-0f691f3d2138be17.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


2\. 为什么要解决过拟合问题？

过拟合对我们最终模型影响是很大的，有时候训练时间越长过拟合越严重，导致模型表现的效果很差，甚至崩溃；由上图我们也能看出来解决过拟合问题的必要性。

3\. 解决过拟合有哪些方法？

（1）正则化

正则化的思想十分简单明了。由于模型过拟合极有可能是因为我们的模型过于复杂。因此，我们需要让我们的模型在训练的时候，在对损失函数进行最小化的同时，也需要让对参数添加限制，这个限制也就是正则化惩罚项。 

假设我们的损失函数是平方损失函数：
![image.png](http://upload-images.jianshu.io/upload_images/4618424-7f92a435f0f19079.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


加入正则化后损失函数将变为：

![image.png](http://upload-images.jianshu.io/upload_images/4618424-10f7700ec20047c7.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


*   L1范数：

![image.png](http://upload-images.jianshu.io/upload_images/4618424-6332fc6f84800905.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


*   L2范数：
![image.png](http://upload-images.jianshu.io/upload_images/4618424-28d5f0dfcd9b0785.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


加入L1范数与L2范数其实就是不要使损失函数一味着去减小，你还得考虑到模型的复杂性，通过限制参数的大小，来限制其产生较为简单的模型，这样就可以降低产生过拟合的风险。 

那么L1和L2的区别在哪里呢？L1更容易得到稀疏解：直接看下面的图吧：（假设我们模型只有 w1,w2 两个参数，下图中左图中黑色的正方形是L1正则项的等值线，而彩色的圆圈是模型损失的等值线；右图中黑色圆圈是L2正则项的等值线，彩色圆圈是同样模型损失的等值线。因为我们引入正则项之后，我们要在模型损失和正则化损失之间折中，因此我们取的点是正则项损失的等值线和模型损失的等值线相交处。通过上图我们可以观察到，使用L1正则项时，两者相交点常在坐标轴上，也就是 w1,w2 中常会出现0；而L2正则项与等值线常相交于象限内，也即为 w1,w2 非0。因此L1正则项时更容易得到稀疏解的。 ）

![image](http://upload-images.jianshu.io/upload_images/4618424-e5ef39ec290e9e58?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

而使用L1正则项的另一个好处是：由于L1正则项求解参数时更容易得到稀疏解，也就意味着求出的参数中含有0较多。因此它自动帮你选择了模型所需要的特征。L1正则化的学习方式是一种嵌入式特征学习方式，它选取特征和模型训练时一起进行的。

（2）Dropout

先看下图：Dropout就是使部分神经元失活，这样就阻断了部分神经元之间的协同作用，从而强制要求一个神经元和随机挑选出的神经元共同进行工作，减轻了部分神经元之间的联合适应性。也可以这么理解：Dropout将一个复杂的网络拆分成简单的组合在一起，这样仿佛是bagging的采样过程,因此可以看做是bagging的廉价的实现（但是它们训练不太一样,因为bagging,所有的模型都是独立的,而dropout下所有模型的参数是共享的。）；

![image.png](http://upload-images.jianshu.io/upload_images/4618424-8399cc97ed6b3ffa.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


Dropout的具体流程如下：

*   对l层第 j 个神经元产生一个随机数 ：

![image.png](http://upload-images.jianshu.io/upload_images/4618424-c75d8a4e71c29777.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


*   将 l 层第 j 个神经元的输入乘上产生的随机数作为这个神经元新的输入：
![image.png](http://upload-images.jianshu.io/upload_images/4618424-9726ff728ed98ec2.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


*   此时第 l 层第 j 个神经元的输出为： 

![image.png](http://upload-images.jianshu.io/upload_images/4618424-079a84cfe62c4d7a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


*   ***注意***：当我们采用了Dropout训练结束之后，应当将网络的权重乘上概率p得到测试网络的权重。

（3）提前终止

由第一副图可以看出，模型在验证集上的误差在一开始是随着训练集的误差的下降而下降的。当超过一定训练步数后，模型在训练集上的误差虽然还在下降，但是在验证集上的误差却不在下降了。此时我们的模型就过拟合了。因此我们可以观察我们训练模型在验证集上的误差，一旦当验证集的误差不再下降时，我们就可以提前终止我们训练的模型。

（4）bagging 和其他集成方法

其实bagging的方法是可以起到正则化的作用,因为正则化就是要减少泛化误差,而bagging的方法可以组合多个模型起到减少泛化误差的作用；在深度学习中同样可以使用此方法,但是其会增加计算和存储的成本。

（5）深度学习中的BN

深度学习中BN也是解决过拟合的有效方法：可以参考之前的文章：[DeepLN之BN](http://mp.weixin.qq.com/s?__biz=MzU3MTM3MTIxOQ==&mid=2247483979&idx=1&sn=3b5cfedc2d475f69e52656e50ff44f36&chksm=fce079f7cb97f0e1217a6b7222cef60930f79f3870692315935dab7abefff32af6b4201c07ab&scene=21#wechat_redirect)。

（6）增加样本量 （[MachineLN之样本不均衡](http://mp.weixin.qq.com/s?__biz=MzU3MTM3MTIxOQ==&mid=2247484011&idx=1&sn=c7ee568af41c28793162e3fa2084c5b6&chksm=fce079d7cb97f0c175e3561b4c999101268984574fea5adc04b1e2ada99bee4277817f969484&scene=21#wechat_redirect)中介绍的数据增强的方法）

在实际的项目中，你会发现，上面讲述的那些技巧虽然都可以减轻过拟合，但是却都比不上增加样本量来的更实在。为什么增加样本可以减轻过拟合的风险呢？这个要从过拟合是啥来说。过拟合可以理解为我们的模型对样本量学习的太好了，把一些样本的特殊的特征当做是所有样本都具有的特征。举个简单的例子，当我们模型去训练如何判断一个东西是不是叶子时，我们样本中叶子如果都是锯齿状的话，如果模型产生过拟合了，会认为叶子都是锯齿状的，而不是锯齿状的就不是叶子了。如果此时我们把不是锯齿状的叶子数据增加进来，此时我们的模型就不会再过拟合了。（这个在最近的项目里常用）因此其实上述的那些技巧虽然有用，但是在实际项目中，你会发现，其实大多数情况都比不上增加样本数据来的实在。

展望：深度学习系列再更新几篇后要暂时告一段落，开始总结传统机器学习的内容：KNN；决策树；贝叶斯；Logistic回归；SVM；AdaBoost；K-means等的理论和实践；中间穿插数据结构和算法（排序；散列表；搜索树；动态规划；贪心；图；字符串匹配等）；再之后我们重回Deep learning；

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

![image.png](http://upload-images.jianshu.io/upload_images/4618424-a5a6b1765814628e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


版权声明：本文为博主原创文章，未经博主允许不得转载。有问题可以加微信：lp9628(注明CSDN)。
