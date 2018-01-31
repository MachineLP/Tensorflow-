
我想说： 

其实....整天其实，感觉没啥好说的了，好好GAN吧。

逻辑回归可以这样理解： 和感知机比较着来看或许更好，将感知机的表达式中的sign函数换成sigmoid就是逻辑回归的表达式，但是这一换不要紧，导致后边参数更新的方式完全不同，因为逻辑回顾的损失函数是参数连续可导的，还记得我们说过感知机的损失函数是参数连续不可导的吗？ ：[MachineLN之感知机](http://mp.weixin.qq.com/s?__biz=MzU3MTM3MTIxOQ==&mid=2247484156&idx=1&sn=5e3e5baab2701bda0c17ef0edb60bac0&chksm=fce07940cb97f056eae2cd8e52171092e893886d731f6bfaaec685ec67400ec82dd1288a04fa&scene=21#wechat_redirect)

还是一如既往：

说起逻辑回归，那么我的问题：（根据[MachineLN之三要素](http://mp.weixin.qq.com/s?__biz=MzU3MTM3MTIxOQ==&mid=2247483841&idx=2&sn=e4a3cff7b12c48af237c577c487ba3a1&chksm=fce07a7dcb97f36be5003c3018b3a391070bdc4e56839cb461d226113db4c5f24032e0bf5809&scene=21#wechat_redirect)：模型、策略、算法）

（1）什么是逻辑回归？（模型）

（2）逻辑回归是如何学习的？（策略）

（3）逻辑回归学习算法？（算法）

看到这里你的答案是什么？下面是我的答案：

1\. 什么是逻辑回归？

前面有提到，逻辑回归是来解决分类问题的，而不是回归问题，想想问什么呢？就是因为sigmoid。说起sigmoid应该很熟悉吧：[MachineLN之激活函数](http://mp.weixin.qq.com/s?__biz=MzU3MTM3MTIxOQ==&mid=2247483968&idx=1&sn=dc2e52c68cd8ea9037b114625c9b1a33&chksm=fce079fccb97f0ea4f3b8f8c74cb779613e06c54a718378c9d174651c16cabb28a1c283b9083&scene=21#wechat_redirect)

直接看一下二项逻辑回归表达式吧：可以理解为事件发生的概率和事件不发生的概率：

![image](http://upload-images.jianshu.io/upload_images/4618424-dc1f60f7bb7fa6ac?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

2. 逻辑回归是如何学习的？

逻辑回归可以通过极大似然估计（什么是极大似然估计，简单说利用已知的样本结果，反推有可能（最大概率）导致这样结果的参数值（模型已知，参数未知））来估计模型参数：

设：

![image](http://upload-images.jianshu.io/upload_images/4618424-d8e870e9e7ce4a6d?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

那么似然函数可表示为： 

![image](http://upload-images.jianshu.io/upload_images/4618424-d98019b34167e75c?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

为了后面求导，这里取对数似然函数： 

那么对数似然函数，也就是损失函数为：

![image](http://upload-images.jianshu.io/upload_images/4618424-8791408dcc55ac7b?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

接下来我们要求其最大值，其最大值的要用梯度上升法，如果想用梯度下降只需要加一个负号，利用梯度下降法求最小值。 

（3）逻辑回归学习算法？

还是用手撕一下吧：  

![image](http://upload-images.jianshu.io/upload_images/4618424-54967a3d5f4f441d?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

下一节的源代码解析你会看到就是按照这个更新公式来做的，但是加了一些优化。

另外可以考虑一下：逻辑回归和softmax回归的关系？ 多项逻辑回归是softmax吗？ 答案是肯定的！

看下面逻辑回归表达式：

![image](http://upload-images.jianshu.io/upload_images/4618424-58200d8f6a2e2f3c?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

在看softmax表达式： 

![image](http://upload-images.jianshu.io/upload_images/4618424-8c501f3b1d7a5cb9?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

有人说，啊，完全不一样，不要急化简一下看看：

![image](http://upload-images.jianshu.io/upload_images/4618424-973bd2a0b3da9856?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

推荐阅读：

1. [机器学习-1：MachineLN之三要素](http://blog.csdn.net/u014365862/article/details/78955063)

2. [机器学习-2：MachineLN之模型评估](http://blog.csdn.net/u014365862/article/details/78959353)

3. [机器学习-3：MachineLN之dl](http://blog.csdn.net/u014365862/article/details/78980142)

4. [机器学习-4：DeepLN之CNN解析](http://blog.csdn.net/u014365862/article/details/78986089)

5. [机器学习-5：DeepLN之CNN权重更新（笔记）](http://blog.csdn.net/u014365862/article/details/78959211)

6. [机器学习-6：DeepLN之CNN源码](http://blog.csdn.net/u014365862/article/details/79010248)

7. [机器学习-7：MachineLN之激活函数](http://blog.csdn.net/u014365862/article/details/79007801)

8. [机器学习-8：DeepLN之BN](http://blog.csdn.net/u014365862/article/details/79019518)

9. [机器学习-9：MachineLN之数据归一化](http://blog.csdn.net/u014365862/article/details/79031089)

10. [机器学习-10：MachineLN之样本不均衡](http://blog.csdn.net/u014365862/article/details/79040390)

11. [机器学习-11：MachineLN之过拟合](http://blog.csdn.net/u014365862/article/details/79057073) 

12. [机器学习-12：MachineLN之优化算法](http://blog.csdn.net/u014365862/article/details/79070721)

13. [机器学习-13：MachineLN之kNN](http://blog.csdn.net/u014365862/article/details/79091913)

14. [机器学习-14：MachineLN之kNN源码](http://blog.csdn.net/u014365862/article/details/79101209)

15. [](http://mp.blog.csdn.net/postedit/79135612)[机器学习-15：MachineLN之感知机](http://blog.csdn.net/u014365862/article/details/79135612)

16. [机器学习-16：MachineLN之感知机源码](http://blog.csdn.net/u014365862/article/details/79135767)

17. [机器学习-17：MachineLN之逻辑回归](http://blog.csdn.net/u014365862/article/details/79157777)

18. [机器学习-18：MachineLN之逻辑回归源码](http://blog.csdn.net/u014365862/article/details/79157841)

![image](http://upload-images.jianshu.io/upload_images/4618424-1919cd11bc3dbb8f?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


版权声明：本文为博主原创文章，未经博主允许不得转载。有问题可以加微信：lp9628(注明CSDN)。
