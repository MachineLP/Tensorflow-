
我想说：

其实想一下从上学到毕业，学了那么多有多少是真实用到的呢？但是这些事潜移默化影响你的东西，其实我们学习的并不是真实的会这些知识（并且有很多知识现在过时），而是我们学习的是一种快速学习一门知识的能力，要的就是这个快字；怎么一个快字了得，对不光快还要稳；直到今天才真正了解一些教育的含义，并不是死记硬背，而是举一反三，并不是拿来主义，而是针对特定问题特定场景特定解决；并不是随波逐流，而是扬起自己的帆远航；并不是svm，而是一种境界；

其实很多时候我们学东西是学习的一种思维方式；怎么说呢？就好比客户谈需求，他们总是下意识的将人的思维加进去，想当然的就说这件事情很简单啊，怎样怎样....那么这就是缺乏AI的思维方式，要和人的思维区分开来，无论是做AI还是像用AI的，这种AI思维方式还是应该有的；否则就是很难交流；整天就是解释这些问题；

接着上一节，svm原理：[machineLN之SVM（1）](http://mp.weixin.qq.com/s?__biz=MzU3MTM3MTIxOQ==&mid=2247484226&idx=1&sn=b5b58841507e61deed49058e1c75cf8a&chksm=fce078fecb97f1e819277f1a5afac8ffe07a69324afa0e0597aa4f3a9203c2667d6c99adf8b4&scene=21#wechat_redirect)

这要处理三个问题：

（1）SVM软间隔最大化；

（2）SVM核技巧；

（3）SVM求解对偶问题的SMO算法；

（4）SVM使用梯度下降损失函数应该怎么设计；

上一节已经将SVM的流程大致走了一遍，但是真实应用的数据往往不是严格线性可分的或者说他就根本不是线性可分的；针对这两个问题有不同的解决方法：

1. SVM软间隔最大化

记得上节提到的硬间隔吗？相对的就是软间隔，软间隔就是针对某些样本点不能满足函数间隔大于等于1的约束条件，或者说不是严格意思上线性可分的；看一下软间隔的支持向量（有了硬间隔的支持向量，软间隔的支持向量也不难理解）。

![image](http://upload-images.jianshu.io/upload_images/4618424-79e07194324b84f5?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

解决这个问题：可以对每个样本点引进一个松弛变量，约束条件变为：

![image](http://upload-images.jianshu.io/upload_images/4618424-00aaf64b3a25bb9e?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

那么针对这种线性不可分的样本，我们的策略是：

![image](http://upload-images.jianshu.io/upload_images/4618424-e73c7156f91d49b5?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

参考上一篇，我们同样引入**学习的对偶算法(dual algorithn) ：拉格朗日函数；转为对偶问题来解决：**

**最后整理可得完整算法流程：**

![image](http://upload-images.jianshu.io/upload_images/4618424-4e84324afabef815?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

现在我们再回到软间隔的支持向量的图上：

若a*<C，则约束![image](http://upload-images.jianshu.io/upload_images/4618424-b58c3a0124b13d8e?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

，支持向量xi恰好落在间隔边界上；

若a*<C，0<约束<1，则分类正确，xi在间隔边界与分离超平面之间；

若a*<C，约束=1，则xi在分离超平面上；

若a*<C，约束>1，则xi位于分离超平面误分一侧；

对于线性不可分引入软间隔是一种方式，下面介绍另外一种方法：核技巧；

2\. 核函数（解决线性不可分的问题还可以通过引入核函数来解决，这里可以联想到神经网络是怎么解决线性不可分问题的呢？它又是否可以引入核函数？那么激活函数是不是可以理解为核函数呢？真实越来越有意思了）

核函数的原理：将一个非线性可分的数据 映射到另一个空间 变为线性可以分；原思维方式解决不了，那我们就换一种思维方式解决；看一下图示：

![image](http://upload-images.jianshu.io/upload_images/4618424-d25332c34e15885e?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

或者看一下动态图：（是不是你看到了从二维的线性不可分到三维的线性可分） 

![image](http://upload-images.jianshu.io/upload_images/4618424-547e641e186db459?imageMogr2/auto-orient/strip)

核函数： 设X是输入空间，H为特征空间，如果存在一个映射映射函数：

![image](http://upload-images.jianshu.io/upload_images/4618424-b88ae6cd6179c0c7?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

使得对所有属于X的x,z，函数K(x,z)满足条件：

![image](http://upload-images.jianshu.io/upload_images/4618424-40ffc3713ab22dc9?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

看到这里应该考虑下面一个问题： 

![image](http://upload-images.jianshu.io/upload_images/4618424-de1d8989becc6d8e?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

只有判定核函数是不是正定核函数才满足以上条件，正定？可以回顾一下高数了；

设X包含于R<sup>n</sup>，K(x,z)为定义在X*X上的对称函数，如果对任意x<sub>i</sub>属于X，i=1,2,... ,m,  K(x,z)对应的Gram矩阵：![image](http://upload-images.jianshu.io/upload_images/4618424-b6d787b10cc84444?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

是半正定矩阵，则称K(x,z)是正定核。

加入核函数的svm优化问题可定义为：

![image](http://upload-images.jianshu.io/upload_images/4618424-d325680fd8dcb712?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

下面又开始有意思了：看一下求解alpha核b的SMO算法：

3. SVM求解对偶问题的SMO算法（手撕SMO）

SMO算法是一种启发式算法，其基本思路是：如果所有变量的解都满足此最优化问题的KKT条件(Karush-Kuhn-Tucker conditions)，那么这个最优化问题的解就得到了。因为KKT条件是该最优化问题的充分必要条件。否则，选择两个变量，固定其他变量，针对这两个变量构建一个二次规划问题。这个二次规划问题关于这两个变量的解应该更接近原始二次规划问题的解，因为这会使得原始二次规划问题的目标函数值变得更小。重要的是，这时子问题可以通过解析方法求解，这样就可以大大提高整个算法的计算速度。子问题有两个变量，一个是违反KKT条件最严重的那一个，另一个由约束条件自动确定。如此，SMO算法将原问题不断分解为子问题并对子问题求解，进而达到求解原问题的目的。

下面用手撕一下吧：

![image](http://upload-images.jianshu.io/upload_images/4618424-d065ced8c4b64593?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![image](http://upload-images.jianshu.io/upload_images/4618424-2657601b20c18124?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![image](http://upload-images.jianshu.io/upload_images/4618424-70d0265d26400209?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![image](http://upload-images.jianshu.io/upload_images/4618424-ef37f12d2dcc8dd0?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

4. SVM使用梯度下降求解参数，损失函数应该怎么设计；

在感知机中我们也提到过，使用梯度下降法求解需要保证参数连续可导，**那么看一下这个损失函数，**hinge loss function: 统计学习一书中称为合页损失函数；则线性支持向量机原始最优化问题等价于最优化问题：

原始优化问题：

![image](http://upload-images.jianshu.io/upload_images/4618424-68292f07139a51d3?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

引入hinge loss的损失函数：

![image](http://upload-images.jianshu.io/upload_images/4618424-8094c1fad3cc359a?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![image](http://upload-images.jianshu.io/upload_images/4618424-7777a766a37571d4?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![image](http://upload-images.jianshu.io/upload_images/4618424-6ba95e9b48ff974d?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

下一节：使用SMO求解是SVM参数源码 和 使用梯度下降求解SVM参数代码；

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

19. [机器学习-19：MachineLN之SVM（1）](http://blog.csdn.net/u014365862/article/details/79184858)

20. [机器学习-20：MachineLN之SVM（2）](http://blog.csdn.net/u014365862/article/details/79202089)

![image](http://upload-images.jianshu.io/upload_images/4618424-3530232aa1c14906?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240) 


版权声明：本文为博主原创文章，未经博主允许不得转载。有问题可以加微信：lp9628(注明CSDN)。
