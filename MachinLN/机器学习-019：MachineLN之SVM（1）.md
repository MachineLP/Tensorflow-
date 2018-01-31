
我想说：

其实很多事情，只要你想做，是肯定有方法做的，并且可以做好；

说起SVM很多人都会感觉头疼，无非就是公式多一个，其实很多时候你真是用的话，都不用你手动自己实现，你是在学习的一种机器学习的思维方式，就要比为什么要提出svm？svm解决了什么问题？svm中的kernel又是想解决线性svm解决不了的问题？svm的优势在哪里？就好比生活中不缺乏美，只是缺少发现美的眼睛，在学习中发现问题的能力及其重要，当你问题多了很多人会感觉你烦，但是没关系，解决了就会柳暗花明；并且要时常问自己从中学到了什么？再遇到问题是否可以拿来主义？还是可以从中借鉴？

说起SVM，那么我的问题：（根据[MachineLN之三要素](http://mp.weixin.qq.com/s?__biz=MzU3MTM3MTIxOQ==&mid=2247483841&idx=2&sn=e4a3cff7b12c48af237c577c487ba3a1&chksm=fce07a7dcb97f36be5003c3018b3a391070bdc4e56839cb461d226113db4c5f24032e0bf5809&scene=21#wechat_redirect)：模型、策略、算法）

（1）什么是SVM？（模型）

（2）SVM是如何学习的？（策略）

（3）SVM学习算法？（算法）

顺便后面的要写的展望一下： SVM软间隔最大化；SVM核技巧；SVM求解对偶问题的SMO算法；SVM不用提到的拉格朗日求解，使用梯度下降损失函数应该怎么设计；SVM源码：smo算法求解参数 和 使用梯度下降求解参数；

看到这里你的答案是什么？下面是我的答案：

（1）什么是SVM？（模型）

在[MachineLN之感知机](http://mp.weixin.qq.com/s?__biz=MzU3MTM3MTIxOQ==&mid=2247484156&idx=1&sn=5e3e5baab2701bda0c17ef0edb60bac0&chksm=fce07940cb97f056eae2cd8e52171092e893886d731f6bfaaec685ec67400ec82dd1288a04fa&scene=21#wechat_redirect)中有提到：感知机的不足和svm的提出；

SVM（支持向量机）表达式：

![](http://upload-images.jianshu.io/upload_images/4618424-01eaf8cf671dcfb7?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

那么分类超平面：

![image](http://upload-images.jianshu.io/upload_images/4618424-93906af01e2f9eb2?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

这里和感知机是一样的，不清楚的可以回过头看一下；不同的是在策略和算法上； 

（2）SVM是如何学习的？（策略）

先看这么一句话，开始看可能比较难理解：下面我画个图就好理解了，一般来说，一个点距离分离超平面的远近可以表示分类预测的确信程度，在超平面wx+b=0确定的情况下，|w x+b|能够相对地表示点距离超平面的远近，看下图：

![image](http://upload-images.jianshu.io/upload_images/4618424-58619ee268effa71?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

这个理解了，那么提到函数间隔应该就好理解了，在感知机章节我们就注意过wx+b与类标记符号y的问题，他俩是否一致表示分类似否正确，所以可用y(wx+b)表示分类的正确性和确信度，这就是函数间隔； 

那么函数间隔 functiona lmargin：对于给定的训练数据集T和超平面(w, b)，定义超平面关于样本点(x<sub>i</sub>, y<sub>i</sub>)的函数间隔为：

![image](http://upload-images.jianshu.io/upload_images/4618424-04e142ff8828b3b1?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

定义超平面(w,b)关于训练数据集T的函数间隔为超平面(w,b)关于T中所有样本点(x<sub>i</sub>, y<sub>i</sub>)的函数间隔之最小值，即：

![image](http://upload-images.jianshu.io/upload_images/4618424-59ccbee7718a0cc6?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

但是把手撕那部分除以||w||作为超平面，你会发现，此时w,b成倍的改变，超平面不变，h1和h2也不变，这就引出了几何间隔，也可以直接理解为点到直接的距离。（大家不要怪学术的大牛不点透点，这些都是基础）  

接下来几何间隔 geometric margin：对于给定的训练数据集T和超平面(w, b)，定义超平面关于样本点(x<sub>i</sub>, y<sub>i</sub>)的函数间隔为：

![image](http://upload-images.jianshu.io/upload_images/4618424-f0a2c8e5553bcdc1?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

定义超平面(w,b)关于训练数据集T的函数间隔为超平面(w,b)关于T中所有样本点(x<sub>i</sub>, y<sub>i</sub>)的函数间隔之最小值，即：

![image](http://upload-images.jianshu.io/upload_images/4618424-9f94a50d3367fed2?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

那么接下来就可以引出支持向量机的思想：求解能够正确分类训练集并且几何间隔最大的分类超平面，对线性可分的训练数据集而言，线性可分分离超平面有无穷多个(等价于感知机)，但是几何间隔最大的分离超平面是唯一的。这里的间隔最大化又称为硬间隔（有硬就有软）最大化。

定义SVM的策略为：

（1）几何间隔最大化；

（2）并且每个样本点的几何间隔大于设最大函数间隔；

可表示为：

![image](http://upload-images.jianshu.io/upload_images/4618424-140b8af58deb5050?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

整理后： 

![image](http://upload-images.jianshu.io/upload_images/4618424-de8e012434c5000d?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

你会发现成倍的增加w，b对上式没有影响，那么就可以转化为一个等价问题，将![image](http://upload-images.jianshu.io/upload_images/4618424-e89cfcb0abf85d42?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

取1；整理后可得：

![image](http://upload-images.jianshu.io/upload_images/4618424-06569bc4df7050ed?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

说到这里了就可以谈一下支持向量的由来：在线性可分情况下，训练数据集的样本点中与分离超平面跄离最近的样本点的实例称为支持向量( support vector )。支持向量是使约束条件式等号成立的点，即

![image](http://upload-images.jianshu.io/upload_images/4618424-133a527f5e3ec59a?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

对于二分类yi的取值为:[-1, 1]，那么应该有表达式满足上式： 

![image](http://upload-images.jianshu.io/upload_images/4618424-dc2107683c3dd759?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

对这两个就是支持向量H1和H2，看下图： 

![image](http://upload-images.jianshu.io/upload_images/4618424-cb8ef9d5f9db6f1e?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

到这里svm的原理应该大概清楚了；下面就是如何求解参数的问题了。

（3）SVM学习算法？（算法）

转为对偶问题（KKT条件成立）：对于拉格朗日大家应该很熟悉，用来构建函数求解凸优化问题，svm优化问题引入拉格朗日因子后成了：  

![image](http://upload-images.jianshu.io/upload_images/4618424-30bc6ab8ce7ce643?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

根据拉格朗日对偶性，原始问题的对偶问题是拉格朗日函数的极大极小问题：

![image](http://upload-images.jianshu.io/upload_images/4618424-d389a75faaf1a759?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

求最值问题我们最擅长的是求导，那么接下来就手撕一下吧： 

![image](http://upload-images.jianshu.io/upload_images/4618424-b8adee665bdabb3b?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

可总结为： 

![image](http://upload-images.jianshu.io/upload_images/4618424-e9c3672da9907e7b?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

先聊到这里吧，接下来的内容：SVM软间隔最大化；SVM核技巧；SVM求解对偶问题的SMO算法；SVM不用提到的拉格朗日求解，使用梯度下降损失函数应该怎么设计；SVM源码：smo算法求解参数 和 使用梯度下降求解参数；更精彩！

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

![image](http://upload-images.jianshu.io/upload_images/4618424-e4e0c6bf47a30a0e?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


版权声明：本文为博主原创文章，未经博主允许不得转载。有问题可以加微信：lp9628(注明CSDN)。
