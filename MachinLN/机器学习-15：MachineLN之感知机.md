我想说： 

其实感知机虽然原理简单，但是不得不说他的意义重大，为什们呢？ 他是SVM的前身，后面的SVM是由此进化来的，其实两个结合起来学习会更好的，但是内容太多，SVM三境界，我可能还是停留在“昨夜西风调碧树，独上高楼，望尽天涯路”， 期待突破后面的两重天：“衣带渐宽终不悔，为伊消得人憔碎”， “众里寻他千百度，蓦然回首，那人却在，灯火阑珊处”。说起三境界不得不提佛家三境界：看山是山，看水是水；看山不是山，看水不是水；看山还是山，看水还是水。两者相通互补吧，才疏学浅不敢瞎说，理解还是有点困难的，突然感觉很多事情都是相通的，分久必合，合久必分？乱了乱了，我整天就知道瞎说，别介意。另外最近开始想这么一个问题：什么样的数据不适合用卷积？ 什么样的数据不适合用池化？ 什么样的数据只适合用全连接的结构？ 稍微有点眉目；感觉真的没有通用的网络！！！真是悲哀，以前提通用AI差点被骂死，出来DL后没人再提，只是说针对特定领域特定问题的AI；

看完文章记得最下面的原文链接，有惊喜哦！

说起感知机，那么我的问题：（根据[MachineLN之三要素](http://mp.weixin.qq.com/s?__biz=MzU3MTM3MTIxOQ==&mid=2247483841&idx=2&sn=e4a3cff7b12c48af237c577c487ba3a1&chksm=fce07a7dcb97f36be5003c3018b3a391070bdc4e56839cb461d226113db4c5f24032e0bf5809&scene=21#wechat_redirect)：模型、策略、算法）

（1）什么是感知机？（模型）

（2）感知机是如何学习的？（策略）

（3）感知机学习算法？（算法）

看到这里你的答案是什么？下面是我的答案：

1\. 什么是感知机？

感知机是一个二类分类的线性分类模型。那么说起线性与非线性请参考[MachineLN之激活函数](http://mp.weixin.qq.com/s?__biz=MzU3MTM3MTIxOQ==&mid=2247483968&idx=1&sn=dc2e52c68cd8ea9037b114625c9b1a33&chksm=fce079fccb97f0ea4f3b8f8c74cb779613e06c54a718378c9d174651c16cabb28a1c283b9083&scene=21#wechat_redirect);

感知机模型：

![image](http://upload-images.jianshu.io/upload_images/4618424-44a36ae71aa8c020?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

其中，w和b为感知机参数，w∈Rn叫做权值或权值向量，b∈R叫做偏置，w⋅x表示w和x的内积。感知机学习的目的就在于确定参数w和b的值。符号函数sign(x)不用多解释吧：

![image](http://upload-images.jianshu.io/upload_images/4618424-99db19ed37ed660e?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

看到这里其实可以联系到两个模型：（1）逻辑回归：把sign改为sigmoid就是逻辑回归表达式；（2）SVM：表达式可以定义为一样；策略和算法差不远了去了（为了解决感知机的不足）； 

几何解释：

可以用下面的线性方程表示：

![image](http://upload-images.jianshu.io/upload_images/4618424-b42bb97dc84922e8?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

可以理解时一个特征空间中的一个超平面，其中w是超平面的法向量（为什么？），b是超平面的截距，这个超平面会把[图片上传失败...(image-6c99cb-1516671284374)]

分成两部分，将点分为正、负两部分，看一下图吧：

![image](http://upload-images.jianshu.io/upload_images/4618424-d9e283874370326c?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

下面是w为什么是超平面的法向量？ （看书时候的笔记）

![image](http://upload-images.jianshu.io/upload_images/4618424-146233b5f9ac30af?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

是的感知机就是在找这么一个能够将两个类别数据分开的超平面；并且超平面不是唯一的（参数更新的时候：样本的顺序是很大因素）；（SVM就是将感知机的不唯一变为唯一，后面我们会撸svm原代码，使用拉格朗日直接求解参数，和使用tensorflow的梯度下降求解参数，这时候损失函数要重新定义）

2\. 感知机是如何学习的？

其实这里就是策略，就是常提的损失函数：用来衡量预测值和真实值的相似度的；有时候损失函数直接选择误分类点的总数更直观，但是这个损失函数不是参数的连续可导的函数（那么为什么非要可导：好像无论梯度下降，最小二乘法等都得可导吧，那么为什么非得用梯度下降最小二乘法等？有人说你这是瞎搞，哦nonono，这才是真正要探索的东西，你如果有好的方法不用非让损失函数可导，那么你就厉害了）；

先看一下下面的公式：应该很熟悉吧，点到直线的距离；

![image](http://upload-images.jianshu.io/upload_images/4618424-f2eeb1206be41181?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![image](http://upload-images.jianshu.io/upload_images/4618424-86f2619d233d847d?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

是L2范数应该很明白了。

但是，这里很重要：要弄明白所关心的是什么点到直线的距离作为函数，是分类正确的点？还是分类错误的点？ 提醒到这里大家就很明白，不说透的话是不是感觉云里雾里！那么说到误分类点，它是满足下面条件的：（为什么呢？ 因为我们预测的输出为[-1, 1],误分类点和真实值肯定异号，正确分类的点肯定同号）

![image](http://upload-images.jianshu.io/upload_images/4618424-8a79fac89b89552a?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

那么我们可以重新定义损失函数：就是 误分类点的点到超平面的距离，注意是误分类！！！下一篇代码实现可以格外注意一下；用下面的式子定义： 

![image](http://upload-images.jianshu.io/upload_images/4618424-2e59b4b879be4a0e?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

那么所有误分类点到超平面的总距离可以定义为：

![image](http://upload-images.jianshu.io/upload_images/4618424-dadcfcd124698c5a?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

不考虑![image](http://upload-images.jianshu.io/upload_images/4618424-41af72bc0405f0b5?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

，得到感知机的损失函数为：

![image](http://upload-images.jianshu.io/upload_images/4618424-ac46e3c9fd29dfe4?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

那么问题又来了，为什么不考虑呢？？？ 这也正是它的分类超平面不唯一的原因之一！（在SVM中为什么又考虑呢？）

个人理解：因为感知机任务是进行二分类工作，最终并不关心得到的超平面点的距离有多少（SVM格外关心哦！）（所以我们才可以不去考虑L2范式；）只是关心最后是否正确分类（也就是只考虑误分类点的个数）正如下面这个图（有点糙）x1,x2是一类，x3是一类，对于感知机来说是一样好的，而SVM就是那么最求完美的人，我只要最好！

![image](http://upload-images.jianshu.io/upload_images/4618424-75fb5cafcaac9b01?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

好了，策略有了，该研究通过什么方法来是损失函数最小了，下面介绍算法；

3\. 感知机学习算法？

其实我们机器学习算法就是在求损失函数的最值，用数学表达式表示为：

![image](http://upload-images.jianshu.io/upload_images/4618424-67856a9dff17cb68?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

下面就是随机梯度下降的过程（随机梯度下降就是在极小化损失函数的过程中，每次只选一个误分类点，不使用所有的点）：

下面是损失函数的导数：也就是我们梯度下降中的那个梯度：

![image](http://upload-images.jianshu.io/upload_images/4618424-3ff05698787aae62?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

随机选一个误分类点，进行参数更新： 

![image](http://upload-images.jianshu.io/upload_images/4618424-e3109543ec6465c8?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

式中η(0<η≤1)是步长，又称为学习率，通过迭代的方式可以不断减小损失函数；（如果是看成数学问题，那么就严重了，说话得有根据，这里还要证明算法的收敛性。。。）

那么感知机原始算法的形式可以总结为：

![image](http://upload-images.jianshu.io/upload_images/4618424-84dd22a6f95042de?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

算法的对偶形式：（对偶？）

对偶形式的基本想法是，将w,b表示成为实例x<sub>i</sub>和标记y<sub>i</sub>的线性组合的形式，通过求解其系数而得到w和b。不失一般性，将初始值w<sub>0</sub>,b<sub>0</sub>设为0.对误分类点（x<sub>i</sub>,y<sub>i</sub>）通过

![image](http://upload-images.jianshu.io/upload_images/4618424-08da8b61f2b7aaa6?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

逐步修该w,b，设修改了n次，则w,b关于（xi,yi）的增量分别为a<sub>i</sub>y<sub>i</sub>x<sub>i</sub>和a<sub>i</sub>y<sub>i</sub>，这里a<sub>i</sub>=n<sub>i</sub>η最终学习到的w,b可以表示为

![image](http://upload-images.jianshu.io/upload_images/4618424-df792da60786f4f5?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

实例点更新次数越多，意味着它距离分离超平面越近，也就越难正确分类。换句话说，这样的实例对学习结果影响很大。

那么感知机算法的对偶形式可以总结为：

![image](http://upload-images.jianshu.io/upload_images/4618424-bbfeacf3bc642e86?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

好好理解一下对偶，SVM也需要对偶一下的； 

好了感知机理论说到这里，有疑惑留言哦，下一遍我们上感知机源代码！

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


![image](http://upload-images.jianshu.io/upload_images/4618424-e648a32ee6d56a4e?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
