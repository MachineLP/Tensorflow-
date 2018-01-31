
我想说：

其实很多东西还是要靠自己，靠自己去完成最大的一点就是远离舒适区，试想一下自己每时每刻都要为下一顿能不能吃上饭而奋斗，是一种什么样的体验，估计你连想都不敢想；最近又听到说下岗的问题，有一个人说他除了收钱什么都不会，有时候也要多培养点自己的能力，做好一项，其他的也了解（当然也不了太多），多给自己备好能力，远离舒适区，但无论在哪里都有这么一批人，那你考虑过没有公司万一不景气，第一个下岗的会是谁？下岗了又可以迅速跨到别的领域的又是谁？我做不到这一点，但我在加油，要永远记住：公司不养闲人！比你优秀的人比你还努力，你还好意思说你不会？不会可以学啊，不学永远不会，哈哈，言辞过激了吗，也不知道咋地，最近着魔了吧！！！

除了宁向东的清华管理学课，又在书单中加了香帅的北大金融学课，我也不是想什么都会，装逼啥滴，我也只是想每周拿出两个小时，学点管理学和金融学的的思维方式而已。

下面是加详细注释的感知机代码：又是截图，记住好代码都是敲出来的！下面代码要结合感知机原理来看：[MachineLN之感知机](http://mp.weixin.qq.com/s?__biz=MzU3MTM3MTIxOQ==&mid=2247484156&idx=1&sn=5e3e5baab2701bda0c17ef0edb60bac0&chksm=fce07940cb97f056eae2cd8e52171092e893886d731f6bfaaec685ec67400ec82dd1288a04fa&scene=21#wechat_redirect)

但是代码很方便理解，还有图示：

1\. 原始形式的感知机算法：  

![image](http://upload-images.jianshu.io/upload_images/4618424-61fd8bfb209d6a29?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![image](http://upload-images.jianshu.io/upload_images/4618424-e638e559c0cf4bcd?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![image](http://upload-images.jianshu.io/upload_images/4618424-87a7b41bd4aa95d0?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![image](http://upload-images.jianshu.io/upload_images/4618424-469bbfa3386a4b24?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![image](http://upload-images.jianshu.io/upload_images/4618424-2b69de5411a722fe?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![image](http://upload-images.jianshu.io/upload_images/4618424-07bf7640e65ab998?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

用来显示样本点 和 分类线：

![image](http://upload-images.jianshu.io/upload_images/4618424-2d1d258f55f85cb6?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240) 

看一下感知机的分类结果：


![image](http://upload-images.jianshu.io/upload_images/4618424-dce56335661c9cfd?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

* * *

2\. 对偶形式的感知机算法 （对偶形式要计算gram矩阵）  

![image](http://upload-images.jianshu.io/upload_images/4618424-509d7fea59dcb821?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

原理跟上面原始感知机对偶算法差不多，所以只加了简单注释：

![image](http://upload-images.jianshu.io/upload_images/4618424-d5acdbdd1eae2487?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![image](http://upload-images.jianshu.io/upload_images/4618424-e8b5f6554951718e?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

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


![image](http://upload-images.jianshu.io/upload_images/4618424-14fe247ce5f45412?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

版权声明：本文为博主原创文章，未经博主允许不得转载。有问题可以加微信：lp9628(注明CSDN)。
