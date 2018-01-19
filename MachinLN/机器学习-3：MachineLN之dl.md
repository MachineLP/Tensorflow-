

开篇废话：

嫌废话太多可以直接跳到正文哦。

对外人提起人工智能感觉很牛逼很了不起、高大上或者一些其他的吹捧、羡慕的词都出来，那么今天就通过一篇文章带到dl的世界，如果你是小白这篇文章会感觉挺好玩，如果你是大牛，你会感觉这个玩意谁不会？！不管怎么样请保持平常心，因为深度学习很普通，并且现在很多人感觉已经遇到了瓶颈，大牛都在寻找其他更好的方法，但不得不承认dl确实比传统机器学习的方法好，先看一下dl的局限性，给你挖一些坑自己去填可好？（1）目前深度学习需要大量数据；（2）深度学习目前还是太表浅，没有足够的能力进行迁移；（3）迄今深度学习没有自然方式来处理层级架构；（3）迄今为止的深度学习无法进行开放推理；（4）迄今为止的深度学习不够透明；（5）迄今为止，深度学习并没有很好地与先验知识相结合；（6）到目前为止，深度学习还不能从根本上区分因果关系和相关关系；（7）深度学习假设世界是大体稳定的，采用的方式可能是概率的；（8）到目前为止，深度学习只是一种良好的近似，其答案并不完全可信；（9）到目前为止，深度学习还难以在工程中使用；以上是Gary Marcus提出对深度学习的系统性批判，文章地址：https://arxiv.org/ftp/arxiv/papers/1801/1801.00631.pdf；那么我的问题又来了，以上9个方面为什么？搞明白了局限性再去思考值不值得学习，不要随大流，李开复老师有句话说的好：千万不要让别人驾驶你的生命之车，你要稳稳地坐在司机的位置上，决定自己何时要停、倒车、转弯、加速、刹车。可以参考别人的意见，但不要随波逐流。估计有人又开始骂了，你tm的废话真多，赶紧上干货，呃，不好意思，能再说几句吗？好吧，总之不建议大家都来追潮流，当大家都去做一件事情的时候，那么机会往往在相反的方向上；估计很多人都记得那条让人深思的技术成熟的曲线，先是疯狂的上去，然后又快速的下来，然后缓慢的爬行；又有人说你太悲观了，过去已去，把握现在，未来谁知？实时确实是这样，大多数人也都这样做的，包括我。

说了这么多废话，该回到dl了。简单说一下AI；AI=感知+理解+决策，目前的dl就是处理AI中的理解问题，无论是人脸检测、识别、行人检测、车辆检测、车道线、语音识别等等；感知就是获取图像、语音等的设备，决策像无人驾驶中来不及刹车时，左右是沟，中间是人，你怎么决策？损人利己？还是损己利人？哲学问题，扯得远了，但是真要商用，确实要面临的问题很多。

那么我的问题是？

（1）你学dl的目的是什么？

（2）你喜欢你现在的工作吗？dl对你现在的工作有什么帮助？

（3）如果那天dl热潮过了，你还有还什么技能可能养家糊口？

接下来不解答上面问题了，而是看一个手写体识别的代码，带你进入word世界：

咱们顺着代码的流程走一遍，估计会有很多坑，有坑时好事，正如学习再多tricks，不如猜一遍坑，（你废话真多，好了好了我不说了）后面的一些文章会逐渐来填起来；are you ready？

这是一段cnn的代码来决解手写体数字识别，（初学者会问了cnn是什么？有坑了？自己填去啊，哈哈）

（1）开始导入一下tensorflow，和去获取一下mnist的数据，用于后面的训练和测试![image](http://upload-images.jianshu.io/upload_images/4618424-d0b360cb96b2177f?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

（2）定义卷积：（问题：卷积是什么？ 为什么用卷积？作用是什么？）

![image](http://upload-images.jianshu.io/upload_images/4618424-49ee09f530913d40?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

（3）定义池化：（问题和上面一样，别嫌我话多，面试时候会问到）

![image.png](http://upload-images.jianshu.io/upload_images/4618424-a84a0a4ca7afd07e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


（4）定义BN：（问题和上面一样）

![image](http://upload-images.jianshu.io/upload_images/4618424-eab4d774297cf02a?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

（5）初始化参数：卷积核参数；全连接参数；

![image](http://upload-images.jianshu.io/upload_images/4618424-4ff1ea5cdcc1af51?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

（6）该盖楼了。

![image.png](http://upload-images.jianshu.io/upload_images/4618424-c1d62c9ce85e4e36.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


（7）训练模型，其中含有了（机器学习三要素中的模型、策略、算法；还有机器学习之模型评估中的loss和准确率），有了上面的基础，下面代码就好理解了。

![image](http://upload-images.jianshu.io/upload_images/4618424-ce7a91c7abb87162?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![image](http://upload-images.jianshu.io/upload_images/4618424-a1d7b714e4226e28?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

（8）一些参数和占位符。

![image](http://upload-images.jianshu.io/upload_images/4618424-90134ba442644cce?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

# 上面的代码整合起来就可以跑了；

看一下结果： loss会慢慢降，而准确率会突变，这又是为什么？

![image](http://upload-images.jianshu.io/upload_images/4618424-e04eb32b1b5f121a?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

说明：像一些衰减和正则化的方法里边没加，后面会慢慢补充，下面给大家一个简单定义模型的框架：

![image](http://upload-images.jianshu.io/upload_images/4618424-bc4cda3fd34b7700?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![image](http://upload-images.jianshu.io/upload_images/4618424-17697657140047c0?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

（9）总结：入门级的感觉这个就够了，把上面的代码玩透了，调调参，挖了坑一定要填起来，要不前功尽弃；到这里是一个节点说明你想明白了，你想学习dl，那就关注我们吧，一起坚持下去，MachineLN与你一年之约，只要想学什么时候开始都不晚。

推荐阅读：

1. [MachineLN之三要素](http://mp.weixin.qq.com/s?__biz=MzU3MTM3MTIxOQ==&mid=2247483841&idx=2&sn=e4a3cff7b12c48af237c577c487ba3a1&chksm=fce07a7dcb97f36be5003c3018b3a391070bdc4e56839cb461d226113db4c5f24032e0bf5809&scene=21#wechat_redirect)

2. [MachineLN之模型评估](http://mp.weixin.qq.com/s?__biz=MzU3MTM3MTIxOQ==&mid=2247483872&idx=2&sn=8436e1eb9055d3a372278ee8688cd703&chksm=fce07a5ccb97f34a4490f60304b206c741d2395149c2c2e68bddb3faf7daf9121ca27a5d6a97&scene=21#wechat_redirect)

![image.png](http://upload-images.jianshu.io/upload_images/4618424-ab37d62464cc2796.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


</article>

版权声明：本文为博主原创文章，未经博主允许不得转载。有问题可以加微信：lp9628(注明CSDN)。
