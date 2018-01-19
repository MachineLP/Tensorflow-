
我想说：

可能一直关注我更新文章的童鞋，可能看出我的布局，基本是先搭一个框架然后挖坑去填，这可能是我做事情一个优点，当接触到新事物总是能快速建立一个框架，然后去慢慢填，可能刚开始建立的框架是错的，但是没关系，后面随着认知的加深慢慢去改，这可能与我数学比较好有点关系（又开始了...对你无语！！！），跟着清华宁向东老师学习管理学半年，感觉在宁老师上课方式跟我学习知识有点相似（当然应该是我跟宁老师相似），框架搭好挖坑去填，然后多问为什么？另外我也一直反对老师上课用ppt，为什么不用板书，由以前的事半功倍，变成现在事倍功半，反而让学生课后要花更多时间去自己琢磨学习，爱学习的还好，就像我这种不爱学习的简直是大坑。清华老校长梅贻琦先生的话：大学者，非有大楼之谓也，而有大师之谓也。

今天我们来研究cnn的源码，不用dl框架，前边文章已经对卷积、池化、全连结、前向传播、后向传播等做了铺垫，还少了激活函数(稍微提一下，使解决非线性成为可能，同时选择不当会导致梯度后向传播失败的问题)、BN(解决训练过程中数据弥散、加速训练，抗过拟合、弥补激活函数造成梯度后向传播失败的问题)等文章，后面会慢慢填起来。

又是截图哈哈，个人观点：好代码是敲出来的；从来不是搬出来的；

开始顺代码：

![image.png](http://upload-images.jianshu.io/upload_images/4618424-434dd0cc1b5ad36f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


![image.png](http://upload-images.jianshu.io/upload_images/4618424-7a0e3f6e1d2ffb76.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![image.png](http://upload-images.jianshu.io/upload_images/4618424-4a93f5645b5046e9.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![image.png](http://upload-images.jianshu.io/upload_images/4618424-90dcb21ff3dd7a7b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![image.png](http://upload-images.jianshu.io/upload_images/4618424-47fb284474252b4f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![image.png](http://upload-images.jianshu.io/upload_images/4618424-e54a4e679f340fcd.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![image.png](http://upload-images.jianshu.io/upload_images/4618424-ad3690cd09215f88.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![image.png](http://upload-images.jianshu.io/upload_images/4618424-4a32f5a9660e143b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![image.png](http://upload-images.jianshu.io/upload_images/4618424-e810593680f35f3a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![image.png](http://upload-images.jianshu.io/upload_images/4618424-93fffada1d02c1fa.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![image.png](http://upload-images.jianshu.io/upload_images/4618424-60797c548e39ae84.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![image.png](http://upload-images.jianshu.io/upload_images/4618424-464e7c63af4d87f0.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![image.png](http://upload-images.jianshu.io/upload_images/4618424-d587460b28e7f356.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


推荐阅读：

1. [MachineLN之三要素](http://mp.weixin.qq.com/s?__biz=MzU3MTM3MTIxOQ==&mid=2247483841&idx=2&sn=e4a3cff7b12c48af237c577c487ba3a1&chksm=fce07a7dcb97f36be5003c3018b3a391070bdc4e56839cb461d226113db4c5f24032e0bf5809&scene=21#wechat_redirect)

2. [MachineLN之模型评估](http://mp.weixin.qq.com/s?__biz=MzU3MTM3MTIxOQ==&mid=2247483872&idx=2&sn=8436e1eb9055d3a372278ee8688cd703&chksm=fce07a5ccb97f34a4490f60304b206c741d2395149c2c2e68bddb3faf7daf9121ca27a5d6a97&scene=21#wechat_redirect)

3. [MachinLN之dl](http://mp.weixin.qq.com/s?__biz=MzU3MTM3MTIxOQ==&mid=2247483894&idx=2&sn=63333c02674e15e84159e064073fe563&chksm=fce07a4acb97f35cc38f75dc891a19129e2406270d04b739cfa9b8a28f9780b4e2a65a7cd39b&scene=21#wechat_redirect)

4. [DeepLN之CNN解析](http://mp.weixin.qq.com/s?__biz=MzU3MTM3MTIxOQ==&mid=2247483906&idx=1&sn=2eceda7d9703d5315638739e04d5b6e7&chksm=fce079becb97f0a8b8dd2e34a9e757f23757cf2699c397707bfaa677c9f8204c91508840d8f7&scene=21#wechat_redirect)

[5\. DeepLN之手撕CNN权值更新（笔记）](http://mp.weixin.qq.com/s?__biz=MzU3MTM3MTIxOQ==&mid=2247483927&idx=1&sn=132b90eb009022d2d792303e6d824df7&chksm=fce079abcb97f0bd9b893889a7c8232c0254e447277b930ab8e35ea1f9c49e57c0fc66bab239&scene=21#wechat_redirect)

![image.png](http://upload-images.jianshu.io/upload_images/4618424-35eca4d01471376b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


版权声明：本文为博主原创文章，未经博主允许不得转载。有问题可以加微信：lp9628(注明CSDN)。
