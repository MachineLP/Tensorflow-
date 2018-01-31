
我想说：

其实做自己喜欢的事情，做一个快乐的码农最重要，灵感也会由此产生！ 

上一节：[MachineLN之逻辑回归](http://mp.weixin.qq.com/s?__biz=MzU3MTM3MTIxOQ==&mid=2247484184&idx=1&sn=8623b05d0e4847ab7464740809ba2618&chksm=fce078a4cb97f1b233d9be8c62dcbf16d8be0d4215ab1f94f7c3fdbbd9739586a37ae17ccc5c&scene=21#wechat_redirect)，讲述了逻辑回归的原理，今天看一下带详细注释的源码：切记好代码都是敲出来的，并且越敲越有感觉：

![image](http://upload-images.jianshu.io/upload_images/4618424-40861703ea5803d0?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![image](http://upload-images.jianshu.io/upload_images/4618424-c55b446a3a75e29e?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

还记得没，上一届提到的梯度上升和梯度下降的不同？

![image](http://upload-images.jianshu.io/upload_images/4618424-14089ea8d0c5dd57?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![image](http://upload-images.jianshu.io/upload_images/4618424-24a66213debddb36?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![image](http://upload-images.jianshu.io/upload_images/4618424-aab344c6d18b005f?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![image](http://upload-images.jianshu.io/upload_images/4618424-8f8e93c77c0c850d?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![image](http://upload-images.jianshu.io/upload_images/4618424-c9ac7361890ace67?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![image](http://upload-images.jianshu.io/upload_images/4618424-c4332ea59684408a?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![image](http://upload-images.jianshu.io/upload_images/4618424-db851657766aa9f8?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

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

11. [MachineLN之过拟合](http://mp.weixin.qq.com/s?__biz=MzU3MTM3MTIxOQ==&mid=2247484026&idx=1&sn=36329f7ee26fb675627d64884ac02d0f&chksm=fce079c6cb97f0d05bb1d67218203bfc4b1234a47404fc9e38399c2f0aca9fe21455c0a97b7c&scene=21#wechat_redirect)

[12\. MachineLN之优化算法](http://mp.weixin.qq.com/s?__biz=MzU3MTM3MTIxOQ==&mid=2247484064&idx=1&sn=f834585e2e77299f28148084e6fddcbc&chksm=fce0791ccb97f00a1c168fc05725c59f9e5be5fd22249e3c49e949d5daf812a9ca9da365b50e&scene=21#wechat_redirect) 

13. [ReinforcementLN之图像分类](http://mp.weixin.qq.com/s?__biz=MzU3MTM3MTIxOQ==&mid=2247484102&idx=1&sn=a44c0efaa3d939d148cc0d3dd2dc06b9&chksm=fce0797acb97f06c83cf33f711165dcef005dd54478503098f904e41800fe3beb855a579ed78&scene=21#wechat_redirect)

14. [MachineLN之kNN](http://mp.weixin.qq.com/s?__biz=MzU3MTM3MTIxOQ==&mid=2247484116&idx=1&sn=b26ee03d2e520aa429e424950e59943d&chksm=fce07968cb97f07e6414d3dad8f1744c39cf99bc4f0f68ead7d153f33bf39b46a513d6223ee2&scene=21#wechat_redirect) 

15. [MachineLN之kNN源码](http://mp.weixin.qq.com/s?__biz=MzU3MTM3MTIxOQ==&mid=2247484129&idx=1&sn=2309171e8414fc3b174743367bdb6234&chksm=fce0795dcb97f04b1cbfb9e5b9b0c36d36309065b782201a5a93bd51a4216c3c61d8024e4a99&scene=21#wechat_redirect)

16. [MachineLN之感知机](http://mp.weixin.qq.com/s?__biz=MzU3MTM3MTIxOQ==&mid=2247484156&idx=1&sn=5e3e5baab2701bda0c17ef0edb60bac0&chksm=fce07940cb97f056eae2cd8e52171092e893886d731f6bfaaec685ec67400ec82dd1288a04fa&scene=21#wechat_redirect)

17. [MachineLN之感知机源码](http://mp.weixin.qq.com/s?__biz=MzU3MTM3MTIxOQ==&mid=2247484170&idx=1&sn=98c56eee629a2731f2a7cc99f4fa0e83&chksm=fce078b6cb97f1a047e5697b72c63947d546117dab7618a52e40190b60ee839259a8c4ef73f1&scene=21#wechat_redirect) 

18. [MachineLN之逻辑回归](http://mp.weixin.qq.com/s?__biz=MzU3MTM3MTIxOQ==&mid=2247484184&idx=1&sn=8623b05d0e4847ab7464740809ba2618&chksm=fce078a4cb97f1b233d9be8c62dcbf16d8be0d4215ab1f94f7c3fdbbd9739586a37ae17ccc5c&scene=21#wechat_redirect)

![image](http://upload-images.jianshu.io/upload_images/4618424-7e4b2ec93bfeaf17?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)



版权声明：本文为博主原创文章，未经博主允许不得转载。有问题可以加微信：lp9628(注明CSDN)。
