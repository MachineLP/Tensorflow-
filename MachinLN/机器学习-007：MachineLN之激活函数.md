

很长一段时间都在想，有些问题不去弄明白为什么，遇到瓶颈就傻逼了，一个bug整你一个月，原来只是一个细节问题，就好如：你不知道从哪里来？ 又怎么知道往哪里去？ 现在遗留的小问题，将来都会是大问题！

真的，有时候需要回过头来重新开始，整理总结再去前行，也许会走的更远。

那么我的问题是：

（1）什么是激活函数？

（2）激活函数的作用是什么？

（3）激活函数有哪些？

（4）各自的优缺点是什么？（解答完1、2、3，就有了答案了）

看到这里，你的答案是什么？ 下面是我的答案：

（1）什么是激活函数？

激活函数就是加在在神经元后的函数，下图所示（例如我们前边提到的在cnn卷积候连接的函数relu，还有Sigmoid、tanh、prelu等等），那么它有什么用呢？不加会怎么样？（2）中回答；

![image.png](http://upload-images.jianshu.io/upload_images/4618424-e5e308aaded8c071.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


当然你也最好了解下面概念，到提起梯度消失就容易理解了，正是因为这么饱和把梯度会传的时候越来越小，使得更新参数失败，整个网络瘫痪：

![image.png](http://upload-images.jianshu.io/upload_images/4618424-1b73e1f039dd7ebe.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


（2）激活函数的作用是什么？ 

简单说：为了解决非线性问题，不加激活函数只能处理线性问题。

下面图来自公众号忆臻笔记，一个用心在做的公众号。

先来看一个图：左侧的网络对应其下方的数学表达式，是一个线性方程（如果这里你还问为什么，那么see you），令其为0，可以画出一条直线，就是右侧的图了，右上方就是y>0的区域，左下方就是y<0的区域，对吧？ 那么它是线性的。

![image.png](http://upload-images.jianshu.io/upload_images/4618424-8013f4f9543424ca.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


那么你再看下面这个图：网络结构用右侧的表达式表示。

![image](http://upload-images.jianshu.io/upload_images/4618424-17dcf6f042d1e57d?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

你可以将右侧的表达式整理一下，其实它是这样的，你会发现原来是它也是线性的。

![image.png](http://upload-images.jianshu.io/upload_images/4618424-3fb20633a3b37718.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


对，你想的没错，没有激活函数的网络只能处理线性问题（佩服大牛！），在看下面的图，在神经元后加上激活函数，至少可以保证输出是非线性的，那么能解决非线性问题吗？再往下走。

![image.png](http://upload-images.jianshu.io/upload_images/4618424-8dec9077d08dfab0.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

然后看下图，多个神经元的情况：

![image.png](http://upload-images.jianshu.io/upload_images/4618424-eb43d05735772411.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


在看一下大神的图：

![image.png](http://upload-images.jianshu.io/upload_images/4618424-c4748d4ad624a9b7.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


是的通过加上激活函数就可以解决这种非线性问题；

![image.png](http://upload-images.jianshu.io/upload_images/4618424-e4f8c72a289afcd4.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


（3）激活函数有哪些？

详细介绍一个sigmod的激活函数：![image.png](http://upload-images.jianshu.io/upload_images/4618424-9372232da6705810.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


左侧是sigmoid函数图，右侧是sigmoid导数的函数图，由[DeepLN之CNN权重更新](http://mp.weixin.qq.com/s?__biz=MzU3MTM3MTIxOQ==&mid=2247483927&idx=1&sn=132b90eb009022d2d792303e6d824df7&chksm=fce079abcb97f0bd9b893889a7c8232c0254e447277b930ab8e35ea1f9c49e57c0fc66bab239&scene=21#wechat_redirect)中的公式5，可知在梯度回传过程中激活函数的影响梯度的问题，当数据落在sigmoid导数的红色区域时就会造成梯度很小，甚至梯度消失；这个问题可以通过改善激活函数来改善，当然也可以通过改变激活函数的输入x来改善，怎么改善？把x重新拉回到0附近不就ok了吗！那么你想到了什么？BN啊！！！那么你还有什么方法？修改激活函数啊，说过了，还有什么？resdual啊！！！对吗？哦，那么还可以有什么？当然真实使用的时候都是一起使用的，例如resdual+bn+relu等（当然bn还有其他作用，这只是其一）。

![image](http://upload-images.jianshu.io/upload_images/4618424-16848a7aa66d378b?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![image](http://upload-images.jianshu.io/upload_images/4618424-3f72c649b2acde00?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

激活函数还有**tanh函数、**ReLU函数、**ELU函数、PReLU函数等，为什么提取出来？ 解决什么问题？ 上面提到的左饱和、右饱和或者称为单侧抑制，对你理解这么激活函数有什么帮助？**

展望，接下来更新：

DeepLN之BN；

MachineLN之过拟合；

MachineLN之数据归一化；

DeepLN之优化（sgd等）：

...

感觉这些是在一起的，之后补习传统机器学习的理论知识 到 实践；

还有之前提到的；

推荐阅读：

1. [MachineLN之三要素](http://mp.weixin.qq.com/s?__biz=MzU3MTM3MTIxOQ==&mid=2247483841&idx=2&sn=e4a3cff7b12c48af237c577c487ba3a1&chksm=fce07a7dcb97f36be5003c3018b3a391070bdc4e56839cb461d226113db4c5f24032e0bf5809&scene=21#wechat_redirect)

2. [MachineLN之模型评估](http://mp.weixin.qq.com/s?__biz=MzU3MTM3MTIxOQ==&mid=2247483872&idx=2&sn=8436e1eb9055d3a372278ee8688cd703&chksm=fce07a5ccb97f34a4490f60304b206c741d2395149c2c2e68bddb3faf7daf9121ca27a5d6a97&scene=21#wechat_redirect)

3. [MachinLN之dl](http://mp.weixin.qq.com/s?__biz=MzU3MTM3MTIxOQ==&mid=2247483894&idx=2&sn=63333c02674e15e84159e064073fe563&chksm=fce07a4acb97f35cc38f75dc891a19129e2406270d04b739cfa9b8a28f9780b4e2a65a7cd39b&scene=21#wechat_redirect)

4. [DeepLN之CNN解析](http://mp.weixin.qq.com/s?__biz=MzU3MTM3MTIxOQ==&mid=2247483906&idx=1&sn=2eceda7d9703d5315638739e04d5b6e7&chksm=fce079becb97f0a8b8dd2e34a9e757f23757cf2699c397707bfaa677c9f8204c91508840d8f7&scene=21#wechat_redirect)

[5\. DeepLN之手撕CNN权值更新（笔记）](http://mp.weixin.qq.com/s?__biz=MzU3MTM3MTIxOQ==&mid=2247483927&idx=1&sn=132b90eb009022d2d792303e6d824df7&chksm=fce079abcb97f0bd9b893889a7c8232c0254e447277b930ab8e35ea1f9c49e57c0fc66bab239&scene=21#wechat_redirect)

6. [DeepLN之CNN源码](http://mp.weixin.qq.com/s?__biz=MzU3MTM3MTIxOQ==&mid=2247483948&idx=1&sn=d1edce6e99dac0437797404d15714876&chksm=fce07990cb97f086f2f24ec8b40bb64b588ce657e6ae8d352d4d20e856b5f0e126eaffc5ec99&scene=21#wechat_redirect)

![image](http://upload-images.jianshu.io/upload_images/4618424-3e8d004c8823f126?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)



版权声明：本文为博主原创文章，未经博主允许不得转载。有问题可以加微信：lp9628(注明CSDN)。
