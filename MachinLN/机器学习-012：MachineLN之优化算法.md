
我想说： 

其实很多时候应该审视一下自己，知道自己的不足和长处，然后静下来去做一些事情，只有真正静下来才能深下去，只有深下去了才能有所突破，不要被别人的脚步带跑，无论什么时候专而精更重要，同时我也知自己的不足，有点狂、有点浮躁、坚持自己观点喜欢争论、说话有时候伤人等等，但是我的优点也正在此（下面是05年9月份写的《自己-社会-机器学习》的一篇文章，虽然有点浮躁，但是值得我再去回顾）：感觉自己成长了不少，不再抱怨，不再发脾气，不再那么要强，不再看重别人的眼光，更加注重自己的评价，开始接受一些事情，棱角开始慢慢圆滑，但是我了解自己，是绝不会消失，有些东西决不能随波逐流，社会锻炼了我们，最终也会将越来越好的自己放到社会中实践，这是一个无限循环的事情，最后的结果就是社会和我们都将越来越好，这也是一个漫长的过程，也需要充足的空间给我们释放，这就要看你的程序的时间复杂度和空间复杂度，这个好了，过程就会快一点，其实想一下，很多时候，我们就是在找一个最优解，但是社会的进步估计我们永远找到的也只能是局部最优了吧，也就是说在某个时间段我们尽最大可能想到的最好决策，至于全局最优解，这个问题还真是个无人能解的问题吧，马克思列宁提的共产主义可能就是我们最想要的那个损失函数的最小值，但是怎么能找到那个最适合的权重呢，来达到全局最优，值得思考？我们可能要像梯度下降那样了，慢慢的来调节权重，达到某阶段的最优，当然大神们都有自己的方法，这点不能否认，但是弯路是要走的，不如把眼光放长远，让我们一起期待。

说起优化算法，那么我的问题是：

（1）什么是优化算法？

（2）优化算法作用是什么？？

（3）优化算法有哪些？

看到这里你的答案是什么？下面是我的答案：

1\. 什么是优化算法？

关于优化算法的内容在[MachineLN之三要素](http://mp.weixin.qq.com/s?__biz=MzU3MTM3MTIxOQ==&mid=2247483841&idx=2&sn=e4a3cff7b12c48af237c577c487ba3a1&chksm=fce07a7dcb97f36be5003c3018b3a391070bdc4e56839cb461d226113db4c5f24032e0bf5809&scene=21#wechat_redirect)已经提到过，就是三要素中的算法：就是那个使策略（损失函数）最小化的方法，或者说通过迭代的方法（优化算法）计算损失函数的最优解。

2. 优化算法作用是什么？

简单说优化算法作用就是，用来调节模型参数，进而使得损失函数最小化，（此时的模型参数就是那个最优解）目前优化算法求解损失函数的最优解一般都是迭代的方式。（可以看一下下面两个图不同的优化算法，在寻找最优解的过程，还有鞍点问题）

*   如果是凸优化问题，如果数据量特别大，那么计算梯度非常耗时，因此会选择使用迭代的方法求解，迭代每一步计算量小，且比较容易实现。

*   对于非凸问题，只能通过迭代的方法求解，每次迭代目标函数值不断变小，不断逼近最优解。

*   因此优化问题的重点是使用何种迭代方法进行迭代，即求迭代公式。

![优化函数.gif](http://upload-images.jianshu.io/upload_images/4618424-4964e87ba7b348b6.gif?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![优化函数2.gif](http://upload-images.jianshu.io/upload_images/4618424-fe6faab01ecaa745.gif?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


3\. 优化算法有哪些？

深度学习中常用的优化算法是梯度下降，着重介绍，一般来说，梯度下降算法有三种变种，它们之间的区别是使用多少数据来计算目标函数的梯度。取决于数据的大小，我们在参数更新的精度和花费的时间进行平衡。

*   Batch gradient descent

    批量梯度下降是一种对参数的update进行累积，然后批量更新的一种方式。这种算法很慢，并且对于大的数据集并不适用；并且使用这种算法，我们无法在线更新参数。

*   Stochastic gradient descent

    随机梯度下降是一种对参数随着样本训练，一个一个的及时update的方式。这种方法迅速并且能够在线使用；但是它频繁的更新参数会使得目标函数波动很大。

*   Mini-batch gradient descent

    吸取了BGD和SGD的优点，每次选择一小部分样本进行训练。

尽管Mini-batch gradient descent算法不错， 它也不能保证收敛到好的值，并且带来了新的挑战：

*   很难选择合适的学习率。过小的学习率会使得收敛非常缓慢；多大的学习率会发生震荡甚至发散。

*   学习率相关的参数只能提前设定，无法使用数据集的特性

*   所有的参数更新都使用相同的学习率

*   如何跳出局部最小

下面是梯度下降的一系列文章：

（1）SGD 随机梯度下降，下面提到的深度学习的优化算法都是基于此展开；

![image.png](http://upload-images.jianshu.io/upload_images/4618424-9db998b3ab6277bd.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


（2）带动量（momentum）的SGD

虽然随机梯度下降仍然是非常受欢迎的优化方法，但其学习过程有时会很慢。 动量方法旨在加速学习，特别是处理高曲率、小但一致的梯度，或是 带噪声的梯度。动量算法积累了之前梯度指数级衰减的移动平均，并且继续沿该方向移动。 

![image](http://upload-images.jianshu.io/upload_images/4618424-a7612a8a01f4251c?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

SGD方法的一个缺点是，其更新方向完全依赖于当前的batch，因而其更新十分不稳定。解决这一问题的一个简单的做法便是引入momentum。momentum即动量，它模拟的是物体运动时的惯性，即更新的时候在一定程度上保留之前更新的方向，同时利用当前batch的梯度微调最终的更新方向。这样一来，可以在一定程度上增加稳定性，从而学习地更快，并且还有一定摆脱局部最优的能力。 

带动量（Momentum）的SGD**特点：**

*   下降初期时，使用上一次参数更新，下降方向一致，乘上较大的能够进行很好的加速

*   下降中后期时，在局部最小值来回震荡的时候，，使得更新幅度增大，跳出陷阱

*   在梯度改变方向的时候，能够减少更新 总而言之，momentum项能够在相关方向加速SGD，抑制振荡，从而加快收敛

（3）Nesterov 动量  SGD

![image](http://upload-images.jianshu.io/upload_images/4618424-ddeba387fd60e368?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![image.png](http://upload-images.jianshu.io/upload_images/4618424-d744364b38138020.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


首先，按照原来的更新方向更新一步（棕色线），然后在该位置计算梯度值（红色线），然后用这个梯度值修正最终的更新方向（绿色线）。上图中描述了两步的更新示意图，其中蓝色线是标准momentum更新路径。其实，momentum项和nesterov项都是为了使梯度更新更加灵活，对不同情况有针对性。

但是，人工设置一些学习率总还是有些生硬，接下来介绍几种自适应学习率的方法 ：

（4）AdaGrad(AdaGrad)算法：在参数空间中更为平缓的倾斜方向会取得 更大的进步。 

![image](http://upload-images.jianshu.io/upload_images/4618424-4883eddf4d35f4be?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

说明一：大多数方法对于所有参数都使用了同一个更新速率。但是同一个更新速率不一定适合所有参数，比如有的参数可能已经到了仅需要微调的阶段，但又有些参数由于对应样本少等原因，还需要较大幅度的调动。Adagrad就是针对这一问题提出的，自适应地为各个参数分配不同学习率的算法。

说明二：AdaGrad是第一个自适应算法，通过每次除以根号下之前所有梯度值的平方和，从而使得步长单调递减。同时因为cache的变化与每个维度上的值有关，所以此方法可以解决各个维度梯度值相差较大的问题。﻿对于每个参数，随着其更新的总距离增多，其学习速率也随之变慢。 

**特点：**

*   前期较小的时候， regularizer较大，能够放大梯度

*   后期较大的时候，regularizer较小，能够约束梯度

*   适合处理稀疏梯度

**缺点：**

*   由算法8.4可以看出，仍依赖于人工设置一个全局学习率

*   设置过大的话，会使regularizer过于敏感，对梯度的调节太大

*   中后期，分母上梯度平方的累加将会越来越大，使，使得训练提前结束

（5）RMSProp 算法

![image.png](http://upload-images.jianshu.io/upload_images/4618424-eb19b279bd844472.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

简单的说，RMSProp对于AdaGrad的改进在于cache的更新方法。不同于不断的累加梯度的平方，RMSProp中引入了泄露机制，使得cache每次都损失一部分，从而使得步长不再是单调递减了。RMSprop可以算作Adadelta的一个特例 

**特点：**

*   其实RMSprop依然依赖于全局学习率

*   RMSprop算是Adagrad的一种发展，和Adadelta的变体，效果趋于二者之间

*   适合处理非平稳目标 - 对于RNN效果很好

RMSProp算法修改AdaGrad以在非凸设定下效果更好，改变梯度累积为**指数加权的移动平均**。 

AdaGrad旨在应用于凸问题时快速收敛。AdaGrad根据平方梯度的整个历史收缩学习率，可能使得学习率在达到这样的凸结构前就变得太小了。 

RMSprop使用指数衰减平均来丢弃遥远过去的历史，使其在找到凸结构后快速收敛，就像一个初始化于该碗装结构的AdaGrad算法实例。 相比于AdaGrad，使用移动平均引入了一个新的超参数ρ，用来控制移动平均的长度范围。 

（6）Nesterov 动量  RMSProp 

![image](http://upload-images.jianshu.io/upload_images/4618424-7968412f478a22b7?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

（7）Adam 算法：Adam被看作结合RMSProp和具有一些重要区别的动量的变种。 首先，Adam中动量直接并入梯度一阶矩（指数加权）的估计。 其次，Adam包括**偏置修正**，修泽和那个从原点初始化的一阶矩（动量项）和（非中心的）二阶矩的估计。RMSProp也采用了（非中心的）二阶矩估计，然而缺失了修正因子。因此，RMSProp二阶矩估计可能在训练初期有很高的偏置。Adam通常被认为对超参数的选择相当鲁棒，尽管学习率有时需要遵从建议的**默认参数0.001**。 （本人比较喜欢用adam，总之适合自己才是最好的）

![image](http://upload-images.jianshu.io/upload_images/4618424-e189292639e65e3f?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

Adam(Adaptive Moment Estimation)本质上是带有动量项的RMSprop，它利用梯度的一阶矩估计和二阶矩估计动态调整每个参数的学习率。Adam的优点主要在于经过偏置校正后，每一次迭代学习率都有个确定范围，使得参数比较平稳。 

**特点：**

*   结合了Adagrad善于处理稀疏梯度和RMSprop善于处理非平稳目标的优点

*   对内存需求较小

*   为不同的参数计算不同的自适应学习率

*   也适用于大多非凸优化 - 适用于大数据集和高维空间

看一下大神是如何总结的：

*   **如果你的数据特征是稀疏的，那么你最好使用自适应学习速率SGD优化方法(Adagrad、Adadelta、RMSprop与Adam)，因为你不需要在迭代过程中对学习速率进行人工调整。 **

*   **RMSprop是Adagrad的一种扩展，与Adadelta类似，但是改进版的Adadelta使用RMS去自动更新学习速率，并且不需要设置初始学习速率。而Adam是在RMSprop基础上使用动量与偏差修正。RMSprop、Adadelta与Adam在类似的情形下的表现差不多。****Kingma****指出收益于偏差修正，Adam略优于RMSprop，因为其在接近收敛时梯度变得更加稀疏。因此，Adam可能是目前最好的SGD优化方法。 **

*   **有趣的是，最近很多论文都是使用原始的SGD梯度下降算法，并且使用简单的学习速率退火调整（无动量项）。现有的已经表明：SGD能够收敛于最小值点，但是相对于其他的SGD，它可能花费的时间更长，并且依赖于鲁棒的初始值以及学习速率退火调整策略，并且容易陷入局部极小值点，甚至鞍点。因此，如果你在意收敛速度或者训练一个深度或者复杂的网络，你应该选择一个自适应学习速率的SGD优化方法。**

个人总结：适合自己的才是最好的！

除梯度下降外的一些优化方法：（感兴趣的自行研究吧，牛顿法、拟牛顿法、最小二乘、启发式优化方法（模拟退火方法、遗传算法、蚁群算法以及粒子群算法等））

（8）二阶近似方法 ：与一阶方法相比，二阶方法使用二阶导 数改进了优化。最广泛使用的二阶方法是牛顿法。

*   牛顿法

牛顿法是二阶收敛的，梯度下降是一阶收敛的，所以牛顿法更快，如果更通俗地说的话，比如你想找一条最短的路径走到一个盆地的最底部，梯度下降法每次只从你当前所处位置选一个坡度最大的方向走一步，牛顿法在选择方向时，不仅会考虑坡度是否够大，还会考虑你走了一步之后，坡度是否会变得更大。所以，可以说牛顿法比梯度下降法看得更远一点，能更快地走到最底部。（牛顿法目光更加长远，所以少走弯路；相对而言，梯度下降法只考虑了局部的最优，没有全局思想。）。 （牛顿法是基于泰勒级数展开的（是不是很傻逼，不知死活的我去参加腾讯美团的面试被问到过各种数学问题，自己一脸傻逼，学的高数早还回去了，被迫自己又买了高数、线代、概率论的书，放在书架上压阵），这里意识到数学的重要性了，BAT的面试算法很看重数学）

![image](http://upload-images.jianshu.io/upload_images/4618424-7df077fdd9df6290?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

根据wiki上的解释，从几何上说，牛顿法就是用一个二次曲面去拟合你当前所处位置的局部曲面，而梯度下降法是用一个平面去拟合当前的局部曲面，通常情况下，二次曲面的拟合会比平面更好，所以牛顿法选择的下降路径会更符合真实的最优下降路径。

![image](http://upload-images.jianshu.io/upload_images/4618424-c69c708ae163ef2b?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

注：红色的牛顿法的迭代路径，绿色的是梯度下降法的迭代路径。

优点：二阶收敛，收敛速度快；

缺点：牛顿法是一种迭代算法，每一步都需要求解目标函数的Hessian矩阵的逆矩阵，计算比较复杂。

*   共轭梯度方法

**共轭梯度法是介于梯度下降法与牛顿法之间的一个方法，它仅需利用一阶导数信息，但克服了梯度降法收敛慢的缺点，又避免了牛顿法需要存储和计算Hesse矩阵并求逆的缺点，共轭梯度法不仅是解决大型线性方程组最有用的方法之一，也是解大型非线性最优化最有效的算法之一。** 在各种优化算法中，共轭梯度法是非常重要的一种。其优点是所需存储量小，具有步收敛性，稳定性高，而且不需要任何外来参数。

![image.png](http://upload-images.jianshu.io/upload_images/4618424-f0b1867b60ede4ff.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


下图为共轭梯度法和梯度下降法搜索最优解的路径对比示意图：

![image](http://upload-images.jianshu.io/upload_images/4618424-32407f52e560ef59?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

注：绿色为梯度下降法，红色代表共轭梯度法

好了看了这么多，还是直接看大神的吧：http://ruder.io/optimizing-gradient-descent/index.html#gradientdescentvariants

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

![image](http://upload-images.jianshu.io/upload_images/4618424-7478749bf959d536?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


版权声明：本文为博主原创文章，未经博主允许不得转载。有问题可以加微信：lp9628(注明CSDN)。
