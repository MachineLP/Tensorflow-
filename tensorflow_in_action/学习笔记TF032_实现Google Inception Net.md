Google Inception Net，ILSVRC 2014比赛第一名。控制计算量、参数量，分类性能非常好。V1，top-5错误率6.67%，22层，15亿次浮点运算，500万参数(AlexNet 6000万)。V1降低参数量目的，参数越多模型越庞大，需数据量越大，高质量数据昂贵；参数越多，耗费计算资源越大。模型层数更深，表达能力更强，去除最后全连接层，用全局平均池化层(图片尺寸变1x1)，参数大减，模型训练更快，减轻过拟合(《Network in Network》论文)，Inception Module提高参数利用效率，大网络中小网络。增加分支网络，NIN级联卷积层、NLPConv层。一般，卷积层增加输出通道数，提升表达能力，计算量增大、过拟合，每个输出通道对应一个滤波器，同一滤波器共享参数，只能提取一类特征。NIN，输出通道组保信息。MLPConv，普通卷积层，接1x1卷积、ReLU激活函数。

Inception Module结构，4个分支。第一分支，输入1x1卷积。1x1卷积，跨通道组织信息，提高网络表达能力，输出通道升维、降维。4个分支都用1x1卷积，低成本跨通道特征变换。第二分支，1x1卷积，3x3卷积，两次特征变换。第三分支，1x1卷积，5x5卷积。第四分支，3x3最大池化，1x1卷积。1x1卷积性价比高，小计算量，特征变换、非线性化。4个分支后聚合操作合并(输出通道数聚合)。Inception Module 包含3种不同尺寸卷积、1个最大池化，增加不同尺度适应性。网络深度、宽度高效扩充，提升准确率，不过拟合。

Inception Net，找到最优稀疏结构单元(Inception Module)。Hebbian原理，神经反射活动持续、重复，神经元连接稳定性持久提升，两个神经元细胞距离近，参与对方重复、持续兴奋，代谢变化成为使对方兴奋细胞。一起发射神经元会连在一起(Cells that fire together,wire together)，学习过程刺激使神经元间突触强度增加。《Provable Bounds for Learning Some Deep Representations》，很大很稀疏神经网络表达数据集概率分布，网络最佳构筑方法是逐层构筑。上层高度相关(correlated)节点聚类，每个小簇(cluster)连接一起。相关性高节点连接一起。

图片数据，临近区域数据相关性高，相邻像素点卷积连接一起。多个卷积核，同一空间位置，不同通道卷积核输出结果，相关性极高。稍大一点卷积(3x3､5x5)，连接节点相关性高，适当用大尺寸卷积，增加多样性(diversity)。Inception Module 4分支，不同尺寸(1x1、3x3、5x5)小型卷积，连接相关性很高节点。

Inception Module，1x1卷积比例(输出通道数占比)最高，3x3､5x5卷积稍低。整个网络，多个Inception Module堆叠。靠后Inception Module卷积空间集中度渐降低，捕获更大面积特征，捕捉更高阶抽象特征。靠后Inception Module，3x3､5x5大面积卷积核占比(输出通道数)更多。

Inception Net 22层，最后一层输出，中间节点分类效果好。使用辅助分类节点(auxiliary classifiers)，中间层输出作分类，按较小权重(0.3)加到最终分类结果。相当模型融合，给网络增加反向传播梯度信号，提供额外正则化。

Google Inception Net家族：2014年9月《Going Deeper with Convolutions》Inception V1，top-5错误率6.67%。2015年2月《Batch Normalization:Accelerating Deep Network Trainign by Reducing Internal Covariate》Inception V2，top-5错误率4.8%。2015年12月《Rethinking the Inception Architecture ofr Computer Vision》Inception V3，top-5错误率3.5%。2016年2月《Inception-v4,Inception-ResNet and the Impact of Residual Connections on Learning》Inception V4，top-5错误率3.08%。

Inception V2，用两个3x3卷积代替5x5大卷积，降低参数量，减轻过拟合，提出Batch Normalization方法。BN，非常有效正则化方法，让大型卷积网络训练速度加快很多倍，收敛后分类准确率大幅提高。BN 对每个mini-batch数据内部标准化(normalization)处理，输出规范化到N(0,1)正态分布，减少Internal Covariate Shift(内部神经元分布改变)。传统深度神经网络，每层输入分布变化，只能用很小学习速率。每层BN 学习速率增大很多倍，迭代次数只需原来的1/14，训练时间缩短。BN正则化作用，减少或者取消Dropout，简化网络结构。

增大学习速率，加快学习衰减速度，适用BN规范化数据，去除Dropout，减轻L2正则，去除LRN，更彻底shuffle训练样本，减少数据增强过程数据光学畸变(BN训练更快，样本被训练次数更少，更真实样本对训练有帮助)。

Inception V3，引入Factorization into small convolutions思想，较大二维卷积拆成两个较小一维卷积，节约大量参数，加速运算，减轻过拟合，增加一层蜚线性，扩展模型表达能力。非对称卷积结构拆分，比对称拆分相同小卷积核效果更明显，处理更多、更丰富空间特征，增加特征多样性。

优化Inception Module结构，35x35，17x17，8x8。分支中使用分支，8x8结构，Network In Network In Network。V3结合微软ResNet。

使用tf.contrib.slim辅助设计42层Inception V3 网络。

                    Inception V3 网络结构
    类型             kernel尺寸/步长(或注释)   输入尺寸
    卷积                    3x3/2           299x299x3
    卷积                    3x3/1           149x149x32
    卷积                    3x3/1           147x147x32
    池化                    3x3/2           147x147x64
    卷积                    3x3/1           73x73x64
    卷积                    3x3/2           71x71x80
    卷积                    3x3/1           35x35x192
    Inception模块组  3个InceptionModule      35x35x288
    Inception模块组  5个InceptionModule      17x17x768
    Inception模块组  3个InceptionModule      8x8x1280
    池化                     8x8            8x8x2048
    线性                   logits           1x1x2048
    Softmax               分类输出           1x1x1000

定义简单函数trunc_normal，产生截断正态分布。

定义函数inception_v3_arg_scope，生成网络常用函数默认参数，卷积激活函数、权重初始化方式、标准化器。设置L2正则weight_decay默认值0.00004，标准差stddev默认值0.1，参数batch_norm_var_collection默认值moving_vars 。

定义batch normalization参数字典，定义衰减系数decay 0.997，epsilon 0.001，updates_collections为tf.GraphKeys.UPADTE_OPS，字典variables_collections中beta、gamma设None，moving_mean、moving_variance设batch_norm_var_collection。

slim.agr_scope，函数参数自动赋默认值。with slim.arg_scope([slim.conv2d, slim.fully_connected], weights_regularizer=slim.l2_regularizer(weight_decay)) ，对[slim.conv2d, slim.fully_connected]两个函数参数自动赋值，参数weights_regularizer值默认设为slim.l2_regularizer(weight_decay)。不需要每次重复设置参数，只需要修改时设置。

嵌套一个slim.arg_scope，卷积层生成函数slim.conv2d参数赋默认值，权重初始化器weights_initializer设trunc_normal(stddev)，激活函数设ReLU，标准化器设slim.batch_norm，标准化器参数设batch_norm_params，返回定义好的scope。

定义函数inception_v3_base，生成Inception V3网络卷积。参数inputs 输入图片数据tensor，scope 函数默认参数环境。定义字典表end_points ，保存关键节点。slim.agr_scope，设置slim.conv2d、slim.max_pool2d、slim_avg_pool2d函数参数默认值，stride设1，padding设VALID。非Inception Module卷积层，slim.conv2d创建卷积层，第一参数输入tensor，第二参数输出通道数，第三参数卷积核尺寸，第四参数步长stride ,第五参数padding模式。第一卷积层输出通道数32,卷积核尺寸3x3,步长 2,padding模式VALID。

非Inception Module卷积层，主要用3x3小卷积核。Factorization into small convolutions思想， 用两个1维卷积模拟大尺寸2维卷积，减少参数量，增加非线性。1x1卷积，低成本跨通道特征组合。第一卷积层步长2,其余卷积层步长1。池化层尺寸3x3､步长2重叠最大池化。网络输入数据惊寸299x299x3,经过3个步长2层，尺寸缩小为35x35x192,空间尺寸大降低，输出通道增加很多。一共5个卷积层，2个池化层，实现输入图片数据尺寸压缩，抽象图片特征。

三个连续Inception模块组。

第1个Inception模块组3个结构类似Inception Module。

第1 Inception模块组第1个Inception Module，名称Mixed_5b。slim.arg_scope设置所有Inception模块组默认参数，所有卷积层、最大池化、平均池化层步长设1,padding模式设SAME。设置Inception Module variable_scope名称Mixed_5b。4个分支，Branch_0到Branch_3。第一分支64输出通道1x1卷积。第二分支48输出通道1x1卷积，连接64输出通道5x5卷积。第三分支64输出通道1x1卷积，连接2个96输出通道3x3卷积。第四分支3x3平均池化，连接32输出通道1x1卷积。最后tf.concat合并4分支输出(第三维度输出通道合并)，生成Inception Module最终输出。所有层步长为1，padding模型SAME，图片尺寸不缩小，维持35x35，通道数增加，4个分支通道数和64+64+96+32=256，最终输出tensor尺寸35x35x256。

第1 Inception模块组第2个Inception Module，名称Mixed_5c。步长1，padding模型SAME。4个分支，第四分支最后接64输出通道1x1卷积。输出tensor尺寸35x35x288。

第1 Inception模块组第3个Inception Module，名称Mixed_5d。输出tensor尺寸35x35x288。

第2个Inception模块组5个Inception Module。第2到第5Inception Module结构类似。

第2 Inception模块组第1个Inception Module，名称Mixed_6a。3个分支。第一分支384输出通道3x3卷积，步长2，padding模式VAILD，图片尺寸压缩为17x17。第二分支3层，64输出通道1x1卷积，两个96输出通道3x3卷积，最后一层步长2，padding模式VAILD，分支输出tensor尺寸17x17x96。第三分支3x3最大池化层，步长2，padding模式VAILD，分支输出tensor尺寸17x17x256。三分支输出通道合并，最终输出尺寸17x17x(384+96+256)=17x17x768。第2 Inception模块组5个Inception Module尺寸相同。

第2 Inception模块组第2个Inception Module，名称Mixed_6b。4个分支。第一分支192输出通道1x1卷积。第二分支3层，第一层128输出通道1x1卷积，第二层128输出通道1x7卷积，第三层192输出通道7x1卷积。Factorization into small convolutions思想，串联1x7卷积和7x1卷积，相当合成7x7卷积，参数量大减，减轻过拟合，增加一个激活函数，增强非线性特征变换。第三分支5层，第一层128输出通道1x1卷积，第二层128输出通道7x1卷积，第三层128输出通道1x7卷积，第四层128输出通道7x1卷积，第五层192输出通道1x7卷积。Factorization into small convolutions典范，反复拆分7x7卷积。第四分支3x3平均池化层，连接192输出通道1x1卷积。四分支合并，最终输出tensor尺寸17x17x(192+192+192+192+192)=17x17x768。

第2 Inception模块组第3个Inception Module，名称Mixed_6c。第二分支和第三分支前几个卷积层输出通道数从128变为160，最终输出通道数还是192。网络每经过一个Inception Module，即使输出尺寸不变，特征被重新精炼一遍，丰富卷积和非线性化，提升网络性能。

第2 Inception模块组第4个Inception Module，名称Mixed_6d。

第2 Inception模块组第5个Inception Module，名称Mixed_6e。Mixed_6e存储end_points，作Auxiliary Classifier输助模型分类。

第3个Inception模块组3个Inception Module。第2到第3Inception Module结构类似。

第3 Inception模块组第1个Inception Module，名称Mixed_7a。3个分支。第一分支2层，192输出通道1x1卷积，连接320输出通道3x3卷积，步长2，padding模式VAILD，图片尺寸压缩为8x8。第二分支4层，192输出通道1x1卷积，192输出通道1x7卷积，192输出通道7x1卷积，192输出通道3x3卷积，最后一层步长2，padding模式VAILD，分支输出tensor尺寸8x8x192。第三分支3x3最大池化层，步长2，padding模式VAILD，池化层不改变输出通道，分支输出tensor尺寸8x8x768。三分支输出通道合并，最终输出尺寸8x8x(320+192+768)=8x8x1280。从这个Inception Module开始，输出图片尺寸缩小，通道数增加，tensor 总size下降。

第3 Inception模块组第2个Inception Module，名称Mixed_7b。4个分支。第一分支320输出通道1x1卷积。第二分支，第一层384输出通道1x1卷积，第二层2个分支，384输出通道1x3卷积和384输出通道3x1卷积，用tf.concat合并两个分支，得到输出tensor尺寸8x8x(384+384)=8x8x768。第三分支，第一层448输出通道1x1卷积，第二层384输出通道3x3卷积，第三层2个分支，384输出通道1x3卷积和384输出通道3x1卷积，合并得到8x8x768输出tensor。第四分支3x3平均池化层，连接192输出通道1x1卷积。四分支合并，最终输出tensor尺寸8x8x(320+768+768+192)=8x8x2048。这个Inception Module，输出通道数从1280增加到2048。

第3 Inception模块组第3个Inception Module，名称Mixed_7c。返回这个Inception Module结果，作inception_v3_base函数最终输出。

Inception V3网络结构，首先5个卷积层和2个池化层交替普通结构，3个Inception模块组，每个模块组内包含多个结构类似Inception Module。设计Inception Net重要原则，图片尺寸不断缩小，从299x299通过5个步长2卷积层或池化层，缩小8x8，输出通道数持续增加，从开始3(RGB三色)到2048。每一层卷积、池化或Inception模块组，空间结构简化，空间信息转化高阶抽象特征信息，空间维度转为通道维度。每层输出tensor总size持续下降，降低计算量。Inception Module规律，一般4个分支，第1分支1x1卷积，第2分支1x1卷积再接分解后(factorized)1xn和nx1卷积，第3分支和第2分支类似，更深，第4分支最大池化或平均池化。Inception Module，通过组合简单特征抽象(分支1)、比较复杂特征抽象(分支2､分支3)、一个简化结构池化层(分支4)，4种不同程度特征抽象和变换来有选择保留不同层高阶特征，最大程度丰富网络表达能力。

全局平均池化、Softmax、Auxiliary Logits。函数inception_v3输入参数，num_classes最后需要分类数量，默认1000ILSVRC比赛数据集种类数，is_training标志是否训练过程，训练时Batch Normalization、Dropout才会被启用，dropout_keep_prob训练时Dropoutr所需保留节点比例，默认0.8。prediction_fn分类函数，默认使用slim.softmax。spatial_squeeze参数标志输出是否进行squeeze操作(去除维数1维度)。reuse标志网络和Variable是否重用。scope包含函数默认参数环境，用tf.variable_scope定义网络name、reuse参数默认值，用slim.arg_scope定义Batch Normalization和Dropout的is_trainin标志默认值。用incepiton_v3_base构筑整个网络卷积，拿到最后一层输出net和重要节点字典表end_points。

Auxiliary Logits，辅助分类节点，帮助预测分类结果。用slim.arg_scope 卷积、最大池化、平均池化设默认步长1，默认padding模式SAME。通过end_points取Mixed_6e，再接5x5平均池化，步长3，padding设VALID，输出尺寸17x17x768变5x5x768。接128输出通道1x1卷积和768输出通道5x5卷积。权重初始化方式重设标准差0.01正态分布，padding模式VALID，输出尺寸变1x1x768。输出变1x1x1000。用tf.squeeze函数消除输出tensor前两个1维度。最后输助分类节点输出aux_logits储存到字典表end_points。

分类预测逻辑。Mixed_7e最后卷积层输出8x8全局平均池化，padding模式VALID，输出tensor尺寸变1x1x2048。接Dropout层，节点保留率dropout_keep_prob。连接输出通道数1000的1x1卷积，激活函数、规范化函数设空。tf.squeeze去除输出tensor维数1维度，接Softmax分类预测结果。最后返回输出结果logits、包含输助节点end_points。

Inception V3 网络构建完成。超参数选择，包括层数、卷积核尺寸、池化位置、步长大小、factorization使用时机、分支设计，需要大量探索和实践。

Inception V3运算性能测试。网络结构大，令batch_size 32。图片尺寸299x299，用tf.random_uniform生成随机图片数据 input。用slim.arg_scope加载inception_v3_arg_scope()，scope包含Batch Normalization默认参数，激活函数和参数初始化方式默认值。在arg_scope，调inception_v3函数，传入inputs，获取logits和end_points。创建Session，初始化全部模型参数。设置测试batch数量100，用time_tensorflow_run测试Inception V3网络forward性能。

Inception V3网络，图片面积比VGGNet 224x224大78%，forward速度比VGGNet快。2500万参数，比Inception V1的700万多，不到AlexNet的6000万的一半，比VGGNet的1.4亿少很多。42层，整个网络浮点计算量仅50亿次，比Inception V1的15亿次多，比VGGNet少。可以移植到普通服务器提供快速响应服务，或移植到手机实时图像识别。

Inception V3 backward性能测试，将整个网络所有参数加入参数列表，测试对全部参数求导所需时间，或直接下载ImageNet数据集，使用真实样本训练并评测所需时间。

Inception V3，Factorization into small convolutions很有效，可以降低参数量、减轻过拟合，增加网络非线性表达能力。卷积网络从输入到输出，图片尺寸逐渐缩小，输出通道数逐渐增加，空间结构简化，空间信息转化为高阶抽象特征信息。Inception Module多个分支提取不同抽象程度高阶特征很有效，丰富网络表达能力。


![DingTalk20170727060324.png](http://upload-images.jianshu.io/upload_images/80690-4a8b35bb3e11a19f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![DingTalk20170727060442.png](http://upload-images.jianshu.io/upload_images/80690-f6c52959f204264a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
![DingTalk20170727060524.png](http://upload-images.jianshu.io/upload_images/80690-4cc8b223d1f229a7.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
![DingTalk20170727060737.png](http://upload-images.jianshu.io/upload_images/80690-920b68c41032be5d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
![DingTalk20170727060822.png](http://upload-images.jianshu.io/upload_images/80690-c47a35362071bb9e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
![DingTalk20170727060857.png](http://upload-images.jianshu.io/upload_images/80690-7b6985068e965143.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
![DingTalk20170727061119.png](http://upload-images.jianshu.io/upload_images/80690-06829046f2dfeec0.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
![](http://upload-images.jianshu.io/upload_images/80690-e70555d528015a0f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
![DingTalk20170727061257.png](http://upload-images.jianshu.io/upload_images/80690-0bb7fac98663450f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

参考资料：
《TensorFlow实战》

