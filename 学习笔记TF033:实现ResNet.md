ResNet(Residual Neural Network)，微软研究院 Kaiming He等4名华人提出。通过Residual Unit训练152层深神经网络，ILSVRC 2015比赛冠军，3.57% top-5错误率，参数量比VGGNet低，效果非常突出。ResNet结构，极快加速超深神经网络训练，模型准确率非常大提升。Inception V4，Inception Module、ResNet结合。ResNet推广性好。

瑞十教授Schmidhuber(LSTM网络发明者，1997年)提出Highway Network。解决极深神经网络难训练问题。修改每层激活函数，此前激活函数只是对输入非线性变换y=H(x,WH)，Highway NetWork保留一定比例原始输入x，y=H(x,WH)·T(x,WT)+x·C(x,WC)，T变换系数，C保留系数，令C=1-T。前面一层信息，一定比例不经过矩阵乘法和非线性变换，直接传输下一层。Highway Network，gating units学习控制网络信息流，学习原始信息应保留比例。gating机制，Schmidhuber教授早年LSTM循环神经网络gating。几百上千层深Highway Network，直接梯度下降算法训练，配合多种非线性激活函数，学习极深神经网络。Highway Network允许训练任意深度网络，优化方法与网络深度独立。

ResNet 允许原始输入信息直接传输到后层。Degradation问题，不断加深神经网络深度，准确率先上升达到饱和，再下降。ResNet灵感，用全等映射直接将前层输出传到后层。神经网络输入x，期望输出H(x)，输入x直接传到输出作初始结果，学习目标F(x)=H(x)-x。ResNet残差学习单元(Residual Unit)，不再学习完整输出H(x)，只学习输出输入差别H(x)-x，残差。

ResNet，很多旁路支线，输入直接连到后层，后层直接学习残差，shortcut或connections。直接将输入信息绕道传到输出，保护信息完整性，整个网络只学习输入、输出差别，简化学习目标、难度。

两层残新式学习单元包含两个相同输出通道数3x3卷积。三层残差网络用Network In Network和Inception Net 1x1卷积。在中间3x3卷积前后都用1x1卷积，先降维再升维。如果输入输出维度不同，对输入x线性映射变换维度，再接后层。

    layername outputsize 18-layer    34-layer  50-layer   101-layer  152-layer
    conv1      112x112                      7x7,64,stride 2
    conv2_x     56x56                    3x3 max pool,stride 2
                         3x3,64x2    3x3,64x3  1x1,64x3   1x1,64x3   1x1,64x3
                         3x3,64      3x3,64    3x3,64     3x3,64     3x3,64
                                               1x1,256    1x1,256    1x1,256
    conv3_x     28x28    3x3,128x2  3x3,128x4  1x1,128x4  1x1,128x4  1x1,128x8
                         3x3,128    3x3,128    3x3,128    3x3,128    3x3,128
                                               1x1,512    1x1,512    1x1,512
    conv4_x     14x14    3x3,256x2  3x3,256x6  1x1,256x6  1x1,256x23 1x1,256x36
                         3x3,256    3x3,256    3x3,256    3x3,256    3x3,256
                                               1x1,1024   1x1,1024   1x1,1024
    conv5_x      7x7     3x3,512x2  3x3,512x3  1x1,512x3  1x1,512x3  1x1,512x3
                         3x3,512    3x3,512    3x3,512    3x3,512    3x3,512
                                               1x1,2048   1x1,2048   1x1,2048
                 1x1                 average pool,1000-d fc,softmax
    FLOPs                1.8x10^9   3.6x10^9   3.8x10^9   7.6x10^9   11.3x10^9

ResNet结构，消除层数不断加深训练集误差增大现象。ResNet网络训练误差随层数增大逐渐减小，测试集表现变好。Google借鉴ResNet，提出Inception V4和Inception-ResNet-V2，ILSVRC错误率3.08%。《Identyty Mappings in Deep Residual Networks》提出ResNet V2。ResNet残差学习单元传播公式，前馈信息和反馈信号可直接传输。skip connection 非线性激活函数，替换Identity Mappings(y=x)。ResNet每层都用Batch Normalization。

Schmidhuber教授，ResNet，没有gates LSTM网络，输入x传递到后层过程一直发生。ResNet等价RNN，ResNet类似多层网络间集成方法(ensemble)。

《The Power of Depth for Feedforward Neural Networks》，理论证明加深网络比加宽网络更有效。

Tensorflow实现ResNet。contrib.slim库，原生collections。collections.namedtuple设计ResNet基本Block模块组named tuple，创建Block类，只有数据结构，没有具体方法。典型Block，三个参数，scope、unit_fn、args。
Block('block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)])，block1是Block名称(或scope)，bottleneck是ResNet V2残差学习单元。最后参数是Block args，args是列表，每个元素对应bottleneck残差学习单元。前面两个元素(256, 64, 1),第三元素(256, 64, 2)，每个元素都是三元tuple(depth,depth_bottleneck,stride)。(256, 64, 3)代表bottleneck残差学习单元(三个卷积层)，第三层输出通道数depth 256，前两层输出通道数depth_bottleneck 64，中间层步长stride 3。残差学习单元结构[(1x1/s1,64),(3x3/s3,64),(1x1/s1,256)]。

降采样subsample方法，参数inputs(输入)、factor(采样因子)、scope。fator1,不做修改直接返回inputsx，不为1,用slim.max_pool2d最大池化实现。1x1池化尺寸，stride步长，实现降采样。

定义conv2d_same函数创建卷积层，如果stride为1，用slim.conv2d，padding模式SAME。stride不为1,显式pad zero。pad zero总数kernel_size-1 pad_beg为pad//2,pad_end为余下部分。tf.pad补零输入变量。已经zero padding，只需padding模式VALID的slim.conv2d创建此卷积层。

定义堆叠Blocks函数，参数net输入，blocks是Block class 列表。outputs_collections收集各end_points collections。两层循环，逐个Block，逐个Residual Unit堆叠。用两个tf.variable_scope命名残差学习单元block/unit_1形式。第2层循环，每个Block每个Residual Unit args，展开depth、depth_bottleneck、stride。unit_fn残差学习单元生成函数，顺序创建连接所有残差学习单元。slim.utils.collect_named_outputs函数，输出net添加到collection。所有Block所有Residual Unit堆叠完，返回最后net作stack_blocks_dense函数结果。

创建ResNet通用arg_scope，定义函数参数默认值。定义训练标记is_training默认True，权重衰减速度weight_decay默认0.001。BN衰减速率默认0.997,BN epsilon默认1e-5,BN scale默认True。先设置好BN各项参数，通过slim.arg_scope设置slim.conv2d默认参数，权重正则器设L2正则，权重初始化器设slim.variance_scaling_initializer()，激活函数设ReLU，标准化器设BN。最大池化padding模式默认设SAME(论文中用VALID)，特征对齐更简单。多层嵌套arg_scope作结果返回。

定义核心bottleneck残差学习单元。ResNet V2论文Full Preactivation Residual Unit 变种。每层前都用Batch Normalization，输入preactivation，不在卷积进行激活函数处理。参数，inputs输入，depth、depth_bottleneck、stride，outputs_collections收集end_points collection，scope是unit名称。用slim.utils.last_dimension函数获取输入最后维度输出通道数，参数min_rank=4限定最少4个维度。slim.batch_norm 输入 Batch Normalization，用ReLU函数预激活Preactivate。

定义shorcut，直连x，如果残差单元输入通道数depth_in、输出通道数depth一致，用subsample，步长stride,inputs空间降采样，确保空间尺寸和残差一致，残差中间层卷积步长stride；如果不一致，用步长stride 1x1卷积改变通道数，变一致。

定义residual(残差)，3层，1x1尺寸、步长1､输出通道数depth_bottleneck卷积，3x3尺寸、步长stride､输出通道数depth_bottleneck卷积，1x1尺寸、步长1､输出通道数depth卷积，得最终residual，最后层没有正则项没有激活函数。residual、shorcut相加，得最后结果output，用slim.utils.collect_named_outputs，结果添加collection，返回output函数结果。

定义生成ResNet V2主函数。参数，inputs输入，blocks为Block类列表，num_classes最后输出类数，global_pool标志是否加最后一层全局平均池化，include_root_block标志是否加ResNet网络最前面7x7卷积、最大池化，reuse标志是否重用，scope整个网络名称。定义variable_scope、end_points_collection，通过slim.arg_scope设slim.con2d、bottleneck、stack_block_dense函数的参数outputs_collections默认end_points_colletion。根据include_root_block标记，创建ResNet最前面64输出通道步长2的7x7卷积，接步长2的3x3最大池化。两个步长2层，图片尺寸缩小为1/4。用stack_blocks_dense生成残差学习模块组，根据标记添加全局平均池化层，用tf.reduce_mean实现全局平均池化，效率比直接avg_pool高。根据是否有分类数，添加输出通道num_classes1x1卷积(无激活函数无正则项)，添加Softmax层输出网络结果。用slim.utils.convert_to_dict 转化collection为Python dict。最后返回net、end_points。

50层ResNet，4个残差学习Blocks，units数量为3､4､6､3，总层数(3+4+6+3)x3+2=50。残差学习模块前，卷积、池化把尺寸缩小4倍，前3个Blocks包含步长2层，总尺寸缩小4x8=32倍。输入图片尺寸最后变224/32=7。ResNet不断用步长2层缩减尺寸，输出通道数持续增加，达到2048。

152层ResNet，第二Block units数8，第三Block units数36。

200层ResNet，第二Block units数23，第三Block units数36。

评测函数time_tensorflow_run测试152层ResNet forward性能。图片尺寸224x224，batch size 32。is_training FLAG设False。resnet_v2_152创建网络，time_tensorflow_run评测forward性能。耗时增加50%，实用卷积神经网络结构，支持超深网络训练，实际工业应用forward性能不差。

![DingTalk20170728013255.png](http://upload-images.jianshu.io/upload_images/80690-e10d1b5d35beac00.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
![DingTalk20170728013351.png](http://upload-images.jianshu.io/upload_images/80690-5d56ebe12ed5d6e5.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
![DingTalk20170728013444.png](http://upload-images.jianshu.io/upload_images/80690-e6adc37493c66038.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
![DingTalk20170728013535.png](http://upload-images.jianshu.io/upload_images/80690-49486f59eb819702.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
![DingTalk20170728013610.png](http://upload-images.jianshu.io/upload_images/80690-cfad470b08a2ef24.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
![DingTalk20170728013712.png](http://upload-images.jianshu.io/upload_images/80690-2a04c31ddb6265b0.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

参考资料：
《TensorFlow实战》


