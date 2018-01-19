系统架构。
自底向上，设备层、网络层、数据操作层、图计算层、API层、应用层。核心层，设备层、网络层、数据操作层、图计算层。最下层是网络通信层和设备管理层。
网络通信层包括gRPC(google Remote Procedure Call Protocol)和远程直接数据存取(Remote Direct Memory Access,RDMA)，分布式计算需要。设备管理层包手包括TensorFlow分别在CPU、GPU、FPGA等设备上的实现。对上层提供统一接口，上层只需处理卷积等逻辑，不需要关心硬件上卷积实现过程。
数据操作层包括卷积函数、激活函数等操作。
图计算层包括本地计算图和分布式计算图实现(图创建、编译、优化、执行)。

应用层：训练相关类库、预测相关类库
API层：Python客户端、C++客户端、Java客户端、Go客户端，TensorFlow核心API
图计算层：分布式计算图、本地计算图
数据操作层：Const、Var、Matmul、Conv2D、Relu、Queue
网络层：gPRC、RDMA
设备层：CPU、GPU

设计理念。
图定义、图运行完全分开。符号主义。命令式编程(imperative style programming)，按照编写逻辑顺序执行，易于理解调试。符号式编程(symbolic style programming)，嵌入、优化，不易理解调试，运行速度快。Torch命令式，Caffe、MXNet混合，TensorFlow完全符号式。符号式计算，先定义各种变量，建立数据流图，规定变量计算关系，编译数据流图，这时还只是空壳，只有把数据输入，模型才能形成数据流，才有输出值。
TensorFlow运算在数据流图中，图运行只发生在会话(session)中。开启会话，数据填充节点，运算。关闭会话，无法计算。会话提供操作运行和Tensor求值环境。

    inport tensorflow as tf
    #创建图
    a = tf.constant([1.0,2.0])
    b = tf.constant([3.0,4.0])
    c = a * b
    #计算c
    print sess.run(c)#进行矩阵乘法，输出[3.,8.]
    sess.close()

编程模型。
TensorFlow用数据流图做计算。创建数据流图(网络结构图)。TensorFlow运行原理，图中包含输入(input)、塑形(reshape)、Relu层(Relu layer)、Logit层(Logit layer)、Softmax、交叉熵(cross entropy)、梯度(gradient)、SGD训练(SGD Trainer)，简单回归模型。
计算过程，从输入开始，经过塑形，一层一层前向传播运算。Relu层(隐藏层)有两个参数,Wh1､bh1,输出前用ReLu(Rectified Linear Units)激活函数做非线性处理。进入Logit层(输出层)，学习两个参数Wsm、bsm。用Softmax计算输出结果各个类别概率分布。用交叉熵度量源样本概率分布和输出结果概率分布之间相似性。计算梯度，需要参数Wh1、bh1、Wsm、bsm、交叉熵结果。SGD训练，反向传播，从上往下计算每层参数，依次更新。计算更新顺序，bsm、Wsm、bh1、Wh1。
TensorFlow，张量流动。TensorFlow数据流图由节点(node)、边(edge)组成有向无环图(directed acycline graph,DAG)。TensorFlwo由Tensor和Flow两部分组成。Tensor(张量)，数据流图的边。Flow(流动)，数据流图节点操作。
SGD训练：
     Wh1          bh1            Wsm         bsm
更新Wh1 更新bh1 更新Wsm 更新bsm
learning)rat=[0.01]
Gradients
交叉熵
classes=[10] 类标记 Softmax
Logit层：
bsm BiasAdd
Wsm MatMul
Relu层:
ReLU
bh1 Bias Add
Wh1 MatMul
塑形shape=[784,1]
输入

边。数据依赖、控制依赖。实线边表示数据依赖，代表数据，张量(任意维度的数据)。机器学习算法，张量在数据流图从前往后流动，前向传播(forword propagation)。残差(实际观察值与训练估计值的差)，从后向前流动，反向传播(backword propagation)。虚线边表示控制依赖(control dependency)，控制操作运行，确保happens-before关系，边上没有数据流过，源节点必须在目的节点开始执行前完成执行。
TensorFlow张量数据属性：
数据类型             Python类型  描述
DT_FLOAT          tf.float32       32位浮点型
DT_DOUBLE       tf.float64       64位浮点型
DT_INT64            tf.int64          64位有符号整型
DT_INT32            tf.int32          32位有符号整型
DT_INT16            tf.int16          16位有符号整型
DT_INT8              tf.int8             8位有符号整型
DT_UINT8            tf.uint8          8位无符号整型
DT_STRING         tf.tring           要变长度字节数组，每一张量元素是一字节数组
DT_BOOL            tf.bool            布尔型
DT_COMPLEX64 tf.complex64 两个32位浮点数组成复数，实部、虚部
DT_QINT32          tf.qint32         量化操作32位有符号整型，信号连续取值或大量可能离散取值，近似为有限多个或较少离散值
DT_QINT8            tf.qint8           量化操作8位有符号整型
DT_QUINT8         tf.quint8         量化操作8位无符号整型
图和张量实现源代码：tensorflow-1.1.0/tensorflow/python/framework/ops.py

节点。算子。代表一个操作(operation,OP)。表示数学运算，也可以表示数据输入(feed in)起点和输出(push out)终点，或者读取、写入持久变量(persistent variable)终点。
操作相关代码位于: tensorflow-1.1.0/tensorflow/python/ops/
TensoFlow实现算子(操作)：
类别                      示例
数学运算操作        Add、Sub、Mul、Div、Exp、Log、Greater、Less、Equal……
tensorflow-1.1.0/tensorflow/python/ops/math_ops.py，每个函数调用gen_math_ops.py，位于Python库stite-packages/tensorflow/python/ops/gen_math_ops.py ,又调用tensorflow-1.1.0/tensorflow/core/kernels/下核函数实现
数组运算操作        Concat、Slice、Split、Constant、Rank、Shape、Shuffle……
tensorflow-1.1.0/tensorflow/python/ops/array_ops.py，每个函数调用gen_array_ops.py，位于Python库stite-packages/tensorflow/python/ops/gen_array_ops.py ,又调用tensorflow-1.1.0/tensorflow/core/kernels/下核函数实现
矩阵运算操作        MatMul、MatrixInverse、MatrixDeterminant……
有状态操作            Variable、Assign、AssignAdd……
神经网络构建操作 SoftMax、Sigmoid、ReLU、Convolution2D、MaxPool……
检查点操作            Save、Restore
队列和同步操作     Enqueue、Dequeue、MutexAcquire、MutexRelease……
控制张量流动操作  Merge、Switch、Enter、Leave、NextIteration

图。操作任务描述成有向无环图。创建各个节点。

    import tensorflow as tf
    #创建一个常量运算操作，产生一个1x2矩阵
    matrix1 = tf.constant([[3.,3.]])
    #创建另外一个常量运算操作，产生一个2x1矩阵
    matrix2 = tf.constant([[2.],[2.]])
    #创建一个矩阵乘法运算，把matrix1和matrix2作为输入
    #返回值product代表矩阵乘法结果
    product = tf.matmul(matrix2,matrix2)

会话。启动图第一步创建一个Session对象。会话(session)提供图执行操作方法。建立会话，生成一张空图，会话添加节点和边，形成图，执行。tf.Session类创建并运行操作。

    with tf.Session as sess:
        result = sess.run([product])
        print result
调用Session对象run()方法执行图，传入Tensor，填充(feed)。返回结果类型根据输入类型而定，取回(fetch)。
会话是图交互桥梁，一个会话可以有多个图，会话可以修改图结构，可以往图流入数据计算。会话两个API:Extend(图添加节点、边)、Run(输入计算节点和和填充必要数据，运算，输出运算结果)。
会话源代码: tensorflow-1.1.0/tensorflow/python/client/session.py

设备(device)。一块用作运算、拥有自己地址空间的硬件。CPU、GPU。TensorFlow可以提定操作在哪个设备执行。with tf.device("/gpu:1"): 。

变量(variable)。特殊数据。图中有固定位置，不流动。tf.Variable()构造函数。初始值形状、类型。

    #创建一个变量，初始化为标量0
    state = tf.Variable(0,name="counter")
创建常量张量：

    state = tf.constant(3.0)
填充机制。构建图用tf.placeholder()临时替代任意操作张量，调用Session对象run()方法执行图，用填充数据作参数。调用结束，填充数据消失。

    input1 = tf.placeholder(tf.float32)
    input2 = tf.placeholder(tf.float32)
    output = tf.mul(input1,input2)
    with tf.Session() as sess:
        #输出[array([24.],dtype=float32)]
        print sess.run([output],feed_dict={input1:[7.],input2:[2.]})
变量源代码: tensorflow-1.1.0/tensorflow/python/ops/variables.py

内核。操作(operation)，抽象操作统称。内核(kernel)，运行在特定设备(CPU、GPU)上操作的实现。同一操作可能对应多个内核。自定义操作，新操作和内核注册添加到系统。

常用API。
图。TensorFlow计算表现为数据流图。tf.Graph类包含一系列计算操作对象(tf.Operation)和操作间流动数据张量对象(tf.Tensor)。
操作                                                             描述
tf.Graph.__init__()                                        创建一个空图
tf.Graph.as_default()                                    将某图设置为默认图，返回一个上下文管理器。不显示添加默认图，系统自动设置全局默认图。模块范围内定义节点都加入默认图
tf.Graph.device(device_name_or_function) 定义运行图所使用设备，返回上下文管理器
tf.Graph.name_scope(name)                       为节点创建层次化名称，返回上下方管理器

tf.Operaiotn类代表图中节点，用于计算张量数据。由节点构造器(如tf.matmul()、Graph.create_op())产生。
操作                                                                      描述
tf.Operation.name                                                操作名称
tf.Operation.type                                                  操作类型
tf.Operation.inputs                                               操作输入
tf.Operation.outputs                                             操作输出
tf.Operation.control_inputs                                   操作依赖
tf.Operation.run(feed_dict=None,session=None) 在会话中运行操作
tf.Operation.get_attr(name)                                   获取操作属性值

tf.Tensor类，操作输出符号句柄，不包含操作输出值，提供在tf.Session中计算值方法。操作间构建数据流连接，TensorFlow能免执行大量步多计算图形。
操作                                                                  描述
tf.Tensor.dtype                                                 张量数据类型
tf.Tensor.name                                                 张量名称
tf.Tensor.value_index                                       张量操作输出索引
tf.Tensor.graph                                                 张量所在图
tf.Tensor.op                                                       产生张量操作
tf.Tensor.consumers()                                       返回使用张量操作列表
tf.Tensor.eval(feed_dict=None,session=None) 会话中求张量值，使用sess.as_default()、eval(session=sess)
tf.Tensor.get_shape()                                        返回表示张量形状(维度)类TensorShape
tf.Tensor.set_shape(shape)                              更新张量形状
tf.Tensor.device                                                设置计算张量设备

可视化。
在程序中给节点添加摘要(summary)，摘要收集节点数据，标记步数、时间戳标识，写入事件文件(event file)。tf.summary.FileWriter类在目录创建事件文件，向文件添加摘要、事件，在TensorBoard展示。
操作                                                                      描述
tf.summary.FileWriter.__init__(logdir,graph=None,max_queue=10, flush_secs=120,graph_def=None) 创建FileWriter和事件文件，logdir中创建新事件文件
tf.summary.FileWriter.add_summary(summary,global_step=None) 摘要添加到事件文件
tf.summary.FileWriter.add_event(event) 事件文件添加事件
tf.summary.FileWriter.add_graph(graph,global_step=None,graph_def=None) 事件文件添加图
tf.summary.FileWriter.get_logdir() 事件文件路径
tf.summary.FileWriter.flush() 所有事件上写入磁盘
tf.summary.FileWriter.close() 事件写入磁盘，关闭文件操作符
tf.summary.scalar(name,tensor,collections=None) 输出单个标量值摘要
tf.summary.histogram(name,values,collections=None) 输出直方图摘要
tf.summary.audio(name,tensor,sample_rate,max_outputs=3,collections=None) 输出音频摘要
tf.summary.image(name,tensor,max_outputs=3,collections=None) 输出图片摘要
tf.summary.merge(inputs,collections=None,name=None) 合并摘要，所有输入摘要值

变量作用域。
TensorFlow两个作用域(scope)，name_scope(给op_name加前缀)，variable_scope(给variable_name、op_name加前缀)。
variable_scope变量作用域机制:
v = tf.get_variable(name,shape,dtype,initializer)#通过名字创建或返回变量
tf.variable_scope(<scope_name>)#给变量指定命名空间
tf.get_variable_scope().reuse==False(默认为False，不能得用)，variable_scope作用域只能创建新变量。tf.get_variable_scope().reuse==True,作用域共享变量，with tf.variable_scope(name,reuse=True)，或scope.reuse_variables()。
tf.variable_scope()获取变量作用域。开户变量作用域使用之前预先定义作用域，跳过当前变量作用域，保持预先存在作用域不变。
变量作用域可以默认携带一个初始化器。子作用域或变量可以继承或重写父作用域初始化器值。
op_name在variable_scope作用域操作，会加上前缀。
variable_scope主要用在循环神经网络(RNN)操作，大量共享变量。
name_scope。划分变量范围，可视化中表示在计算图一个层级。name_scope影响op_name，不影响用get_variable()创建变量。影响用Variable()创建变量。给操作加名字前缀。

批标准化。batch normalization,BN。优化梯度弥散问题(vanishing gradient problem)。
统计机器学习，ICS(Internal Covariate Shift)理论，源域(source domain)和目标域(target domain)数据分布一致。训练数据和测试数据满足相同分布。是通过训练数据获得模型在测试集获得好效果的基本保障。Covariate Shift，训练集样本数据和目标集分布不一致，训练模型无法很好泛化(generalization)。源域和目标域条件概率一致，边缘概率不同。神经网络各层输出，经层内操作，各层输出分布与输入信号分布不同，差异随网络加深变大，但每层指向样本标记(label)不变。解决，根据训练样本和目标样本比例矫正训练样本。引入批标准化规范化层输入(数据按比例缩放，落入小特定区间，数据去平均值，除以标准差)，固定每层输入信号均值、方差。
方法。批标准化通过规范化激活函数分布在线性区间，加大梯度，模型梯度下降。加大探索步长，加快收敛速度。更容易跳出局部最小值。破坏原数据分布，缓解过拟合。解决神经网络收敛速度慢或梯度爆炸(gradient explode，梯度非常大，链式求导乘积变得很大，权重过大，产生指数级爆炸)。

    #计算Wx_plus_b均值方差，axes=[0]标准化维度
    fc_mean,fc_var = tf.nn.moments(Wx_plus_b, axes=[0])
    scale = tf.Variable(tf.ones([out_size]))
    shift = tf.Variable(tf.zeros([out_size]))
    epsilon = 0.001
    Wx_plus_b = tf.nn.batch.normalization(Wx_plus_b,fc_mean,fc_var,shift,scale,epsilon) 
    #Wx_plus_b = (Wx_plus_b - fc_mean) / tf.sqrt(fc_var + 0.001)
    #Wx_plus_b = Wx_plus_b * scale + shift
《Batch Normalization:Accelerating Deep Network Training by Reducing Internal Covariate Shift》，Serger Ioffe、Christian Szegedy。

神经元函数优化方法。

激活函数。activation function，运行时激活神经网络某部分神经元，激活信息向后传入下层神经网络。加入非线性因素，弥补线性模型表达力，把“激活神经元特征”通过函数保留映射到下层。神经网络数学基础处处可微，选取激活函数保证输入输出可微。激活函数不改变输入数据维度，输入输出维度相同。TensorFlow激活函数定义在tensorflow-1.1.0/tensorflow/python/ops/nn.py。tf.nn.relu()、tf.nn.sigmoid()、tf.nn.tanh()、tf.nn.elu()、tf.nn.bias_add()、tf.nn.crelu()、tf.nn.relu6()、tf.nn.softplus()、tf.nn.softsign()、tf.nn.dropout()。输入张量，输出与输入张量数据类型相同张量。
sigmoid函数。输出映射在(0,1)内，单调连续，适合作输出层，求导容易。软饱和性，输入落入饱和区，f'(x)变得接近0,容易产生梯度消失。软饱和，激活函数h(x)取值趋于无穷大时，一阶导数趋于0。硬饱和，当|x|>c，c为常数，f'(x)=0。relu左侧硬饱和激活函数。梯度消失，更新模型参数，采用链式求导法则反向求导，越往前梯度越小。最终结果到达一定深度后梯度对模型更新没有任何贡献。
tanh函数。软饱和性，输出0为中心，收敛速度比sigmoid快。也无法解决梯度消失。
relu函数。最受欢迎。softplus是ReLU平滑版本。relu,f(x)=max(x,0)。softplus, f(x)=log(1+exp(x))。relu在x<0时硬饱和。x>0,导数为1,relu在x>0时保持梯度不衰减，缓解梯度消失，更快收敛，提供神经网络稀疏表达能力。部分输入落到硬饱和区，权重无法更新，神经元死亡。TensorFlow relu6,min(max(features,0)) ,tf.nn.relu6(features,name=None)。crelu，tf.nn.crelu(features,name=None) 。
dropout函数。神经元以概率keep_prob决定是否被抑制。如果被抑制，神经元就输出0,否则输出被放到原来的1/keep_prob倍。神经元是否被抑制，默认相互独立。noise_shape调节,noise_shape[i]==shape(x)[i]，x中元素相互独立。shape(x)=[k,l,m,n]，x维度顺序批、行、列、通道。noise_shape=[k,1,1,n]，批、通道互相独立，行、列数据关联，都为0,或原值。论文中最早做法，训练中概率p丢弃。预测中，参数按比例缩小，乘p。框架实现，反向ropout代替dropout，训练中一边dropout，再按比例放大，即乘以1/p，预测中不做任何处理。
激活函数选择。输入数据特征相差明显，用tanh，循环过程不断扩大特征效果显示。特征相差不明显，用sigmoid。sigmoid、tanh，需要输入规范化，否则激活后值全部进入平坦区，隐层输出全部趋同，丧失原有特征表达。relu会好很多，有时可以不做输入规范化。85%-90%神经网络都用ReLU。10-15%用tanh。

卷积函数。图像扫描二维过滤器。卷积函数定义,tensorflow-1.1.0/tensorflow/python/ops/nn_impl.py、nn_ops.py 。
tf.nn.convolution(input,filter,padding,strides=None,dilation_rate=None, name=None,data_format=None) 计算N维卷积和。
tf.conv2d(input,filter,strides,padding,use_cudnn_on_gpu=None,data_format=None,name=None)，四维输入数据ipnut和四维卷积核filter操作，输入数据二维卷积操作，得到卷积结果。input,Tensor,数据类型floate32､floate64。filter,Tensor,数据类型floate32､floate64。strides:长度4一维整数类型数组，每一维度对应input每一维移动步数。padding,字符串,SAME 全尺寸操作 输入、输出数据维度相同，VALID 部分窗口 输入、输出数据维度不同。use_cudnn_on_gpu ,可选布尔值,默认True。name,可选,操作名字。输出，Tensor,数据类型floate32､floate64。
tf.nn.depthwise_conv2d(input,filter,strides,padding,rate=None,name=None, data_format=None)，输入张量数据维度[batch,in_height,in_width,in_width,in_channels] ，卷积核维度[filter_height,filter_width,in_channel_multiplier],通道in_channels卷积深度1,depthwise_conv2d函数将不同卷积核独立应用在in_channels每个通道上，再把所有结果汇总。输出通道总数in_channels*channel_multiplier 。
tf.nn.separable_conv2d(input,depthwise_filter,pointwise_filter,strides,padding,rate=None,name=None,data_format=None) 用几个分离卷积核做卷积。用二维卷积核在每个通道上，以深度channel_multiplier卷积。depthwise_filter, 张量,数据四维[filter_height,filter_width,in_channels,channel_multiplier]，in_channels卷积深度1。pointwise_filter,张量,数据四维[1,1,channel_multiplier*in_channels,out_channels]，pointwise_filter，在depthwise_filter卷积后混合卷积。
tf.nn.atrous_conv2d(value,filters,rate,padding,name=None)计算Atrous卷积，孔卷积，扩张卷积。
tf.nn.conv2d_transpose(value,filter,output_shape,strides,padding='SAME', data_format='NHWC',name=None),解卷积网络(deconvolutional network)中称'反卷积'，实际上是conv2d的转置。output_shape，一维张量，反卷积运算输出形状。
tf.nn.conv1d(value,filters,stride,padding,use_cudnn_on_gpu=None,data_format=None,name=None)，计算给定三维输入和过滤器的一维卷积。输入三维[batch,in_width,in_channels],卷积核三维，少filter_height,[filter_width,in_channels,out_channels] ，stride正整数，卷积核向右移动每一步长度。
tf.nn.conv3d(input,filter,strides,padding,name=None)计算给定五维输入和过滤器的三维卷积。input shape多一维in_depth，形状Shape[batch,in_depth,in_height,in_width,in_channels] 。filter shape多一维filter_depth，卷积核大小filter_depth,filter_height,filter_width。strides多一维，[strides_batch, strides_depth,strides_height,strides_sidth,strides_channel],必须保证strides[0]=strides[4]=1。
tf.nn.conv3d_transpose(value,filter,output_shape,strides,padding='SAME', name=None)。

池化函数。神经网络，池化函数一般跟在卷积函数下一层，定义在tensorflow-1.1.0/tensorflow/python/ops/nn.py、gen_nn_ops.py。
池化操作，用一个矩阵窗口在张量上扫描，每个矩阵窗口中的值通过取最大值或平均值来减少元素个数。每个池化操作矩阵窗口大小ksize指定，根据步长strides移动。
tf.nn.avg_pool(value,ksize,strides,padding,data_format='NHWC',name=None)计算池化区域元素平均值。value,四维张量，数据维度[batch,height,width, channels]。ksize,长度不小于4整型数组，每位值对应输入数据张量每维窗口对应值。strides,长度不小于4整型数组，批定滑动窗口在输入数据张量每一维上的步长。padding，字符串，SAME或VALID。data_format,'NHWC'，输入张量维度顺序，N个数，H高度，W宽度，C通道数(RGB三通道或灰度单通道)。name，可选，操作名字。
tf.nn.max_pool(value,ksize,strides,padding,data_format='NHWC', name=None)计算池化区域元素最大值。
tf.nn.max_pool_with_argmax(input,ksize,strides,padding,Targmax=None, name=None)，计算池化区域元素最大值和所在位置。计算位置agrmax,input铺平。如input=[b,y,x,c],索引位置((b*height+y)*width+x)*channels+c。只能在GPU运行。返回张量元组(output,argmax)，output池化区域最大值，argmax数据类型Targmax，四维。
tf.nn.avg_pool3d()、tf.nn.max_pool3d() 三维平均池化和最大池化。
tf.nn.fractional_avg_pool()、tf.nn.tractional_max_pool()
tf.nn.pool(input,window_shape,pooling_type,padding,dilation_rate=None, strides=None,name=None,data_format=None)执行N维池化操作。

分类函数。定义在tensorflow-1.1.0/tensorflow/python/ops/nn.py、nn_ops.py。
tf.nn.sigmoid_cross_entropy_with_logits(logits,targets,name=None)。输入,logtis:[batch_size,num_classes],targets:[batch_size,size],logits用最后一层输入。输出,loss [batch_size,num_classes]。如作损失函数，神经网络最后一层不需要sigmoid运算。
tf.nn.softmax(logits,dim=-1,name=None)计算Softmax激活，softmax=exp(logits) /reduce_sum(exp(logits),dim)。
tf.nn.log_softmax(logits,dim=-1,name=None)计算log softmax激活，logsoftmax=logits-log(reduce_sum(exp(logits),dim))。
tf.nn.softmax_cross_entropy_with_logits(_sentinel=None,labels=None,logits=None,dim=-a,name=None)。输入，logits、lables [batch_size,num_classes] ，输出，loss [batch_size],保存batch 每个样本交叉熵。
tf.nn.sparse_softmax_cross_entropy_with_logits(logits,labels,name=None)。logits神经网络最后一层结果。输入，logits [batch_size,num_classes],labels [batch_size]，必须在[0,num_classes]。输出，loss [batch_size],保存batch 每个样本交叉熵。

优化方法。加速训练优化方法，多数基于梯度下降。梯度下降求函数极值。学习最后求损失函数极值。TensorFlow提供很多优化器(optimizer)。
 BGD法。bat gradient descent，批梯度下降。利用现有参数对训练集每个输入生成一个估计输出yi。跟实际输出yi比较，统计所有误差，求平均以后得到平均误差，以此更新参数。迭代过程，提取训练集中所有内容{x1,...,xn}，相关输出yi 。计算梯度和误差，更新参数。使用所有训练数据计算，保证收敛，不需要逐渐减少学习率。每一步都需要使用所有训练数据，速度越来越慢。
SDG法。stochastic gradient descent，随机梯度下降。数据集拆分成一个个批次(batch)，随机抽取一个批次计算，更新参数，MBGD(minibatch gradient descent)。每次迭代计算mini-batch梯度，更新参数。训练数据集很大，仍能较快速度收敛。抽取不可避免梯度误差，需要手动调整学习率(learning rate)。选择适合学习率比较困难。想对常出现特征更新速度快，不常出现特征更新速度慢。SGD更新所有参数用一样学习率。SGD容易收敛到局部最优，可能被困在鞍点。
Momentum法。模拟物理学动量概念。更新时在一定程度保留之前更新方向，当前批次再微调本次更新参数，引入新变量v(速度)，作为前几次梯度累加。Momentum更新学习率，在下降初期，前后梯度方向一致时，加速学习，在下降中后期，在局部最小值附近来回震荡时，抑制震荡，加快收敛。
Nesterov Momentum法。Ilya Sutskever，Nesterov。标准Momentum法，先计算一个梯度，在加速更新梯度方向大跳跃。Nesterov法，先在原来加速梯度方向大跳跃，再在该位置计算梯度值，用这个梯度值修正最终更新方向。
Adagrad法。自适应为各个参数分配不同学习率，控制每个维度梯度方向。实现学习率自动更改。本次更新梯度大，学习率衰减快，否则慢。
Adadelta法。Adagrad法，学习单调递减，训练后期学习率非常小，需要手动设置一个全局初始学习率。Adadelta法用一阶方法，近似模拟二阶牛顿法，解决问题。
RMSprop法。引入一个衰减系数，每一回合都衰减一定比例。对循环神经网络(RNN)效果很好。
Adam法。自适应矩估计(adaptive moment estimation)。Adam法根据损失函数针对每个参数梯度一阶矩估计和二阶矩估计动态调整每个参数学习率。矩估计，利用样本矩估计总体相应参数。一个随机变量X服从某种分布，X一阶矩是E(X)，样本平均值，X二阶矩E(X2)，样本平方平均值。
方法比较。Karpathy在MNIST数据集发现规律：不调整参数，Adagrad法比SGD法、Momentum法更稳定、性能更优。精调参数，精调SGD法、Momentum法收敛速度和准确性优于Adagrad法。http://sebastianruder.com/optimizing-gradient-descent/。《An overview of gradient descent optimization algorithms》。

参考资料：
《TensorFlow技术解析与实战》


