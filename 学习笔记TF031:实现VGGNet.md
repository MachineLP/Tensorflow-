VGGNet，牛津大学计算机视觉组(Visual Geometry Group)和Google DeepMind公司一起研发，深度卷积神经网络。VGGNet反复堆叠3x3小型卷积核和2x2最大池化层，成功构筑16~19层深卷积神经网络。比state-of-the-art网络结构，错误率幅下降，取得ILSVRC 2014比赛分类第2名和定位第1名。拓展性强，迁移其他图片数据泛化性好。结构简洁，整个网络都用同样大小卷积核尺寸和最大池化尺寸。VGGNet训练后模型参数官方开源，domain specific图像分类任务再训练,提供较好初始化权重。

                       ConvNet Configuration
    A           A-LRN      B         C         D         E
    weight layers 11 11    13        16        16        19 
                      input(224x224 RGB image)
    conv3-64  conv3-64  conv3-64  conv3-64  conv3-64  conv3-64
                 LRN    conv3-64  conv3-64  conv3-64  conv3-64
                             maxpool
    conv3-128 conv3-128 conv3-128 conv3-128 conv3-128 conv3-128
                        conv3-128 conv3-128 conv3-128 conv3-128
                             maxpool
    conv3-256 conv3-256 conv3-256 conv3-256 conv3-256 conv3-256
    conv3-256 conv3-256 conv3-256 conv3-256 conv3-256 conv3-256
                        conv1-256 conv3-256 conv3-256 conv3-256
                              maxpool
    conv3-512 conv3-512 conv3-512 conv3-512 conv3-512 conv3-512
    conv3-512 conv3-512 conv3-512 conv3-512 conv3-512 conv3-512
                        conv1-512 conv3-512 conv3-512 conv3-512
                             maxpool
    conv3-512 conv3-512 conv3-512 conv3-512 conv3-512 conv3-512
    conv3-512 conv3-512 conv3-512 conv3-512 conv3-512 conv3-512
                        conv1-512 conv3-512 conv3-512 conv3-512
                             maxpool
                             FC-4096
                             FC-4096
                             FC-1000
                             soft-max

    Network                 A,A-LRN   B    C    D    E
    Number of parameters      133    133  134  138  144

卷积层参数量少，最后3个全连接层参数多。训练耗时在卷积，计算量较大。D为VGGNet-16，E为VGGNet-19。C比B多3个1x1卷积层，线性变换，输入、输出通道数不变，没降维。

VGGNet 5段卷积，每段2~3卷积层，每段后接最大池化层给缩小图片尺寸。每段卷积核数量一样，越后段卷积核数量越多，64-128-256-512-512。多个3x3卷积层堆叠。2个3x3卷积层串联相当1个5x5。3个3x3卷积层串联相当1个7x7。 参数更少，非线性变换更多，增强特征学习能力。

先训练级别A简单网络，再复用A网络权重初如化复杂模型，训练收敛速度更快。预测，Multi-Scale，图像scale尺寸Q，图片输入卷积网络计算。最后卷积层，滑窗分类预测，不同窗口分类结果平均，不同尺寸Q结果平均得最后结果，提高图片数据利用率，提升预测准确率。训练过程，用Multi-Scale数据增强，原始图像缩放不同尺寸S，随机裁切224x224图片，增加数据量，防止过拟合。

LRN层作用不大，越深网络效果越好，1x1卷积很有效，但大卷积核可以学习更大空间特征。

载入系统库、TensorFlow。

conv_op函数，创建卷积层，参数存入参数列表。输入，input_op tensor，name 层名，kh kernel height 卷积核高，kw kernel width 卷积核宽，n_out 卷积核数量 输出通道数，dh 步长高，dw 步长宽，p参数列表。get_shape()[-1].value获取输入input_op通道数。tf.name_scope(name)设置scope。tf.get_variable创建kernel(卷积核)，shape [kh,kw,n_in,n_out]，卷积核高宽、输入输出通道数。tf.contrib.layers.xavier_initializer_conv2d()参数初始化。

tf.nn.conv2d卷积处理input_op。卷积核kernel，步长dhxdw，paddings模式SAME。tf.constant 赋值biases 0，tf.Variable转可训练参数。tf.nn.bias_add 相加卷积结果conv和bias，tf.nn.relu非线性处理得activation。创建卷积层，参数kernel、biases添加到参数列表p，卷积层输出activation返回。

全连接层创建函数 fc_op。先获取输入input_op通道数。tf.get_variable创建全连接层参数，第一维度输入通道数n_in，第二维度输出通道数n_out。xavier_initializer参数初始化。biases初始化0.1,避免dead neuron。tf.nn.relu_layer矩阵相乘input_op、kernel，加biases，ReLU非线性，交换得activation。全连接层参数kernel、biases添加参数列表p， activation返回。

定义最大池化层创建函数mpool_op。tf.nn.max_pool，输入input_op，池化尺寸khxkw，步长dhxdw，padding模式SAME。

VGGNet-16网络结构,6个部分，前5段卷积网络，最后一段全连接网络。定义创建VGGNet网络结构函数inference_op。输入input_op、keep_prob(控制dropout比率，placeholder)。先初始化参数列表p。

创建第一段卷积网络，两个卷积层(conv_op)，一个最大池化层(mpool_op)。卷积核大小3x3，卷积核数量(输出通道数) 64，步长1x1，全像素扫描。第一卷积层输入input_op尺寸224x224x3，输出尺寸224x224x64。第二卷积层输入输出尺寸224x224x64。最大池化层2x2，输出112x112x64。

第二段卷积网络，2个卷积层，1个最大池化层。卷积输出通道数128。输出尺寸56x56x128。

第三段卷积网络，3个卷积层，1个最大池化层。卷积输出通道数256。输出尺寸28x28x256。

第四段卷积网络，3个卷积层，1个最大池化层。卷积输出通道数512。输出尺寸14x14x512。

第五段卷积网络，3个卷积层，1个最大池化层。卷积输出通道数512。输出尺寸7x7x512。输出结果每个样本，tf.reshape 扁平化为长度7x7x512=25088一维向量。

连接4096隐含点全连接层，激活函数ReLU。连接Dropout层，训练节点保留率0.5，预测1.0。

全连接层，Dropout层。

最后连接1000隐含点全连接层，Softmax 分类输出概率。tf.argmax 输出概率最大类别。返回fc8､softmax、predictions、参数列表p。

VGGNet-16网络结构构建完成。

评测函数time_tensorflow_run。session.run()方法，引入feed_dict，方便传入keep_prob控制Dropout层保留比率。

评测主函数run_benchmark。评测forward(inference)、backward(trainning)运算性能。生成尺寸224x224随机图片，tf.random_nornal函数生成标准差0.1正态分布随机数。

创建keep_prob placeholder，调用inference_op函数构建VGGNet-16网络结构，获得predictions、softmax、fc8、参数列表p。

创建Session，初始化全局参数。设keep_prob 1.0 预测。time_tensorflow_run评测forward运算时间。

计算VGGNet-16最后全连接层输出fc8 l2 loss。tf.gradients求loss所有模型参数梯度。time_tensorflow_run评测backward运算时间。target为求解梯度操作grad，keep_prob 0.5。设batch_size 32。

执行评测主函数run_benchmark()，测试VGGNet-16 TensorFlow forward、backward耗时。forward平均每个batch耗时0.152s。backward求解梯度，平均每个batch耗时0.617s。

VGGNet，7.3%错误率。更深网络，更小卷积核，隐式正则化。


    from datetime import datetime
    import math
    import time
    import tensorflow as tf
    def conv_op(input_op, name, kh, kw, n_out, dh, dw, p):
        n_in = input_op.get_shape()[-1].value
        with tf.name_scope(name) as scope:
            kernel = tf.get_variable(scope+"w",
                                 shape=[kh, kw, n_in, n_out],
                                 dtype=tf.float32, 
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())
            conv = tf.nn.conv2d(input_op, kernel, (1, dh, dw, 1), padding='SAME')
            bias_init_val = tf.constant(0.0, shape=[n_out], dtype=tf.float32)
            biases = tf.Variable(bias_init_val, trainable=True, name='b')
            z = tf.nn.bias_add(conv, biases)
            activation = tf.nn.relu(z, name=scope)
            p += [kernel, biases]
            return activation
    def fc_op(input_op, name, n_out, p):
        n_in = input_op.get_shape()[-1].value
        with tf.name_scope(name) as scope:
            kernel = tf.get_variable(scope+"w",
                                 shape=[n_in, n_out],
                                 dtype=tf.float32, 
                                 initializer=tf.contrib.layers.xavier_initializer())
            biases = tf.Variable(tf.constant(0.1, shape=[n_out], dtype=tf.float32), name='b')
            activation = tf.nn.relu_layer(input_op, kernel, biases, name=scope)
            p += [kernel, biases]
            return activation
    def mpool_op(input_op, name, kh, kw, dh, dw):
        return tf.nn.max_pool(input_op,
                          ksize=[1, kh, kw, 1],
                          strides=[1, dh, dw, 1],
                          padding='SAME',
                          name=name)
    def inference_op(input_op, keep_prob):
        p = []
        # assume input_op shape is 224x224x3
        # block 1 -- outputs 112x112x64
        conv1_1 = conv_op(input_op, name="conv1_1", kh=3, kw=3, n_out=64, dh=1, dw=1, p=p)
        conv1_2 = conv_op(conv1_1,  name="conv1_2", kh=3, kw=3, n_out=64, dh=1, dw=1, p=p)
        pool1 = mpool_op(conv1_2,   name="pool1",   kh=2, kw=2, dw=2, dh=2)
        # block 2 -- outputs 56x56x128
        conv2_1 = conv_op(pool1,    name="conv2_1", kh=3, kw=3, n_out=128, dh=1, dw=1, p=p)
        conv2_2 = conv_op(conv2_1,  name="conv2_2", kh=3, kw=3, n_out=128, dh=1, dw=1, p=p)
        pool2 = mpool_op(conv2_2,   name="pool2",   kh=2, kw=2, dh=2, dw=2)
        # # block 3 -- outputs 28x28x256
        conv3_1 = conv_op(pool2,    name="conv3_1", kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
        conv3_2 = conv_op(conv3_1,  name="conv3_2", kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
        conv3_3 = conv_op(conv3_2,  name="conv3_3", kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)    
        pool3 = mpool_op(conv3_3,   name="pool3",   kh=2, kw=2, dh=2, dw=2)
        # block 4 -- outputs 14x14x512
        conv4_1 = conv_op(pool3,    name="conv4_1", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
        conv4_2 = conv_op(conv4_1,  name="conv4_2", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
        conv4_3 = conv_op(conv4_2,  name="conv4_3", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
        pool4 = mpool_op(conv4_3,   name="pool4",   kh=2, kw=2, dh=2, dw=2)
        # block 5 -- outputs 7x7x512
        conv5_1 = conv_op(pool4,    name="conv5_1", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
        conv5_2 = conv_op(conv5_1,  name="conv5_2", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
        conv5_3 = conv_op(conv5_2,  name="conv5_3", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
        pool5 = mpool_op(conv5_3,   name="pool5",   kh=2, kw=2, dw=2, dh=2)
        # flatten
        shp = pool5.get_shape()
        flattened_shape = shp[1].value * shp[2].value * shp[3].value
        resh1 = tf.reshape(pool5, [-1, flattened_shape], name="resh1")
        # fully connected
        fc6 = fc_op(resh1, name="fc6", n_out=4096, p=p)
        fc6_drop = tf.nn.dropout(fc6, keep_prob, name="fc6_drop")
        fc7 = fc_op(fc6_drop, name="fc7", n_out=4096, p=p)
        fc7_drop = tf.nn.dropout(fc7, keep_prob, name="fc7_drop")
        fc8 = fc_op(fc7_drop, name="fc8", n_out=1000, p=p)
        softmax = tf.nn.softmax(fc8)
        predictions = tf.argmax(softmax, 1)
        return predictions, softmax, fc8, p
    
    def time_tensorflow_run(session, target, feed, info_string):
        num_steps_burn_in = 10
        total_duration = 0.0
        total_duration_squared = 0.0
        for i in range(num_batches + num_steps_burn_in):
            start_time = time.time()
            _ = session.run(target, feed_dict=feed)
            duration = time.time() - start_time
            if i >= num_steps_burn_in:
                if not i % 10:
                    print ('%s: step %d, duration = %.3f' %
                       (datetime.now(), i - num_steps_burn_in, duration))
                total_duration += duration
                total_duration_squared += duration * duration
        mn = total_duration / num_batches
        vr = total_duration_squared / num_batches - mn * mn
        sd = math.sqrt(vr)
        print ('%s: %s across %d steps, %.3f +/- %.3f sec / batch' %
           (datetime.now(), info_string, num_batches, mn, sd))
    def run_benchmark():
        with tf.Graph().as_default():
            image_size = 224
            images = tf.Variable(tf.random_normal([batch_size,
                                               image_size,
                                               image_size, 3],
                                               dtype=tf.float32,
                                               stddev=1e-1))
            keep_prob = tf.placeholder(tf.float32)
            predictions, softmax, fc8, p = inference_op(images, keep_prob)
            init = tf.global_variables_initializer()
            config = tf.ConfigProto()
            config.gpu_options.allocator_type = 'BFC'
            sess = tf.Session(config=config)
            sess.run(init)
            time_tensorflow_run(sess, predictions, {keep_prob:1.0}, "Forward")
            objective = tf.nn.l2_loss(fc8)
            grad = tf.gradients(objective, p)
            time_tensorflow_run(sess, grad, {keep_prob:0.5}, "Forward-backward")
    batch_size=32
    num_batches=100
    run_benchmark()

参考资料：
《TensorFlow实战》


