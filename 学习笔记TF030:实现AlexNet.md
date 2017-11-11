ILSVRC(ImageNet Large Scale Visual Recognition Challenge)分类比赛。AlexNet 2012年冠军(top-5错误率16.4%，额外数据15.3%，8层神经网络)。VGGNet 2014年亚军(top-5错误率7.3%，19层神经网络)。Google Inception 2014年冠军(top-5错误率6.7%，22层神经网络)。ResNet 2015年冠军(top-5错误率3.57%，152层神经网络)。人眼错误率5.1%。卷积神经网络基本解决ImageNet数据集图片分类问题。

ImageNet，2007年斯坦福大学李飞飞教授创办，收集大量带标注信息图片数据供计算机视觉模型训练。1500万张标注高清图片，22000类，100万张标注图片主要物体定位边框。眼睛生物照相机，每200ms拍一张，3岁已经上亿张。ImageNet下载互联网10亿张图片，亚马逊土耳其机器人平台众包标注过程，167个国家5万名工作者筛选标注。每年ILSVRC比赛数据120万张图片，1000类标注。top-5､top-1分类错误率模型性能评测指标。

2012年，Hinton学生Alex Krizhevsky提出LeNet更深更宽版深度神经网络模型AlexNet。首次在CNN用ReLU、Dropout、LRN等Trick。GPU运算加速。开源GPU训红卷积神经网络CUDA代码。6亿3000万连接，6000万参数，65万神经元，5个卷积层，3个卷积层后连最大池化层，3个全连接层。top-5错误率16.4%。第二名26.2%。神经网络低谷期后第一次发声，确立深度学习(深度卷积网络)计算机视觉统治地位，推动深度学习语音识别、自然语言处理、强化学习领域拓展。

成功使用ReLU CNN激活函数，验证较深网络效果超过Sigmoid，解决梯度弥散问题。训练用Dropout随机忽略部分神经元，避免模型过拟合。CNN用重叠最大池化，避免平均池化模糊化效果，步长比池化核尺寸小，池化层输出重叠覆盖，提升特征丰富性。LRN层，创建局部神经元活动竞争机制，响应较大值变更大，抑制其他反馈较小神经元，增强模型泛化能力。CUDA加速深度卷积网络训练，并行计算大量矩阵，GPU通信方便，互相访问显存，只在网络某些层进行，控制通信性能损耗。数据增强，随机从256x256原始图像截取224x224区域，水平翻转镜像，增加(256-224)^2x2=2048倍数据量，减轻过拟合，提升泛化能力，预测取图片四角加中间5个位置，左右翻转，10张图片，10次预测结果求均值，图像RGB数据PCA处理，主成分标准差0.1高斯扰动，增加噪声，错误率下降1%。

8个参数训练层(不包括池化层、LRN层)，前5卷积层，后3全连接层。最后一层1000类输出Softmax层分类。LRN层在第1､2卷积层后，最大池化层在LRN层、最后卷积层后。ReLU激活函数在每个参数层后。

AlexNet超参数：
params  AlexNext                        FLOPs
4M         FC1000                             4M
16M       FC4096/ReLU                   4M
37M       FC4096/ReLU                 37M
              Max Pool 3x3s2   
442K     Conv 3x3s1,256/ReLU     74M
1.3M     Conv 3x3s1,384/ReLU     112M
884K     Conv 3x3s1,384/ReLU     149M
              Max Pool 3x3s2   
              Local Response Norm  
307K     Conv 5x5s1,256/ReLU     223M
              Max Pool 3x3s2   
              Local Response Norm  
35K       Conv 11x11s4,96/ReLU   105M

输入图片尺寸224x224，第一卷积层卷积核尺寸11x11，步长4，96个卷积核。LRN层。3x3 最大池化层，步长2。后续卷积核5x5或3x3，步长2。通过较小参数提取有效特征。

导入系统库datetime、math、time，载入TensorFlow。

batch_size 32，num_batches 100，共测试100个batch数据。

定义网络结构显示函数print_actications，卷积层或池化层输出tensor尺寸。接受tensor输入，显示名称(t.op.name)、tensor尺寸(t.get_shape.as_list())。

网络结构。定义inference函数，接受images输入，返回最后一层pool5(第5个池化层）及parameters(模型参数)。多个卷积层、池化层。

第一卷积层conv1，TensorFlow name_scope，with tf.name_scope('conv1') as scope，scope内生成Variable自动命名为conv1/xxx，区分不同卷积层组件。定义第一卷积层，tf.truncated_normal截断正态分布函数(标准差0.1)，初始化卷积核参数kernel。卷积核尺寸11x11，颜色通道3，卷积核64。tf.nn.conv2d卷积images，strides步长4x4(图片每4x4区域只取样一次，横向间隔4，纵向间隔4，取样卷积核尺寸11x11)，padding模式SAME。卷积层biases全初始化0。tf.nn.bias_add，conv、biases加。用激活函数tf.nn.relu结果非线性。print_activations 打印最后输出tensor conv1结构。可训练参数kernel、biases添加parameters。

第一卷积层后添加LRN层、最大池化层。tf.nn.lrn LRN处理前面输出tensor conv1，depth_radius 4，bias  1，alpha 0.001/9，beta 0.75。其他经典卷积神经网络放充LRN。LRN让前馈、反馈速度下降到1/3。tf.nn.max_pool 最大池化处理前面输出lrn1，尺寸 3x3，3x3像素块降为1x1像素，取样步长2x2，padding模式VALID，取样不超过边框，不填充边界外点(SAME)。打印输出结果pool1结构。

第二卷积层，卷积核尺寸5x5，输入通道数 64(上一层输出通道数，上一层卷积核数量)，卷积核192，步长1，扫描全图像素。

处理第二卷积层输出conv2，先LRN处理，再最大池化。

第三卷积层，卷积核尺寸3x3，输入通道数 192，卷积核 384，步长1。

第四卷积层，卷积核尺寸3x3，输入通道数 384，卷积核 256，步长1。

第五卷积层，卷积核尺寸3x3，输入通道数 256，卷积核 256，步长1。

最大池化层，返回池化层输出pool5。卷积结束。

3个全连接层，隐含节点4096､4096、1000，计算量很小。

评估AlexNet每轮计算时间函数time_tensorflow_run。第一输入TensorFlow Session，第二变量评测运算算子，第三变量测试名称。定义预热轮数num_steps_burn_in=10，给程序热身，头几轮迭代显存加载、cache命中问题跳过，10轮迭代后计算时间。记录总时间total_duration、平方和total_duration_squared，计算方差。

num_batches+num_steps_burn_in次迭代计算，time.time()记录时间，session.run(target)执行每次迭代。初始热身num_steps_burn_in次迭代后，每10轮迭代显示当前迭代时间。每轮total_duration、total_duration_squared累加。

循环结束，计算每轮耗时均值mn、标准差sd，显示结果。

主函数run_benchmark。with tf.Graph().as_default()定义默认Graph。tf.random_nomal函数构造正态颁上(标准差 0.1)随机tensor，第一维度batch_size，每轮迭代样本数，第二、三维度图片尺寸image_size 224，第四维度图片颜色通道数。inference函数构建整个AlexNet网络，最后池化层输出pool5，训练参数集合parameters。tf.Session()创建新Session，tf.global_variables_initializer()初始化所有参数。

AlexNet forward计算评测，time_tensorflow_run 统计运算时间，传入target pool5，卷积网络最后池化层输出。backward 训练过程评测，最后输出pool5设置优化目标loss。tf.nn.l2_loss计算，tf.gradients求loss所有模型参数梯度，模拟训练过程，根据梯度更新参数。time_tensorflow_run统计backward运算时间，target求整个网络梯度gard。

执行主函数。

程序显示三段结果。AlexNet网络结构、输出tensor尺寸。forward计算时间，有LRN层每轮迭代时间0.026s，去除LRN层0.007s，对最终准确率影响不大。backward运算时间，有LRN层每轮迭代时间0.078s，去除LRN层0.025s。backward运算耗时约forward三倍。

CNN训练过程(backward计算)比较耗时，过很多遍数据，大量迭代。CNN瓶劲在训练。TensorFlow已经支持iOS、Android，手机CPU做人脸识别、图片分类非常方便、响应速度很快。

传统机器学习模型适合学习小型数据集，大型数据集需要更大学习容量(Learning Capacity)模型，深度学习模型。卷积层参数量少，抽取特征能力非常强。


    from datetime import datetime
    import math
    import time
    import tensorflow as tf
    batch_size=32
    num_batches=100
    def print_activations(t):
        print(t.op.name, ' ', t.get_shape().as_list())
    def inference(images):
        parameters = []
        # conv1
        with tf.name_scope('conv1') as scope:
            kernel = tf.Variable(tf.truncated_normal([11, 11, 3, 64], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(images, kernel, [1, 4, 4, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                             trainable=True, name='biases')
            bias = tf.nn.bias_add(conv, biases)
            conv1 = tf.nn.relu(bias, name=scope)
            print_activations(conv1)
            parameters += [kernel, biases]
      # pool1
        lrn1 = tf.nn.lrn(conv1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='lrn1')
        pool1 = tf.nn.max_pool(lrn1,
                           ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1],
                           padding='VALID',
                           name='pool1')
        print_activations(pool1)
      # conv2
        with tf.name_scope('conv2') as scope:
            kernel = tf.Variable(tf.truncated_normal([5, 5, 64, 192], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[192], dtype=tf.float32),
                             trainable=True, name='biases')
            bias = tf.nn.bias_add(conv, biases)
            conv2 = tf.nn.relu(bias, name=scope)
            parameters += [kernel, biases]
        print_activations(conv2)
      # pool2
        lrn2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='lrn2')
        pool2 = tf.nn.max_pool(lrn2,
                           ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1],
                           padding='VALID',
                           name='pool2')
        print_activations(pool2)
      # conv3
        with tf.name_scope('conv3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 192, 384],
                                                 dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32),
                             trainable=True, name='biases')
            bias = tf.nn.bias_add(conv, biases)
            conv3 = tf.nn.relu(bias, name=scope)
            parameters += [kernel, biases]
            print_activations(conv3)
      # conv4
        with tf.name_scope('conv4') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 256],
                                                 dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                             trainable=True, name='biases')
            bias = tf.nn.bias_add(conv, biases)
            conv4 = tf.nn.relu(bias, name=scope)
            parameters += [kernel, biases]
            print_activations(conv4)
      # conv5
        with tf.name_scope('conv5') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256],
                                                 dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                             trainable=True, name='biases')
            bias = tf.nn.bias_add(conv, biases)
            conv5 = tf.nn.relu(bias, name=scope)
            parameters += [kernel, biases]
            print_activations(conv5)
      # pool5
        pool5 = tf.nn.max_pool(conv5,
                           ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1],
                           padding='VALID',
                           name='pool5')
        print_activations(pool5)
        return pool5, parameters
    def time_tensorflow_run(session, target, info_string):
        num_steps_burn_in = 10
        total_duration = 0.0
        total_duration_squared = 0.0
        for i in range(num_batches + num_steps_burn_in):
            start_time = time.time()
            _ = session.run(target)
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
            pool5, parameters = inference(images)
            init = tf.global_variables_initializer()
            config = tf.ConfigProto()
            config.gpu_options.allocator_type = 'BFC'
            sess = tf.Session(config=config)
            sess.run(init)
            time_tensorflow_run(sess, pool5, "Forward")
            objective = tf.nn.l2_loss(pool5)
            grad = tf.gradients(objective, parameters)
            time_tensorflow_run(sess, grad, "Forward-backward")
    run_benchmark()

参考资料：
《TensorFlow实战》


