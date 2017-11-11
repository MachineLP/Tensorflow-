TensorFlow实现Softmax Regression(回归)识别手写数字。MNIST(Mixed National Institute of Standards and Technology database)，简单机器视觉数据集，28X28像素手写数字，只有灰度值信息，空白部分为0,笔迹根据颜色深浅取[0, 1], 784维，丢弃二维空间信息，目标分0~9共10类。数据加载，data.read_data_sets, 55000个样本，测试集10000样本，验证集5000样本。样本标注信息，label，10维向量，10种类one-hot编码。训练集训练模型，验证集检验效果，测试集评测模型(准确率、召回率、F1-score)。

算法设计，Softmax Regression训练手写数字识别分类模型，估算类别概率，取概率最大数字作模型输出结果。类特征相加，判定类概率。模型学习训练调整权值。softmax，各类特征计算exp函数，标准化(所有类别输出概率值为1)。y = softmax(Wx+b)。

NumPy使用C、fortran，调用openblas、mkl矩阵运算库。TensorFlow密集复杂运算在Python外执行。定义计算图，运算操作不需要每次把运算完的数据传回Python，全部在Python外面运行。

import tensor flow as tf，载入TensorFlow库。less = tf.InteractiveSession()，创建InteractiveSession，注册为默认session。不同session的数据、运算，相互独立。x = tf.placeholder(tf.float32, [None,784])，创建Placeholder 接收输入数据，第一参数数据类型，第二参数代表tensor shape 数据尺寸。None不限条数输入，每条输入为784维向量。

tensor存储数据，一旦使用掉就会消失。Variable在模型训练迭代中持久化，长期存在，每轮迭代更新。Softmax Regression模型的Variable对象weights、biases 初始化为0。模型训练自动学习合适值。复杂网络，初始化方法重要。w = tf.Variable(tf.zeros([784, 10]))，784特征维数，10类。Label，one-hot编码后10维向量。

Softmax Regression算法，y = tf.nn.softmax(tf.matmul(x, W) + b)。tf.nn包含大量神经网络组件。tf.matmul，矩阵乘法函数。TensorFlow将forward、backward内容自动实现，只要定义好loss，训练自动求导梯度下降，完成Softmax Regression模型参数自动学习。

定义loss function描述问题模型分类精度。Loss越小，模型分类结果与真实值越小，越精确。模型初始参数全零，产生初始loss。训练目标是减小loss，找到全局最优或局部最优解。cross-entropy，分类问题常用loss function。y预测概率分布，y'真实概率分布(Label one-hot编码)，判断模型对真实概率分布预测准确度。cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))。定义placeholder，输入真实label。tf.reduce_sum求和，tf.reduce_mean每个batch数据结果求均值。

定义优化算法，随机梯度下降SGD(Stochastic Gradient Descent)。根据计算图自动求导，根据反向传播(Back Propagation)算法训练，每轮迭代更新参数减小loss。提供封装优化器，每轮迭代feed数据，TensorFlow在后台自动补充运算操作(Operation)实现反向传播和梯度下降。train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)。调用tf.train.GradientDescentOptimizer，设置学习速度0.5,设定优化目标cross-entropy，得到训练操作train_step。

tf.global_variables_initializer().run()。TensorFlow全局参数初始化器tf.golbal_variables_initializer。

batch_xs,batch_ys = mnist.train.next_batch(100)。训练操作train_step。每次随机从训练集抽取100条样本构成mini-batch，feed给 placeholder，调用train_step训练样本。使用小部分样本训练，随机梯度下降，收敛速度更快。每次训练全部样本，计算量大，不容易跳出局部最优。

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmzx(y_,1))，验证模型准确率。tf.argmax从tensor寻找最大值序号，tf.argmax(y,1)求预测数字概率最大，tf.argmax(y_,1)找样本真实数字类别。tf.equal判断预测数字类别是否正确，返回计算分类操作是否正确。

accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))，统计全部样本预测正确度。tf.cast转化correct_prediction输出值类型。

print(accuracy.eval({x: mnist.test.images,y_: mnist.test.labels}))。测试数据特征、Label输入评测流程，计算模型测试集准确率。Softmax Regression  MNIST数据分类识别，测试集平均准确率92%左右。

TensorFlow 实现简单机器算法步骤：
1､定义算法公式，神经网络forward计算。
2､定义loss，选定优化器，指定优化器优化loss。
3､迭代训练数据。
4､测试集、验证集评测准确率。

定义公式只是Computation Graph，只有调用run方法，feed数据，计算才执行。

    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    print(mnist.train.images.shape, mnist.train.labels.shape)
    print(mnist.test.images.shape, mnist.test.labels.shape)
    print(mnist.validation.images.shape, mnist.validation.labels.shape)
    import tensorflow as tf
    sess = tf.InteractiveSession()
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    y_ = tf.placeholder(tf.float32, [None, 10])
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    tf.global_variables_initializer().run()
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        train_step.run({x: batch_xs, y_: batch_ys})
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))


参考资料：
《TensorFlow实战》


