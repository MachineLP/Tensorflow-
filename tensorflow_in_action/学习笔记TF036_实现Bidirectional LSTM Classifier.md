双向循环神经网络(Bidirectional Recurrent Neural Networks,Bi-RNN)，Schuster、Paliwal，1997年首次提出，和LSTM同年。Bi-RNN，增加RNN可利用信息。普通MLP，数据长度有限制。RNN，可以处理不固定长度时序数据，无法利用历史输入未来信息。Bi-RNN，同时使用时序数据输入历史及未来数据，时序相反两个循环神经网络连接同一输出，输出层可以同时获取历史未来信息。

Language Modeling，不适合Bi-RNN，目标是通过前文预测下一单词，不能将下文信息传给模型。分类问题，手写文字识别、机器翻译、蛋白结构预测，Bi-RNN提升模型效果。百度语音识别，通过Bi-RNN综合上下文语境，提升模型准确率。

Bi-RNN网络结构核心，普通单向RNN拆成两个方向，随时序正向，逆时序反赂。当前时间节点输出，同时利用正向、反向两个方向信息。两个不同方向RNN不共用state，正向RNN输出state只传给正向RNN，反向RNN输出state只传给反向RNN，正反向RNN没有直接连接。每个时间节点输入，分别传给正反向RNN，根据各自状态产生输出，两份输出一起连接到Bi-RNN输出节点，共同合成最终输出。对当前时间节点输出贡献(或loss)，在训练中计算出来，参数根据梯度优化到合适值。

Bi-RNN训练，正反向RNN没有交集，分别展开普通前馈网络。BPTT(back-propagation through time)算法训练，无法同时更新状态、输出。正向state在t=1时未知，反向state在t=T时未知，state在正反向开始处未知，需人工设置。正向状态导数在t=T时未知，反向状态导数在t=1时未知，state导数在正反向结晶尾处未知，需设0代表参数更新不重要。

开始训练，第一步，输入数据forward pass操作，inference操作，先沿1->T方向计算正向RNN state，再沿T->1方向计算反向RNN state，获得输出output。第二步，backward pass操作，目标函数求导操作，先求导输出output，先沿T->1方向计算正向RNN state导数，再沿1->T方向计算反向RNN state导数。第三步，根据求得梯度值更新模型参数，完成训练。

Bi-RNN每个RNN单元，可以是传统RNN，可以是LSTM或GRU单元。可以在一层Bi-RNN上再叠加一层Bi-RNN，上层Bi-RNN输出作下层Bi-RNN输入，可以进一步抽象提炼特征。分类任务，Bi-RNN输出序列连接全连接层，或连接全局平均池化Global Average Pooling，再接Softmax层，和卷积网络一样。

TensorFlow实现Bidirectional LSTM Classifier，在MNIST数据集测试。载入TensorFlow、NumPy、TensorFlow自带MNIST数据读取器。input_data.read_data_sets下载读取MNIST数据集。

设置训练参数。设置学习速率 0.01,优化器选择Adam，学习速率低。最大训练样本数 40万，batch_size 128,设置每间隔10次训练展示训练情况。

MNIST图像尺寸 28x28,输入n_input 28(图像宽)，n_steps LSTM展开步数(unrolled steps of LSTM)，设28(图像高)，图像全部信息用上。一次读取一行像素(28个像素点)，下个时间点再传入下一行像素点。n_hidden(LSTM隐藏节点数)设256,n_classes(MNIST数据集分类数目)设10。

创建输入x和学习目标y 的place_holder。输入x每个样本直接用二维结构。样本为一个时间序列，第一维度 时间点n_steps，第二维度 每个时间点数据n_input。设置Softmax层weights和biases，tf.random_normal初始化参数。双向LSTM，forward、backward两个LSTM cell，weights参数数量翻倍，2*n_hidden。

定义Bidirectional LSTM网络生成函数。形状(batch_size,n_steps,n_input)输入变长度n_steps列表，元素形状(batch_size,n_input)。输入转置，tf.transpose(x,[1,0,2])，第一维度batch_size，第二维度n_steps，交换。tf.reshape，输入x变(n_steps*batch_size,n_input)形状。 tf.split，x拆成长度n_steps列表，列表每个tensor尺寸(batch_size,n_input)，符合LSTM单元输入格式。tf.contrib.rnn.BasicLSTMCell，创建forward、backward LSTM单元，隐藏节点数设n_hidden，forget_bias设1。正向lstm_fw_cell和反向lstm_bw_cell传入Bi-RNN接口tf.nn.bidirectional_rnn，生成双向LSTM，传入x输入。双向LSTM输出结果output做矩阵乘法加偏置，参数为前面定义weights、biases。

最后输出结果，tf.nn.softmax_cross_entropy_with_logits，Softmax处理计算损失。tf.reduce_mean计算平均cost。优化器Adam，学习速率learning_rate。tf.argmax得到模型预测类别，tf.equal判断是否预测正确。tf.reduce_mean求平均准确率。

执行训练和测试操作。执行初始化参数，定义一个训练循环，保持总训练样本数(迭代数*batch_size)小于设定值。每轮训练迭代，mnist.train.next_batch拿到一个batch数据，reshape改变形状。包含输入x和训练目标y的feed_dict传入，执行训练操作，更新模型参数。迭代数display_step整数倍，计算当前batch数据预测准确率、loss，展示。

全部训练迭代结果，训练好模型，mnist.test.images全部测试数据预测，展示准确率。

完成40万样本训练，训练集预测准确率基本是1,10000样本测试集0.983准确率。

Bidirectional LSTM Classifier，MNIST数据集表现不如卷积神经网络。Bi-RNN、双向LSTM网络，时间序列分类任务表现更好，同时利用时间序列历史和未来信息，结合上下文信息，结果综合判断。


    import tensorflow as tf
    import numpy as np
    # Import MINST data
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
    # Parameters
    learning_rate = 0.01
    max_samples = 400000
    batch_size = 128
    display_step = 10
    # Network Parameters
    n_input = 28 # MNIST data input (img shape: 28*28)
    n_steps = 28 # timesteps
    n_hidden = 256 # hidden layer num of features
    n_classes = 10 # MNIST total classes (0-9 digits)
    # tf Graph input
    x = tf.placeholder("float", [None, n_steps, n_input])
    y = tf.placeholder("float", [None, n_classes])
    # Define weights
    weights = {
        # Hidden layer weights => 2*n_hidden because of foward + backward cells
        'out': tf.Variable(tf.random_normal([2*n_hidden, n_classes]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([n_classes]))
    }
    def BiRNN(x, weights, biases):
        # Prepare data shape to match `bidirectional_rnn` function requirements
        # Current data input shape: (batch_size, n_steps, n_input)
        # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
    
        # Permuting batch_size and n_steps
        x = tf.transpose(x, [1, 0, 2])
        # Reshape to (n_steps*batch_size, n_input)
        x = tf.reshape(x, [-1, n_input])
        # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        x = tf.split(x, n_steps)
        # Define lstm cells with tensorflow
        # Forward direction cell
        lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
        # Backward direction cell
        lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
        # Get lstm cell output
    #    try:
        outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                           dtype=tf.float32)
    #    except Exception: # Old TensorFlow version only returns outputs not states
    #        outputs = rnn.bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
    #                                        dtype=tf.float32)
        # Linear activation, using rnn inner loop last output
        return tf.matmul(outputs[-1], weights['out']) + biases['out']
    
    pred = BiRNN(x, weights, biases)
    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    # Evaluate model
    correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    # Initializing the variables
    init = tf.global_variables_initializer()
    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)
        step = 1
        # Keep training until reach max iterations
        while step * batch_size < max_samples:
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Reshape data to get 28 seq of 28 elements
            batch_x = batch_x.reshape((batch_size, n_steps, n_input))
            # Run optimization op (backprop)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
            if step % display_step == 0:
                # Calculate batch accuracy
                acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
                # Calculate batch loss
                loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
                print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))
            step += 1
        print("Optimization Finished!")
        # Calculate accuracy for 128 mnist test images
        test_len = 10000
        test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
        test_label = mnist.test.labels[:test_len]
        print("Testing Accuracy:", \
            sess.run(accuracy, feed_dict={x: test_data, y: test_label}))

参考资料：
《TensorFlow实战》


