首先向大家和《TensorFlow实战》的作者说句不好意思。我现在看的书是《TensorFlow实战》。但从TF024开始，我在学习笔记的参考资料里一直写的是《TensorFlow实践》，我自己粗心搞错了，希望不至于对大家造成太多误导。

TensorBoard，TensorFlow官方可视化工具。展示模型训练过程各种汇总数据。标量(Scalars)、图片(Images)、音频(audio)、计算图(Graphs)、数据分布(Distributions)、直方图(Histograms)、嵌入向量(Embeddings)。TensorBoard展示数据，执行TensorFlow计算图过程，各种类型数据汇总并记录到日志文件。TensorBoard读取日志文件，解析数据，生成数据可视化Web页，浏览器观察各种汇总数据。

载入TesnsorFlow，设置训练最大步数1000,学习速率0.001，dropout保留比率0.9。设置MNIST数据下载地址data_dir、汇总数据日志存放路径log_dir。日志路径log_dir存所有汇总数据。

input_data.read_data_sets下载MNIST数据，创建TensorFlow默认Session。

with tf.name_scope限定命名空间。定义输入x、y placeholder。输入一维数据变形28x28图片储存到tensor，tf.summary.image汇总图片数据TensorBoard展示。

定义神经网络模型参数初始化方法，权重用truncated_normal初始化，偏置赋值0.1。

定义Variable变量数据汇总函数，计算Variable mean、stddev、max、min，tf.summary.scalar记录、汇总。tf.summary.histogram记录变量var直方图数据。

设计MLP多层神经网络训练数据，每一层汇总模型参数数据。定义创建一层神经网络数据汇总函数nn_layer。输入参数，输入数据input_tensor、输入维度input_dim、输出维度output_dim、层名称layer_name。激活函数act用ReLU。初始化神经网络权重、偏置，用variable_summaries汇总variable数据。输入，矩阵乘法，加偏置，未激活结果用tf.summary.histogram统计直方图。用激活函数后，tf.summary.histogram再统计一次。

nn_layer创建一层神经网络，输入维度图片尺寸28x28=784，输出维度隐藏节点数500。创建Dropout层，用tf.summary.scalar记录keep_prob。用nn_layer定义神经网络输出层，输入维度为上层隐含节点数500,输出维度类别数10,激活涵数全等映射identity。

tf.nn.softmax_cross_entropy_with_logits()对前面输出层结果Softmax处理，计算交叉熵损失cross_entropy。计算平均损失，tf.summary.scalar统计汇总。

Adma优化器优化损失。统计预测正确样本数，计算正确率accury， tf.summary.scalar统计汇总accuracy。

tf.summary.merger_all()获取所有汇总操作。定义两个tf.summary.FileWriter(文件记录器)在不同子目录，分别存放训练和测试日志数据。Session计算图sess.graph加入训练过程记录器，TensorBoard GRAPHS窗口展示整个计算图可视化效果。tf.global_variables_initializer().run()初始化全部变量。

定义feed_dict损失函数。先判断训练标记，True，从mnist.train获取一个batch样本，设置dropout值；False，获取测试数据，设置keep_prob 1,没有dropout效果。

实际执行具体训练、测试、日志记录操作。tf.train.Saver()创建模型保存器。进入训练循环，每隔10步执行merged(数据汇总)、accuracy(求测试集预测准确率)操作，test_writer.add_sumamry将汇总结果summary和循环步数i写入日志文件。每隔100步，tf.RunOptions定义TensorFlow运行选项，设置trace_lever FULL_TRACE。tf.RunMetadata()定义TensorFlow运行元信息，记录训练运算时间和内存占用等信息。执行merged数据汇总操作，train_step训练操作，汇总结果summary、训练元信息run_metadata添加到train_writer。执行merged、train_step操作，添加summary到train_writer。所有训练全部结束，关闭train_writer、test_writer。

切换Linux命令行，执行TensorBoard程序，--logdir指定TensorFlow日志路径，TensorBoard自动生成所有汇总数据可视化结果。
tensorboard --logdir=/tmp/tensorflow/mnist/logs/mnist_with_summaries
复制网址到浏览器。

打开标量SCALARS窗口，打开accuracy图表。调整Smoothing参数，控制曲线平滑处理，数值越小越接近实际值，波动大；数值越大曲线越平缓。图表下方按钮放大图片，右边按钮调整坐标轴范围。

切换图像IMAGES窗口，可以看到所有tf.summary.image()汇总数据。

计算图GRAPHS窗口，整个TensorFlow计算图结构。网络forward inference流程，backward训练更新参数流程。实线代表数据依赖关系，虚线代表控制条件依赖关系。节点窗口，看属性、输入、输出及tensor尺寸。

"+"按钮，展示node内部细节。所有同一命名空间节点被折叠一起。右键单击节点选择删除。

切换配色风络，基于结构，同结构节点同颜色；基于运算硬件，同运算硬件节点同颜色。

Session runs，选择run_metadata训练元信息。

切换DISTRIBUTIONS窗口，看各个神经网络层输出分布，激活函数前后结果。看看有没有被屏蔽节点(dead neurons)。转为直方图。

EMBEDDINGS窗口，降维嵌入向量可视化效果。tf.save.Saver保存整个模型，TensorBoard自动对模型所有二维Variable可视化(只有Variable可以被保存，Tensor不行)。选择T-SNE或PCA算法对数据列(特征)降维，在3D、2D坐标可视化展示。对Word2Vec计算或Language Model非常有用。


    import tensorflow as tf
    from tensorflow.examples.tutorials.mnist import input_data
    max_steps=1000
    learning_rate=0.001
    dropout=0.9
    data_dir='/tmp/tensorflow/mnist/input_data'
    log_dir='/tmp/tensorflow/mnist/logs/mnist_with_summaries'
      # Import data
    mnist = input_data.read_data_sets(data_dir,one_hot=True)
    sess = tf.InteractiveSession()
      # Create a multilayer model.
      # Input placeholders
    with tf.name_scope('input'):
      x = tf.placeholder(tf.float32, [None, 784], name='x-input')
      y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')
    with tf.name_scope('input_reshape'):
      image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
      tf.summary.image('input', image_shaped_input, 10)
      # We can't initialize these variables to 0 - the network will get stuck.
    def weight_variable(shape):
      """Create a weight variable with appropriate initialization."""
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial)
    def bias_variable(shape):
      """Create a bias variable with appropriate initialization."""
      initial = tf.constant(0.1, shape=shape)
      return tf.Variable(initial)
    def variable_summaries(var):
      """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
      with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
          stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)
    def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
      """Reusable code for making a simple neural net layer.
      It does a matrix multiply, bias add, and then uses relu to nonlinearize.
      It also sets up name scoping so that the resultant graph is easy to read,
      and adds a number of summary ops.
      """
      # Adding a name scope ensures logical grouping of the layers in the graph.
      with tf.name_scope(layer_name):
        # This Variable will hold the state of the weights for the layer
        with tf.name_scope('weights'):
          weights = weight_variable([input_dim, output_dim])
          variable_summaries(weights)
        with tf.name_scope('biases'):
          biases = bias_variable([output_dim])
          variable_summaries(biases)
        with tf.name_scope('Wx_plus_b'):
          preactivate = tf.matmul(input_tensor, weights) + biases
          tf.summary.histogram('pre_activations', preactivate)
        activations = act(preactivate, name='activation')
        tf.summary.histogram('activations', activations)
        return activations
    hidden1 = nn_layer(x, 784, 500, 'layer1')
    with tf.name_scope('dropout'):
      keep_prob = tf.placeholder(tf.float32)
      tf.summary.scalar('dropout_keep_probability', keep_prob)
      dropped = tf.nn.dropout(hidden1, keep_prob)
      # Do not apply softmax activation yet, see below.
    y = nn_layer(dropped, 500, 10, 'layer2', act=tf.identity)
    with tf.name_scope('cross_entropy'):
        # The raw formulation of cross-entropy,
        #
        # tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.softmax(y)),
        #                               reduction_indices=[1]))
        #
        # can be numerically unstable.
        #
        # So here we use tf.nn.softmax_cross_entropy_with_logits on the
        # raw outputs of the nn_layer above, and then average across
        # the batch.
      diff = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_)
      with tf.name_scope('total'):
        cross_entropy = tf.reduce_mean(diff)
    tf.summary.scalar('cross_entropy', cross_entropy)
    with tf.name_scope('train'):
      train_step = tf.train.AdamOptimizer(learning_rate).minimize(
      cross_entropy)
    with tf.name_scope('accuracy'):
      with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
      with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)
      # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(log_dir + '/test')
    tf.global_variables_initializer().run()
      # Train the model, and also write summaries.
      # Every 10th step, measure test-set accuracy, and write test summaries
      # All other steps, run train_step on training data, & add training summaries
    def feed_dict(train):
      """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
      if train:
        xs, ys = mnist.train.next_batch(100)
        k = dropout
      else:
        xs, ys = mnist.test.images, mnist.test.labels
        k = 1.0
      return {x: xs, y_: ys, keep_prob: k}
  
    saver = tf.train.Saver()  
    for i in range(max_steps):
      if i % 10 == 0:  # Record summaries and test-set accuracy
        summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
        test_writer.add_summary(summary, i)
        print('Accuracy at step %s: %s' % (i, acc))
      else:  # Record train set summaries, and train
        if i % 100 == 99:  # Record execution stats
          run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
          run_metadata = tf.RunMetadata()
          summary, _ = sess.run([merged, train_step],
                                feed_dict=feed_dict(True),
                                options=run_options,
                                run_metadata=run_metadata)
          train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
          train_writer.add_summary(summary, i)
          saver.save(sess, log_dir+"/model.ckpt", i)
          print('Adding run metadata for', i)
        else:  # Record a summary
          summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
          train_writer.add_summary(summary, i)
    train_writer.close()
    test_writer.close()

参考资料：
《TensorFlow实战》


