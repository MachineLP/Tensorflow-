回答多选项问题，使用softmax函数，对数几率回归在多个可能不同值上的推广。函数返回值是C个分量的概率向量，每个分量对应一个输出类别概率。分量为概率，C个分量和始终为1。每个样本必须属于某个输出类别，所有可能样本均被覆盖。分量和小于1,存在隐藏类别；分量和大于1,每个样本可能同时属于多个类别。类别数量为2,输出概率与对数几率回归模型输出相同。

变量初始化，需要C个不同权值组，每个组对应一个可能输出，使用权值矩阵。每行与输入特征对应，每列与输出类别对应。

鸢尾花数据集Iris，包含4个数据特征、3类可能输出，权值矩阵4X3。

训练样本每个输出类别损失相加。训练样本期望类别为1,其他为0。只有一个损失值被计入，度量模型为真实类别预测的概率可信度。每个训练样本损失相加，得到训练集总损失值。TensorFlow的softmax交叉熵函数，sparse_softmax_cross_entropy_with_logits版本针对训练集每个样本只对应单个类别优化，softmax_cross_entropy_with_logits版本可使用包含每个样本属于每个类别的概率信息的训练集。模型最终输出是单个类别值。

不需要每个类别都转换独立变量，需要把值转换为0~2整数(总类别数3)。tf.stack创建张量，tf.equal把文件输入与每个可能值比较。tf.argmax找到张量值为真的位置。

推断过程计算测试样本属于每个类别概率。tf. argmax函数选择预测输出值最大概率类别。tf.equal与期望类别比较。tf.reduce_meen计算准确率。


    import tensorflow as tf#导入TensorFlow库
    import os#导入OS库
    W = tf.Variable(tf.zeros([4, 3]), name="weights")#变量权值，矩阵，每个特征权值列对应一个输出类别
    b = tf.Variable(tf.zeros([3], name="bias"))#模型偏置，每个偏置对应一个输出类别
    def combine_inputs(X):#输入值合并
        print "function: combine_inputs"
        return tf.matmul(X, W) + b
    def inference(X):#计算返回推断模型输出(数据X)
        print "function: inference"
        return tf.nn.softmax(combine_inputs(X))#调用softmax分类函数
    def loss(X, Y):#计算损失(训练数据X及期望输出Y)
        print "function: loss"
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=combine_inputs(X), labels=Y))#求平均值，针对每个样本只对应单个类别优化
    def read_csv(batch_size, file_name, record_defaults):#从csv文件读取数据，加载解析，创建批次读取张量多行数据
        filename_queue = tf.train.string_input_producer([os.path.join(os.getcwd(), file_name)])
        reader = tf.TextLineReader(skip_header_lines=1)
        key, value = reader.read(filename_queue)
        decoded = tf.decode_csv(value, record_defaults=record_defaults)#字符串(文本行)转换到指定默认值张量列元组，为每列设置数据类型
        return tf.train.shuffle_batch(decoded, batch_size=batch_size, capacity=batch_size * 50, min_after_dequeue=batch_size)#读取文件，加载张量batch_size行
    def inputs():#读取或生成训练数据X及期望输出Y
        print "function: inputs"
        #数据来源：https://archive.ics.uci.edu/ml/datasets/Iris
        #iris.data改为iris.csv，增加sepal_length, sepal_width, petal_length, petal_width, label字段行首行
        sepal_length, sepal_width, petal_length, petal_width, label =\
            read_csv(100, "iris.csv", [[0.0], [0.0], [0.0], [0.0], [""]])
        #转换属性数据
        label_number = tf.to_int32(tf.argmax(tf.to_int32(tf.stack([
            tf.equal(label, ["Iris-setosa"]),
            tf.equal(label, ["Iris-versicolor"]),
            tf.equal(label, ["Iris-virginica"])
        ])), 0))#将类名称转抽象为从0开始的类别索引
        features = tf.transpose(tf.stack([sepal_length, sepal_width, petal_length, petal_width]))#特征装入矩阵，转置，每行一样本，每列一特征
        return features, label_number
    def train(total_loss):#训练或调整模型参数(计算总损失)
        print "function: train"
        learning_rate = 0.01
        return tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)
    def evaluate(sess, X, Y):#评估训练模型
        print "function: evaluate"
        predicted = tf.cast(tf.arg_max(inference(X), 1), tf.int32)#选择预测输出值最大概率类别
        print sess.run(tf.reduce_mean(tf.cast(tf.equal(predicted, Y), tf.float32)))#统计所有正确预测样本数，除以批次样本总数，得到正确预测百分比
    with tf.Session() as sess:#会话对象启动数据流图，搭建流程
        print "Session: start"
        tf.global_variables_initializer().run()
        X, Y = inputs()
        total_loss = loss(X, Y)
        train_op = train(total_loss)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        training_steps = 1000#实际训练迭代次数
        for step in range(training_steps):#实际训练闭环
            sess.run([train_op])
            if step % 10 == 0:#查看训练过程损失递减
                print str(step)+ " loss: ", sess.run([total_loss])
        print str(training_steps) + " final loss: ", sess.run([total_loss])
        evaluate(sess, X, Y)#模型评估
        coord.request_stop()
        coord.join(threads)
        sess.close()


参考资料：
《面向机器智能的TensorFlow实践》


