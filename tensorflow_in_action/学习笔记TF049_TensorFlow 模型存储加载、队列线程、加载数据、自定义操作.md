生成检查点文件(chekpoint file)，扩展名.ckpt，tf.train.Saver对象调用Saver.save()生成。包含权重和其他程序定义变量，不包含图结构。另一程序使用，需要重新创建图形结构，告诉TensorFlow如何处理权重。
生成图协议文件(graph proto file)，二进制文件，扩展名.pb，tf.tran.write_graph()保存，只包含图形结构，不包含权重，tf.import_graph_def加载图形。

模型存储，建立一个tf.train.Saver()保存变量，指定保存位置，扩展名.ckpt。

神经网络，两个全连接层和一个输出层，训练MNIST数据集，存储训练好的模型。
加载数据、定义模型：

    #加载数据
    mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
    trX,trY,teX,teY = mnist.train.images,mnist.train.labels,mnist.test.images,mnist.test.labels
    X = tf.placeholder("float",[None,784])
    Y = tf.placeholder("float",[None,10])
    #初始化权重参数
    w_h = init_weights([784,625])
    w_h2 = init_weights([625,625])
    w_o = int_weights([625,10])

    #定义权重函数
    def init_weights(shape):
        return tf.Variable(tf.random_normal(shape,stddev=0.01))

    #定义模型
    def model(X,w_h,w_h2,w_o,p_keep_input,p_keep_hidden):
        #第一个全连接层
        X = tf.nn.dropout(X,p_keep_input)
        h = tf.nn.relu(tf.matmul(X,w_h))
    
        h = tf.nn.dropout(h,p_keep,hidden)
        #第二个全连接层
        h2 = tf.nn.relu(tf.matmul(h,w_h2))
        h2 = tf.nn.dropout(h2,p_keep_hidden)

        return tf.matmul(h2,w_o)#输出预测值

    #定义损失函数
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x,Y))
    train_op = tf.train.RMSPropOptimizer(0.001,0.9).minimize(cost)
    predict_op = tf.argmax(py_x,1)

训练模型及存储模型：

    #定义存储路径
    ckpt_dir = "./ckpt_dir"
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

定义计数器，为训练轮数计数：

    #计数器变量，设置它的trainable=False，不需要被训练
    global_step = tf.Variable(0,name='global_step',trainable=False)

定义完所有变量，tf.train.Saver()保存、提取变量，后面定义变量不被存储：

    #声明完所有变量，调tf.train.Saver
    saver = tf.train.Saver()
    #之后变量不被存储
    non_storable_variable = tf.Variable(777)

训练模型并存储：

    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        start = global_step.eval() #得到global_stepp初始值
        print("Start from:",start)
        for i in range(start,100):
            #128 batch_size
            for start,end in zip(range(0,len(trX),128),range(128,len(trX)+1,128)):
                sess.run(train_op,feed_dict={X:trX[start:end],Y:trY[start:end],p_keep_input:0.8, p_keep_hidden:0.5})
            global_step.assign(i).eval() #更新计数器
            saver.save(sess,ckpt_dir+"/model.ckpt",global_step=global_step) #存储模型

训练过程，ckpt_dir出现16个文件，5个model.ckpt-{n}.data00000-of-00001文件，保存的模型。5个model.ckpt-{n}.meta文件，保存的元数据。TensorFlow默认只保存最近5个模型和元数据，删除前面没用模型和元数据。5个model.ckpt-{n}.index文件。{n}代表迭代次数。1个检查点文本文件，保存当前模型和最近5个模型。

将之前训练参数保存下来，在出现意外状竞接着上一次地方开始训练。每个固定轮数在检查点保存一个模型(.ckpt文件)，随时将模型拿出来预测。

加载模型。
saver.restore加载模型：

    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print(ckpt.model_checkpoint_path)
            saver.restore(sess,ckpt.model_checkpoint_path) #加载所有参数

图存储加载。
仅保存图模型，图写入二进制协议文件：

    v = tf.Variable(0,name='my_variable')
    sess = tf.Session()
    tf.train.write_graph(sess.graph_def,'/tmp/tfmodel','train.pbtxt')
读取:

    with tf.Session() as _sess:
        with grile.FastGFile("/tmp/tfmodel/train.pbtxt",'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _sess.graph.as_default()
            tf.import_graph_def(graph_def,name='tfgraph')

队列、线程。
队列(queue)，图节点，有状态节点。其他节点(入队节点enqueue,出队节点depueue)，修改它内容。入队节点把新元素插到队列末尾，出队节点把队列前面元素删除。
FIFOQueue、RandomShuffleQueue，源代码在tensorflow-1.0.0/tensorflow/python/ops/data_flow_ops.py 。
FIFOQueue，创建先入先出队列。循环神经网络结构，读入训练样本需要有序。
创建含队列图：

    import tensorflow as tf
    #创建先入先出队列，初始化队列插入0.1、0.2、0.3数字
    q = tf.FIFOQueue(3,"float")
    init = q.enqueue_many(([0.1,0.2,0.3]))
    #定义出队、+1、入队操作
    x = q.dequeue()
    y = x +1
    q_inc = q.enqueue([y])

开户会话，执行2次q_inc操作，查看队列内容：

    with tf.Session() as sess:
        sess.run(init)
        quelen = sess.run(q.size())
        for i in range(2):
            sess.run(q_inc) #执行2次操作，队列值变0.3,1.1,1.2
        quelen = sess.run(q.size())
        for i in range(quelen):
            print(sess.run(q.dequeue())) #输出队列值

RandomShuffleQueue，创建随机队列。出队列，以随机顺序产生元素。训练图像样本，CNN网络结构，无序读入训练样本，每次随机产生一个训练样本。
异步计算时非常重要。TensorFlow会话支持多线程，在主线程训练操作，RandomShuffleQueue作训练输入，开多个线程准备训练样本。样本压入队列，主线程从队列每次取出mini-batch样本训练。
创建随机队列，最大长度10,出队后最小长度2:

    q = tf.RandomShuffleQueue(capacity=10,min_after_dequeue=2,dtype="float")
开户会话，执行10次入队操作，8次出队操作:

    sess = tf.Session()
    for i in range(0,10): #10次入队
        sess.run(q.enqueue(i))
    for i in range(0,8): #8次出队
        print(sess.run(q.dequeue()))

阻断，队列长度等于最小值，执行出队操作；队列长度等于最大值，执行入队操作。
设置绘画运行时等待时间解除阻断：

    run_iptions = tf.RunOptions(timeout_in_ms = 10000) #等待10秒
    try:
        sess.run(q.dequeue(),options=run_options)
    except tf.errors.DeadlineExceededError:
        print('out of range')

会话主线程入队操作，数据量很大时，入队操作从硬盘读取数据，放入内存，主线程需要等待入队操作完成，才能进行训练操作。会话运行多个线程，线程管理器QueueRunner创建一系列新线程进行入队操作，主线程继续使用数据，训练网张和读取数据是异步的，主线程训练网络，另一线程将数据从硬盘读入内存。

队列管理器。
创建含队列图:

    q = tf.FIFOQueue(1000,"float")
    counter = tf.Variable(0.0) #计数器
    increment_op = tf.assign_add(counter,tf.constant(1.0)) #操作:给计算器加1
    enqueue_op = q.enqueue(counter) #操作:计数器值加入队列
创建队列管理器QueueRunner，两个操作向队列q添加元素。只用一个线程：

    qr = tf.train.QueueRunner(q,enqueue_ops=[increment_op,enqueue_op] * 1)
启动会话，从队列管理器qr创建线程：

    #主线程
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        enqueue_threads = qr.create_threads(sess,start=True) #启动入队线程
        #主线程
        for i in range(10):
            print(sess.run(q.dequeue()))

不是自然数列，线程被阴断。加1操作和入队操作不同步。加1操作执行多次后，才进行一次入队操作。主线程训练(出队操作)和读取数据线程的训练(入队操作)异步，主线程一直等待数据送入。
入队线程自顾执行，出队操作完成，程序无法结束。tf.train.Coordinator实现线程同步，终止其他线程。

线程、协调器。
协调器(coordinator)管理线程。

    #主线程
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    #Coordinator:协调器，协调线程间关系可以视为一种信号量，用来做同步
    coord = tf.train.Coordinator()
    #启动入队线程，协调器是线程的参数
    enqueue_threads = qr.create_threads(sess,coord = coord,start=True)
    #主线程
    for i in range(0,10):
        print(sess.run(q.dequeue()))
    coord.request_stop() #通知其他线程关闭
    coord.join(enqueue_threads) #join操作等待其他线程结束，其他所有线程关闭之后，函数才能返回

关闭队列线程，执行出队操作，抛出tf.errors.OutOfRange错误。coord.request_stop()和主线程出队操作q.dequeue()调换位置。

    coord.request_stop()
    #主线程
    for i in range(0,10):
        print(sess.run(q.dequeue()))
    coord.join(enqueue_threads)
    tf.errors.OutOfRangeError捕捉错误：
    coord.request_stop()
    #主线程
    for i in range(0,10):
        try:
            print(sess.run(q.dequeue()))
        except tf.errors.OutOfRangeError:
            break
    coord.join(enqueue_threads)
所有队列管理器默认加在图tf.GraphKeys.QUEUE_RENNERS集合。

加载数据。
预加载数据(preloaded data)，在TensorFlow图中定义常量或变量保存所有数据。填充数据(feeding)，Python产生数据，数据填充后端。从文件读取数据(reading from file)，队列管理器从文件中读取数据。

预加载数据。数据直接嵌在数据流图中，训练数据大时，很消耗内存。

    x1 = tf.constant([2,3,4])
    x2 = tf.constant([4,0,1])
    y = tf.add(x1,x2)

填充数据。sess.run()中的feed_dict参数，Python产生数据填充后端。数据量大、消耗内存，数据类型转换等中间环节增加开销。

    import tensorflow as tf
    #设计图
    a1 = tf.placeholder(tf.int16)
    a2 = tf.placeholder(tf.int16)
    b = tf.add(x1,x2)
    #用Python产生数据
    li1 = [2,3,4]
    li2 = [4,0,1]
    #打开会话，数据填充后端
    with tf.Session() as sess:
        print sess.run(b,feed_dict={a1:li1,a2:li2})

从文件读取数据。图中定义好文件读取方法，TensorFlow从文件读取数据，解码成可使用样本集。
把样本数据写入TFRecords二进制文件。再从队列读取。
TFRecords二进制文件，更好利用内存，方便复制和移动，不需要单独标记文件。tensorflow-1.1.0/tensorflow/examples/how_tos/reading_data/convert_to_records.py。
生成TFRecords文件。定义主函数，给训练、验证、测试数据集做转换。获取数据。编码uint8。数据转换为tf.train.Example类型，写入TFRecords文件。转换函数convert_to，数据填入tf.train.Example协议缓冲区(protocol buffer)，协议组冲区序列转为字符串，tf.python_io.TFRecordWriter写入TFRecords文件。55000个训练数据，5000个验证数据,10000个测试数据。黑白图像，单通道。写入协议缓冲区，height、width、depth、label编码int64类型，image_raw编码成二进制。序列化为字符串。运行结束，在/tmp/data生成文件train.tfrecords、validation.tfrecords、test.tfrecords。
从队列读取。创建张量，从二进制文件读取一个样本。创建张量，从二进制文件随机读取一个mini-batch。把每一批张量传入网络作为输入节点。代码 tensorflow-1.1.0/tensorflow/examples/how_tos/reading_data/fully_connected_reader.py。首先定义从文件中读取并解析一个样本。输入文件名队列。解析example。写明features里key名称。图片 string类型。标记 int64类型。BytesList 重新解码，把string类型0维Tensor变成unit8类型一维Tensor。image张量形状Tensor("input/sub:0",shape=(784,),dtype=float32)。把标记从uint8类型转int32类型。label张量形状Tensor("input/Cast_1:0",shape=(),dtype=int32)。
tf.train.shuffle_batch样本随机化，获得最小批次张量。输入参数，train 选择输入训练数据/验证数据，batch_size 训练每批样本数，num_epochs 过几遍数据 设置0/None表示永远训练下去。返回结果，images，类型float, 形状[batch_size,mnist.IMAGE_PIXELS]，范围[-0.5,0.5]。labels,类型int32,形状[batch_size],范围[0,mnist.NUM_CLASSES]。tf.train.QueueRunner用tf.train.start_queue_runners()启动线程。获取文件路径，/tmp/data/train.tfrecords, /tmp/data/validation.records。tf.train.string_input_producer返回一个QueueRunner，里面有一个FIFOQueue。如果样本量很大，分成若干文件，文件名列表传入。随机化example,规整成batch_size大小。留下部分队列，保证每次有足够数据做随机打乱。
生成batch张量作网络输入，训练。输入images、labels。构建从揄模型预测数据的图。定义损失函数。加入图操作，训练模型。初始化参数，string_input_producer内疗创建一个epoch计数变量，归入tf.GraphKeys.LOCAL_VARIABLES集合，单独用initialize_local_variables()初始化。开启输入入阶线程。进入永久循环。每100次训练输出一次结果。通知其他线程关闭。数据集大小55000,2轮训练，110000个数据，batch_size大小100,训练次数1100次，每100次训练输出一次结果，输出11次结果。
TensorFlow使用TFRecords文件训练样本步骤：在生成文件名队列中，设定epoch数量。训练时，设定为无穷循环。在读取数据时，如果捕捉到错误，终止。

实现自定义操作。
需要熟练掌握C++语言。对张量流动和前向传播、反向传播有深理解。
步骤。在C++文件(*_ops.cc文件)中注册新操作。定义操作功能接口规范，操作名称、输入、输出、属性等。在C++文件(*_kenels.cc文件)实现操作，可以实现在CPU、GPU多个内核上。测试操作，编译操作库文件(*_ops.so文件)，Python使用操作。
最佳实践。词嵌入。源代码 https://github.com/tensorflow/models/blob/master/tutorials/embedding/word2vec_optimized.py 。
第一步，创建word2vec_ops.cc注册两个操作，SkipgramWord2vec、NegTrainWord2vec。
第二步，两个操作在CPU设备实现，生成word2vec_kernels.cc文件。
第三步，编译操作类文件，测试。在特定头文件目录下编译，Python提供get_include获取头文件目录，C++编译器把操作编译成动态库。TensorFlow Python API提供tf.load_op_library函数加载动态库，向TensorFlow 框架注册操作。load_op_library返回包含操作和内核Python模块。

参考资料：
《TensorFlow技术解析与实战》


