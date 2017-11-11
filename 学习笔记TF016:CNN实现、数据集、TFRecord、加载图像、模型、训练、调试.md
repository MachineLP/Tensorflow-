AlexNet(Alex Krizhevsky,ILSVRC2012冠军)适合做图像分类。层自左向右、自上向下读取，关联层分为一组，高度、宽度减小，深度增加。深度增加减少网络计算量。

训练模型数据集 Stanford计算机视觉站点Stanford Dogs http://vision.stanford.edu/aditya86/ImageNetDogs/ 。数据下载解压到模型代码同一路径imagenet-dogs目录下。包含的120种狗图像。80%训练，20%测试。产品模型需要预留原始数据交叉验证。每幅图像JPEG格式(RGB)，尺寸不一。

图像转TFRecord文件，有助加速训练，简化图像标签匹配，图像分离利用检查点文件对模型进行不间断测试。转换图像格式把颜色空间转灰度，图像修改统一尺寸，标签除上每幅图像。训练前只进行一次预处理，时间较长。

glob.glob 枚举指定路径目录，显示数据集文件结构。“*”通配符可以实现模糊查找。文件名中8个数字对应ImageNet类别WordNetID。ImageNet网站可用WordNetID查图像细节: http://www.image-net.org/synset?wnid=n02085620 。

文件名分解为品种和相应的文件名，品种对应文件夹名称。依据品种对图像分组。枚举每个品种图像，20%图像划入测试集。检查每个品种测试图像是否至少有全部图像的18%。目录和图像组织到两个与每个品种相关的字典，包含各品种所有图像。分类图像组织到字典中，简化选择分类图像及归类过程。

预处理阶段，依次遍历所有分类图像，打开列表中文件。用dataset图像填充TFRecord文件，把类别包含进去。dataset键值对应文件列表标签。record_location 存储TFRecord输出路径。枚举dataset，当前索引用于文件划分，每隔100m幅图像，训练样本信息写入新的TFRecord文件，加快写操作进程。无法被TensorFlow识别为JPEG图像，用try/catch忽略。转为灰度图减少计算量和内存占用。tf.cast把RGB值转换到[0,1)区间内。标签按字符串存储较高效，最好转换为整数索引或独热编码秩1张量。

打开每幅图像，转换为灰度图，调整尺寸，添加到TFRecord文件。tf.image.resize_images函数把所有图像调整为相同尺寸，不考虑长宽比，有扭曲。裁剪、边界填充能保持图像长宽比。

按照TFRecord文件读取图像，每次加载少量图像及标签。修改图像形状有助训练和输出可视化。匹配所有在训练集目录下TFRecord文件加载训练图像。每个TFRecord文件包含多幅图像。tf.parse_single_example只从文件提取单个样本。批运算可同时训练多幅图像或单幅图像，需要足够系统内存。

图像转灰度值为[0,1)浮点类型，匹配convolution2d期望输入。卷积输出第1维和最后一维不改变，中间两维发生变化。tf.contrib.layers.convolution2d创建模型第1层。weights_initializer设置正态随机值，第一组滤波器填充正态分布随机数。滤波器设置trainable，信息输入网络，权值调整，提高模型准确率。
max_pool把输出降采样。ksize、strides ([1,2,2,1])，卷积输出形状减半。输出形状减小，不改变滤波器数量(输出通道)或图像批数据尺寸。减少分量，与图像(滤波器)高度、宽度有关。更多输出通道，滤波器数量增加，2倍于第一层。多个卷积和池化层减少输入高度、宽度，增加深度。很多架构，卷积层和池化层超过5层。训练调试时间更长，能匹配更多更复杂模式。
图像每个点与输出神经元建立全连接。softmax，全连接层需要二阶张量。第1维区分图像，第2维输入张量秩1张量。tf.reshape 指示和使用其余所有维，-1把最后池化层调整为巨大秩1张量。
池化层展开，网络当前状态与预测全连接层整合。weights_initializer接收可调用参数，lambda表达式返回截断正态分布，指定分布标准差。dropout 削减模型中神经元重要性。tf.contrib.layers.fully_connected 输出前面所有层与训练中分类的全连接。每个像素与分类关联。网络每一步将输入图像转化为滤波减小尺寸。滤波器与标签匹配。减少训练、测试网络计算量，输出更具一般性。

训练数据真实标签和模型预测结果，输入到训练优化器(优化每层权值)计算模型损失。数次迭代，每次提升模型准确率。大部分分类函数(tf.nn.softmax)要求数值类型标签。每个标签转换代表包含所有分类列表索引整数。tf.map_fn 匹配每个标签并返回类别列表索引。map依据目录列表创建包含分类列表。tf.map_fn 可用指定函数对数据流图张量映射，生成仅包含每个标签在所有类标签列表索引秩1张量。tf.nn.softmax用索引预测。

调试CNN，观察滤波器(卷积核)每轮迭代变化。设计良好CNN，第一个卷积层工作，输入权值被随机初始化。权值通过图像激活，激活函数输出(特征图)随机。特征图可视化，输出外观与原始图相似，被施加静力(static)。静力由所有权值的随机激发。经过多轮迭代，权值被调整拟合训练反馈，滤波器趋于一致。网络收敛，滤波器与图像不同细小模式类似。tf.image_summary得到训练后的滤波器和特征图简单视图。数据流图图像概要输出(image summary output)从整体了解所使用的滤波器和输入图像特征图。TensorDebugger，迭代中以GIF动画查看滤波器变化。

文本输入存储在SparseTensor，大部分分量为0。CNN使用稠密输入，每个值都重要，输入大部分分量非0。


    import tensorflow as tf
    import glob
    from itertools import groupby
    from collections import defaultdict
    sess = tf.InteractiveSession()
    image_filenames = glob.glob("./imagenet-dogs/n02*/*.jpg")
    image_filenames[0:2]
    training_dataset = defaultdict(list)
    testing_dataset = defaultdict(list)
    image_filename_with_breed = map(lambda filename: (filename.split("/")[2], filename), image_filenames)
    for dog_breed, breed_images in groupby(image_filename_with_breed, lambda x: x[0]):
        for i, breed_image in enumerate(breed_images):
            if i % 5 == 0:
                testing_dataset[dog_breed].append(breed_image[1])
            else:
                training_dataset[dog_breed].append(breed_image[1])
        breed_training_count = len(training_dataset[dog_breed])
        breed_testing_count = len(testing_dataset[dog_breed])
        breed_training_count_float = float(breed_training_count)
        breed_testing_count_float = float(breed_testing_count)
        assert round(breed_testing_count_float / (breed_training_count_float + breed_testing_count_float), 2) > 0.18, "Not enough testing images."
    print "training_dataset testing_dataset END ------------------------------------------------------"
    def write_records_file(dataset, record_location):
        writer = None
        current_index = 0
        for breed, images_filenames in dataset.items():
            for image_filename in images_filenames:
                if current_index % 100 == 0:
                    if writer:
                        writer.close()
                    record_filename = "{record_location}-{current_index}.tfrecords".format(
                        record_location=record_location,
                        current_index=current_index)
                    writer = tf.python_io.TFRecordWriter(record_filename)
                    print record_filename + "------------------------------------------------------" 
                current_index += 1
                image_file = tf.read_file(image_filename)
                try:
                    image = tf.image.decode_jpeg(image_file)
                except:
                    print(image_filename)
                    continue
                grayscale_image = tf.image.rgb_to_grayscale(image)
                resized_image = tf.image.resize_images(grayscale_image, [250, 151])
                image_bytes = sess.run(tf.cast(resized_image, tf.uint8)).tobytes()
                image_label = breed.encode("utf-8")
                example = tf.train.Example(features=tf.train.Features(feature={
                    'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_label])),
                    'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes]))
                }))
                writer.write(example.SerializeToString())
        writer.close()
    write_records_file(testing_dataset, "./output/testing-images/testing-image")
    write_records_file(training_dataset, "./output/training-images/training-image")
    print "write_records_file testing_dataset training_dataset END------------------------------------------------------"
    filename_queue = tf.train.string_input_producer(
    tf.train.match_filenames_once("./output/training-images/*.tfrecords"))
    reader = tf.TFRecordReader()
    _, serialized = reader.read(filename_queue)
    features = tf.parse_single_example(
    serialized,
        features={
            'label': tf.FixedLenFeature([], tf.string),
            'image': tf.FixedLenFeature([], tf.string),
        })
    record_image = tf.decode_raw(features['image'], tf.uint8)
    image = tf.reshape(record_image, [250, 151, 1])
    label = tf.cast(features['label'], tf.string)
    min_after_dequeue = 10
    batch_size = 3
    capacity = min_after_dequeue + 3 * batch_size
    image_batch, label_batch = tf.train.shuffle_batch(
        [image, label], batch_size=batch_size, capacity=capacity, min_after_dequeue=min_after_dequeue)
    print "load image from TFRecord END------------------------------------------------------"
    float_image_batch = tf.image.convert_image_dtype(image_batch, tf.float32)
    conv2d_layer_one = tf.contrib.layers.convolution2d(
        float_image_batch,
        num_outputs=32,
        kernel_size=(5,5),
        activation_fn=tf.nn.relu,
        weights_initializer=tf.random_normal,
        stride=(2, 2),
        trainable=True)
    pool_layer_one = tf.nn.max_pool(conv2d_layer_one,
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding='SAME')
    conv2d_layer_one.get_shape(), pool_layer_one.get_shape()
    print "conv2d_layer_one pool_layer_one END------------------------------------------------------"
    conv2d_layer_two = tf.contrib.layers.convolution2d(
        pool_layer_one,
        num_outputs=64,
        kernel_size=(5,5),
        activation_fn=tf.nn.relu,
        weights_initializer=tf.random_normal,
        stride=(1, 1),
        trainable=True)
    pool_layer_two = tf.nn.max_pool(conv2d_layer_two,
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding='SAME')
    conv2d_layer_two.get_shape(), pool_layer_two.get_shape()
    print "conv2d_layer_two pool_layer_two END------------------------------------------------------"
    flattened_layer_two = tf.reshape(
        pool_layer_two,
        [
            batch_size,
            -1
        ])
    flattened_layer_two.get_shape()
    print "flattened_layer_two END------------------------------------------------------"
    hidden_layer_three = tf.contrib.layers.fully_connected(
        flattened_layer_two,
        512,
        weights_initializer=lambda i, dtype: tf.truncated_normal([38912, 512], stddev=0.1),
        activation_fn=tf.nn.relu
    )
    hidden_layer_three = tf.nn.dropout(hidden_layer_three, 0.1)
    final_fully_connected = tf.contrib.layers.fully_connected(
        hidden_layer_three,
        120,
        weights_initializer=lambda i, dtype: tf.truncated_normal([512, 120], stddev=0.1)
    )
    print "final_fully_connected END------------------------------------------------------"
    labels = list(map(lambda c: c.split("/")[-1], glob.glob("./imagenet-dogs/*")))
    train_labels = tf.map_fn(lambda l: tf.where(tf.equal(labels, l))[0,0:1][0], label_batch, dtype=tf.int64)
    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            final_fully_connected, train_labels))
    batch = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(
        0.01,
        batch * 3,
        120,
        0.95,
        staircase=True)
    optimizer = tf.train.AdamOptimizer(
        learning_rate, 0.9).minimize(
        loss, global_step=batch)
    train_prediction = tf.nn.softmax(final_fully_connected)
    print "train_prediction END------------------------------------------------------"
    filename_queue.close(cancel_pending_enqueues=True)
    coord.request_stop()
    coord.join(threads)
    print "END------------------------------------------------------"


参考资料：
《面向机器智能的TensorFlow实践》


