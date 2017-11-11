创建Graph对象，.Graph()，无需接收任何参数。.as_default()访问上下文管理器，添加Op。加载TensorFlow库时，自动创建一个Graph对象作为默认数据流图，Op、Tensor对象自动放置在默认数据流图。多个不依赖模型需要创建多个Graph对象，节点添加到正确的数据流图。不要将默认数据流图和用户创建数据流图混合使用，既存在自定义数据流图，又存在默认数据流图时，应把各自的Op写在with .as_default里。可以从其他TensorFlow脚本加载已定义模型，用Graph.as_graph_def()、tf.import_graph_def()赋给Graph对象。

TensorFlow Session负责执行数据流图。.Session()构造方法有3个可选参数。target指定执行引擎，默认空字符串，分布式中用于连接不同tf.train.Server实例。graph加载Graph对象，默认None，默认当前数据流图，区分多个数据流图时的执行，不在with语句块内创建Session对象。config指定Session对象配置选项。
.run()方法接收一个参数fetches指定执行对象，任意数据流图元素(Op 或Tensor对象)，Tensor对象输出NumPy数组，Op对象输出None。fetches为列表，输出元素对应值列表。三个可选参数feed_dict、options、run_metadata。feed_dict覆盖数据流图Tensor对象值，输入Python字典对象，字典键为被覆盖Tensor对象句柄，值为数字、字符串、列表、NumPy数组，类型与键相同，用于虚构值测试或指定输入值。Session对象找到所需全部节点，顺序执行节点，输出。
最后需要调用.close()方法释放资源。Session对象作为上下文管理器，离开作用域自动关闭。.as_default()作为作为上下文管理器，作为with语句块默认Session对象，必须手工关闭Session对象。
InteractiveSession运行时作为默认会话。尽量不要用。

.placeholder Op创建占位符，利用占位节点添加输入，以便数据流图变换、数值复用。dtype参数为数据类型，必须指定。shape参数为张量维数长度，默认为None，可选。name参数为标识符，可选。Session.run()中feed_dict参数，占位符输出句柄为键，传入Tensor对象为值。必须在feed_dict为计算节点每个依赖占位符包含一个健值对。placeholder的值无法计算。

    import tensorflow as tf
    import numpy as np
    in_default_graph = tf.add(1, 2)#放置在默认数据流图
    g1 = tf.Graph()#创建数据流图
    g2 = tf.Graph()
    default_graph = tf.get_default_graph()#获取默认数据流图句柄
    with g1.as_default():#放置在g数据流图
        in_graph_g1 = tf.multiply(2, 3)#此Op将添加到Graph对象g1中
    with g2.as_default():#放置在g数据流图
        in_graph_g2 = tf.div(4, 2)#此Op将添加到Graph对象g2中
    with default_graph.as_default():#放置在默认数据流图
        also_in_default_graph1 = tf.subtract(5, 2)#此Op将添加到默认数据流图中
        also_in_default_graph2 = tf.multiply(also_in_default_graph1, 2)#此Op将添加到默认数据流图中
        replace_dict = {also_in_default_graph1: 15}
        also_in_default_graph3 = tf.placeholder(tf.int32, shape=[2], name="my_input")#创建指定长度、数据类型的占位tensor
        also_in_default_graph4 = tf.reduce_prod(also_in_default_graph3, name="prod_also_in_default_graph4")#占位tensor使用
        also_in_default_graph5 = tf.reduce_sum(also_in_default_graph3, name="sum_also_in_default_graph4")#占位tensor使用
        also_in_default_graph6 = tf.add(also_in_default_graph4,also_in_default_graph5, name="add_also_in_default_graph6")#占位tensor使用
    #with tf.Session(graph=g1) as sess:#离开作用域后自动关闭
    #    sess.run(in_graph_g1)
    sess = tf.Session(graph=tf.get_default_graph())#以默认数据流图创建Session对象
    input_dict = {also_in_default_graph3: np.array([5, 3], dtype=np.int32)}#创建传给feed_dict的字典
    #sess.run([also_in_default_graph1, also_in_default_graph2], feed_dict=replace_dict)#执行Session对象，将replace_dict赋给feed_dict
    sess.run(also_in_default_graph6, feed_dict=input_dict)#将input_dict的值传给占符节点并执行
    #with sess.as_default():
    #    also_in_default_graph1.eval()
    #sess = tf.Session(graph=g1)
    #sess.run(in_graph_g1)
    #sess = tf.Session(graph=g2)
    #sess.run(in_graph_g2)
    writer = tf.summary.FileWriter('./my_graph', sess.graph)
    writer.close()
    sess.close()#关闭Session对象，释放资源


![graph?run=-3.png](http://upload-images.jianshu.io/upload_images/80690-749b3ead9a115510.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
参考资料：
《面向机器智能的TensorFlow实践》


