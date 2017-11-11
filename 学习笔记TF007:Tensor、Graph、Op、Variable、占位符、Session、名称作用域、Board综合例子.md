输入采用占位符，模型接收任意长度向量，随时间计算数据流图所有输出总和，采用名称作用域合理划分数据流图，每次运行保存数据流图输出、累加、均值到磁盘。

[None]代表任意长度向量，[]代表标量。update环节更新各Variable对象以及将数据传入TensorBoard汇总Op。与交换工作流分开，独立名称作用域包含Variable对象，存储输出累加和，记录数据流图运行次数。独立名称作用域包含TensorBoard汇总数据，tf.scalar_summary Op。汇总数据在Variable对象更新完成后才添加。

构建数据流图。
导入TensorFlow库。Graph类构造方法tf.Graph()，显式创建Graph对象。两个“全局”Variable对象，追踪模型运行次数，追踪模型所有输出累加和。与其他节点区分开，放入独立名称作用域。trainable=False设置明确指定Variable对象只能手工设置。
模型核心的变换计算，封装到名称作用域"transformation"，又划分三个子名称作用域"input"、"intermediate_layer"、"output"。.multiply、.add只能接收标量参数，.reduce_prod、. reduce_sum可以接收向量参数。
在"update"名称作用域内更新Variable对象。.assign_add实现Variable对象递增。
在"summaries"名称作用域内汇总数据供TensorBoard用。.cast()做数据类型转换。.summary.scalar()做标量数据汇总。
在"global_ops"名称作用域创建全局Operation(Op)。初始化所有Variable对象。合并所有汇总数据。

运行数据流图。
.Session()启动Session对象，graph属性加载Graph对象，.summary.FileWriter()启动FileWriter对象，保存汇总数据。
初始化Variable对象。
创建运行数据流图辅助函数，传入向量，运行数据流图，保存汇总数据。创建feed_dict参数字典，以input_tensor替换a句柄的tf.placeholder节点值。使用feed_dict运行output不关心存储，运行increment_step保存到step，运行merged_summaries Op保存到summary。添加汇总数据到FileWriter对象，global_step参数随时间图示折线图横轴。
变换向量长度多次调用运行数据流图辅助函数。.flush()把汇总数据写入磁盘。

查看数据流图。
Graph标签，变换运算流入update方框，为summaries、variables提供输入，global_ops包含变换计算非关键运算。输入层、中间层、输出层分离。
Scalars标签，summary.scalar对象标签查看不同时间点汇总数据变化。


    import tensorflow as tf#导入TensorFlow库
    #构建数据流图
    graph = tf.Graph()#显式创建Graph对象
    with graph.as_default():#设为默认Graph对象
    with tf.name_scope("variables"):#创建Variable对象名称作用域
        global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name="global_step")#记录数据流图运行次数的Variable对象，初值为0，数据类型为32位整型，不可自动修改，以global_step标识
        total_output = tf.Variable(0.0, dtype=tf.float32, trainable=False, name="total_output")#追踪模型所有输出累加和的Variable对象，初值为0.0，数据类型为32位浮点型，不可自动修改，以total_output标识
    with tf.name_scope("transformation"):#创建变换计算Op名称作用域
        with tf.name_scope("input"):#创建独立输入层名称作用域
            a = tf.placeholder(tf.float32, shape=[None], name="input_placeholder_a")#创建占位符，接收一个32位浮点型任意长度的向量作为输入，以input_placeholder_a标识
        with tf.name_scope("intermediate_layer"):#创建独立中间层名称作用域
            b = tf.reduce_prod(a, name="product_b")#创建创建归约乘积Op，接收张量输入，输出张量所有分量(元素)的乘积，以product_b标识
            c = tf.reduce_sum(a, name="sum_c")#创建创建归约求和Op，接收张量输入，输出张量所有分量(元素)的求和，以sum_c标识
        with tf.name_scope("output"):#创建独立输出层名称作用域
            output = tf.add(b, c, name="output")#创建创建求和Op，接收两个标量输入,输出标量求和,以output标识
    with tf.name_scope("update"):
        update_total = total_output.assign_add(output)#用最新的输出更新Variable对象total_output
        increment_step = global_step.assign_add(1)#增1更新Variable对象global_step，记录数据流图运行次数
    with tf.name_scope("summaries"):#创建数据汇总Op名称作用域
        avg = tf.div(update_total, tf.cast(increment_step, tf.float32), name="average")#计算平均值，输出累加和除以数据流图运行次数，把运行次数数据类型转换为32位浮点型，以average标识
        tf.summary.scalar(b'output_summary',output)#创建输出节点标量数据统计汇总，以output_summary标识
        tf.summary.scalar(b'total_summary',update_total)#创建输出累加求和标量数据统计汇总，以total_summary标识
        tf.summary.scalar(b'average_summary',avg)#创建平均值标量数据统计汇总，以average_summary标识
    with tf.name_scope("global_ops"):#创建全局Operation(Op)名称作用域
        init = tf.global_variables_initializer()#创建初始化所有Variable对象的Op
        merged_summaries = tf.summary.merge_all()#创建合并所有汇总数据的Op
    #运行数据流图
    sess = tf.Session(graph=graph)#用显式创建Graph对象启动Session会话对象
    writer = tf.summary.FileWriter('./improved_graph', graph)#启动FileWriter对象，保存汇总数据
    sess.run(init)#运行Variable对象初始化Op
    def run_graph(input_tensor):#定义数据注图运行辅助函数
        """
        辅助函数：用给定的输入张量运行数据流图，
        并保存汇总数据
        """
        feed_dict = {a: input_tensor}#创建feed_dict参数字典，以input_tensor替换a句柄的tf.placeholder节点值
        _, step, summary = sess.run([output, increment_step, merged_summaries], feed_dict=feed_dict)#使用feed_dict运行output不关心存储，运行increment_step保存到step，运行merged_summaries Op保存到summary
        writer.add_summary(summary, global_step=step)#添加汇总数据到FileWriter对象，global_step参数时间图示折线图横轴
    #用不同的输入用例运行数据流图
    run_graph([2,8])
    run_graph([3,1,3,3])
    run_graph([8])
    run_graph([1,2,3])
    run_graph([11,4])
    run_graph([4,1])
    run_graph([7,3,1])
    run_graph([6,3])
    run_graph([0,2])
    run_graph([4,5,6])
    writer.flush()#将汇总数据写入磁盘
    writer.close()#关闭FileWriter对象，释放资源
    sess.close()#关闭Session对象，释放资源


![WX20170513-142835@2x.png](http://upload-images.jianshu.io/upload_images/80690-e802d7684dae4f6e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
![graph?run=-8.png](http://upload-images.jianshu.io/upload_images/80690-86abeb9a29c9c27b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
![graph?run=-9.png](http://upload-images.jianshu.io/upload_images/80690-e9b752d7e8fdae77.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

参考资料：
《面向机器智能的TensorFlow实践》


