Tensor、Op对象不可变(immutable)。.Variable()构造方法创建Variable对象，包含Session.run()调用中可持久化的可变张量值。Variable对象初值通常为全0､全1或用随机数填充阶数较高张量，创建初值张量Op，.zeros()、.ones()、.random_normal()、.random_uniform()，接收shape参数。

Graph管理Tensor对象，Session管理Variable对象。Variable对象必须在Session对象内初始化。初始化所有Variable对象，把.global_variables_initializer() Op传给Session.run()。初始化部分Variable对象，把.variables_initializer() Op传给Session.run()。Variable.assign()Op，修改Variable对象，必须在Session对象中运行。.assign_add()创建自增Op，.assign_sub()创建自减Op。不同Session对象独立维护在Graph对象定义的Variable对象值。Optimizer类自动训练机器学习模型，自动修改Variable对象值。创建Variable对象时trainable参数设False，只允许手工修改值。

    import tensorflow as tf

    my_var = tf.Variable(3, name="my_variable")#创建Variable对象
    add = tf.add(5, my_var)
    mul = tf.multiply(8, my_var)
    zeros = tf.zeros([2, 2])#零矩阵
    ones = tf.ones([6])#全1向量
    uniform = tf.random_uniform([3,3,3], minval=0, maxval=10)#三维张量，元素服从0~10均匀分布
    normal = tf.random_normal([3,3,3], mean=0.0, stddev=2.0)#三维张量，元素服从0均值，标准差为2正态分布
    trunc = tf.truncated_normal([2,2], mean=5.0, stddev=1.0)#不会返回小于3.0或大于7.0的张量
    radom_var = tf.Variable(tf.truncated_normal([2,2]))#默认值0,默认标准差1.0
    init_global = tf.global_variables_initializer()#所有Varialbe对象初始化
    sess = tf.Session()
    sess.run(init_global)
    sess.run(add)

    var1 = tf.Variable(1, name="initialize_me")
    var2 = tf.Variable(2, name="no_initialize")
    init_part = tf.variables_initializer([var1], name="initialize_var1")
    sess.run(init_part)

    var_assign = tf.Variable(1)
    var_assign_times_two = var_assign.assign(var_assign * 2)#创建赋值Op
    init_assign = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init_assign)
    sess.run(var_assign_times_two)#2
    sess.run(var_assign_times_two)#4
    sess.run(var_assign_times_two)#8
    sess.run(var_assign.assign_add(1))#自增Op，8+1
    sess.run(var_assign.assign_sub(2))#自减Op，9-2
    var_idpt = tf.Variable(0)#0
    init_idpt = tf.global_variables_initializer()

    sess1 = tf.Session()
    sess2 = tf.Session()
    sess1.run(init_idpt)
    sess1.run(var_idpt.assign_add(5))#0+5
    sess2.run(init_idpt)
    sess2.run(var_idpt.assign_add(2))#0+2
    sess1.run(var_idpt.assign_add(10))#5+10
    sess2.run(var_idpt.assign_add(20))#2+20
    sess1.run(init_idpt)#0
    sess2.run(init_idpt)#0

    not_trainable = tf.Variable(0, trainable=False)#不可自动修改

with tf.name_scpoe(<name>)，名称作用域(name scope)组织数据流图，将Op划分到较大有名称语句块。TensorBoard加载数据流图，名称作用域封装Op。可以把名称作用域嵌在其他名称作用域内。

    import tensorflow as tf
    with tf.name_scope("Scope_A"):#建立Scope_A作用域
        a = tf.add(1, 2, name="A_add")
        b = tf.multiply(a, 3, name="A_mul")
    
    with tf.name_scope("Scope_B"):#建立Scope_B作用域
        c = tf.add(4, 5, name="B_add")
        d = tf.multiply(c, 6, name="B_mul")
    
    e = tf.add(b, d, name="output")
    writer = tf.summary.FileWriter('./name_scope_1', graph=tf.get_default_graph())
    writer.close()

    graph = tf.Graph()
    with graph.as_default():
    in_1 = tf.placeholder(tf.float32, shape=[], name="input_a")
    in_2 = tf.placeholder(tf.float32, shape=[], name="input_b")
    const = tf.constant(3, dtype=tf.float32, name="static_value")
    
    with tf.name_scope("Transformation"):
        
         with tf.name_scope("A"):
              A_mul = tf.multiply(in_1, const)
              A_out = tf.subtract(A_mul, in_1)
            
          with tf.name_scope("B"):
              B_mul = tf.multiply(in_2, const)
              B_out = tf.subtract(B_mul, in_2)
            
          with tf.name_scope("C"):
              C_div = tf.div(A_out, B_out)
              C_out = tf.add(C_div, const)
            
          with tf.name_scope("D"):
              D_div = tf.div(B_out, A_out)
              D_out = tf.add(D_div, const)
            
    out = tf.maximum(C_out, D_out)
            
    writer = tf.summary.FileWriter('./name_scope_2', graph=graph)
    writer.close()            


![graph?run=-5.png](http://upload-images.jianshu.io/upload_images/80690-10966d4746599d82.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


![graph?run=-7.png](http://upload-images.jianshu.io/upload_images/80690-b4d79ce1ae7f429d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

参考资料：
《面向机器智能的TensorFlow实践》


