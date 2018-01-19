tensorflow是通过计算图的方式建立网络。
比喻说明：
**结构**：计算图建立的只是一个网络框架。**编程时框架中不会出现任何的实际值**，所有权重（weight）和偏移是框架中的一部分，初始时要给定初始值才能形成框架，因此需要初始化。
**比喻**：计算图就是一个管道，编写网络就是搭建一个管道结构。在使用前，不会有任何的液体进入管道。我们可以将神将网络的权重和偏移当成管道中的阀门，可以控制液体的流动强弱和方向，在神经网络的训练中，阀门会根据数据进行自我调节、更新，但是使用之前至少给所有的阀门一个初始的状态才能形成结构，并且**计算图允许我们可以从任意一个节点处取出液体。**

下面是Graph的详细介绍：
tf计算都是通过数据流图（Graph）来展现的，一个数据流图包含一系列**节点（op）操作以及在节点之间流动的数据**，这些节点和数据流分别称之为计算单元和Tensor对象。**当进入tf时（例如import tensorflow as tf），tf内部会注册一个默认的Graph**，可通过 tf.get_default_graph()  来获得这个默认的Default_Graph，只要简单地调用某个函数，就可以向这个默认的图里面添加操作（节点）。
（1）tf.Graph()

**[python]** [view plain](http://blog.csdn.net/u014365862/article/details/77944301#) [copy](http://blog.csdn.net/u014365862/article/details/77944301#)

g = tf.Graph()  
with g.as_default():  
    # Define operations and tensors in `g`.  
    c = tf.constant(30.0)  
    assert c.graph is g  

重要提示:Graph类在构建图时非线程安全。所有的节点（操作）必须在单线程内创建或者必须提供或外部同步。除非另有规定,不然所有的方法都不是线程安全的。（2）tf.Graph.as_default()
        该方法返回一个上下文管理器，并将Graph当做默认图。若想在同一进程中创建多个图，可调用此方法。**为了方便,tf-开始就提供了一个全局缺省图,所有节点将被添加到这个缺省图（开始时候提到了）**，如果没有显式地创建一个新的图的话。
        默认图是当前线程的属性。如果您想创建一个新的线程，并希望在新线程使用默认的图，就必须明确使用g.as_default（）搭配with关键字来创建一个新作用域，并在该新作用域内执行一系列节点。

**[python]** [view plain](http://blog.csdn.net/u014365862/article/details/77944301#) [copy](http://blog.csdn.net/u014365862/article/details/77944301#)

# 1. Using Graph.as_default():  
g = tf.Graph()  
with g.as_default():  
    c = tf.constant(5.0)  
    assert c.graph is g  
  
# 2. Constructing and making default:  
with tf.Graph().as_default() as g:  
    c = tf.constant(5.0)  
    assert c.graph is g  

（3）tf.Graph.as_graph_def(from_version=None, add_shapes=False)
        该方法返回一个序列化的GraphDef。可在另一个图中调用该序列化的GraphDef（通过 [import_graph_def()
](https://www.tensorflow.org/versions/r0.11/api_docs/python/framework.html#import_graph_def))或者[C++ Session API](https://www.tensorflow.org/versions/r0.11/api_docs/cc/index.html).）
该方法是线程安全的。
（4）tf.Graph.finalize()
   使用该方法后，后续节点（操作）不能再添加到改图（图结构被锁定了）。该方法可以确保此图在不同线程之间计算时，不会再被添加额外的节点。使用场景有QueueRunner（多线程读取队列文件）

（5）tf.Graph.finalized
        若图锁定，就返回True
。。。。。直接看图吧，更直接。

tf.Graph
操作
描述

class tf.Graph
tensorflow中的计算以图数据流的方式表示一个图包含一系列表示计算单元的操作对象以及在图中流动的数据单元以tensor对象表现

tf.Graph.__init__()
建立一个空图

tf.Graph.as_default()
一个将某图设置为默认图，并返回一个上下文管理器如果不显式添加一个默认图，系统会自动设置一个全局的默认图。所设置的默认图，在模块范围内所定义的节点都将默认加入默认图中

tf.Graph.as_graph_def(from_version=None, add_shapes=False)
返回一个图的序列化的GraphDef表示序列化的[GraphDef](https://github.com/tensorflow/tensorflow/blob/r0.10/tensorflow/core/framework/graph.proto)可以导入至另一个图中(使用 import_graph_def())或者使用C++ Session API

tf.Graph.finalize()
完成图的构建，即将其设置为只读模式

tf.Graph.finalized
返回True，如果图被完成

tf.Graph.control_dependencies(control_inputs)
定义一个控制依赖，并返回一个上下文管理器with g.control_dependencies([a, b, c]):# `d` 和 `e` 将在 `a`, `b`, 和`c`执行完之后运行.d = …e = …

tf.Graph.device(device_name_or_function)
定义运行图所使用的设备，并返回一个上下文管理器with g.device('/gpu:0'): ...
with g.device('/cpu:0'): ...

tf.Graph.name_scope(name)
为节点创建层次化的名称，并返回一个上下文管理器

tf.Graph.add_to_collection(name, value)
将value以name的名称存储在收集器(collection)中

tf.Graph.get_collection(name, scope=None)
根据name返回一个收集器中所收集的值的列表

tf.Graph.as_graph_element(obj, allow_tensor=True, allow_operation=True)
返回一个图中与obj相关联的对象，为一个操作节点或者tensor数据

tf.Graph.get_operation_by_name(name)
根据名称返回操作节点

tf.Graph.get_tensor_by_name(name)
根据名称返回tensor数据

tf.Graph.get_operations()
返回图中的操作节点列表

tf.Graph.gradient_override_map(op_type_map)
用于覆盖梯度函数的上下文管理器

****
tf.Operation（节点op：开始时候提到过，节点就是计算单元）
操作
描述

class tf.Operation
代表图中的一个节点，用于计算tensors数据该类型将由python节点构造器产生(比如tf.matmul())或者Graph.create_op()例如c = tf.matmul(a, b)创建一个Operation类为类型为”MatMul”,输入为’a’,’b’，输出为’c’的操作类

tf.Operation.name
操作节点(op)的名称

tf.Operation.type
操作节点(op)的类型，比如”MatMul”

tf.Operation.inputstf.Operation.outputs
操作节点的输入与输出

tf.Operation.control_inputs
操作节点的依赖

tf.Operation.run(feed_dict=None, session=None)
在会话(Session)中运行该操作

tf.Operation.get_attr(name)
获取op的属性值

tf.Tensor（节点间流动的数据，上面也有所提到）
操作
描述

class tf.Tensor
表示一个由操作节点op产生的值，TensorFlow程序使用tensor数据结构来代表所有的数据, 计算图中, 操作间传递的数据都是 tensor，一个tensor是一个符号handle,里面并没有表示实际数据，而相当于数据流的载体

tf.Tensor.dtype
tensor中数据类型

tf.Tensor.name
该tensor名称

tf.Tensor.value_index
该tensor输出外op的index

tf.Tensor.graph
该tensor所处在的图

tf.Tensor.op
产生该tensor的op

tf.Tensor.consumers()
返回使用该tensor的op列表

tf.Tensor.eval(feed_dict=None, session=None)
在会话中求tensor的值需要使用with sess.as_default()
或者 eval(session=sess)

tf.Tensor.get_shape()
返回用于表示tensor的shape的类TensorShape

tf.Tensor.set_shape(shape)
更新tensor的shape

tf.Tensor.device
设置计算该tensor的设备

****
tf.DType
操作
描述

class tf.DType
数据类型主要包含tf.float16，tf.float16,tf.float32,tf.float64,tf.bfloat16,tf.complex64,tf.complex128,tf.int8,tf.uint8,tf.uint16,tf.int16,tf.int32,tf.int64,tf.bool,tf.string

tf.DType.is_compatible_with(other)
判断other的数据类型是否将转变为该DType

tf.DType.name
数据类型名称

tf.DType.base_dtype
返回该DType的基础DType，而非参考的数据类型(non-reference)

tf.DType.as_ref
返回一个基于DType的参考数据类型

tf.DType.is_floating
判断是否为浮点类型

tf.DType.is_complex
判断是否为复数

tf.DType.is_integer
判断是否为整数

tf.DType.is_unsigned
判断是否为无符号型数据

tf.DType.as_numpy_dtype
返回一个基于DType的numpy.dtype类型

tf.DType.maxtf.DType.min
返回这种数据类型能表示的最大值及其最小值

tf.as_dtype(type_value)
返回由type_value转变得的相应tf数据类型

* 通用函数（Utility functions）
操作
描述

tf.device(device_name_or_function)
基于默认的图，其功能便为Graph.device()

tf.container(container_name)
基于默认的图，其功能便为Graph.container()

tf.name_scope(name)
基于默认的图，其功能便为 Graph.name_scope()

tf.control_dependencies(control_inputs)
基于默认的图，其功能便为Graph.control_dependencies()

tf.convert_to_tensor(value, dtype=None, name=None, as_ref=False)
将value转变为tensor数据类型

tf.get_default_graph()
返回返回当前线程的默认图

tf.reset_default_graph()
清除默认图的堆栈，并设置全局图为默认图

tf.import_graph_def(graph_def, input_map=None,return_elements=None, name=None, op_dict=None,producer_op_list=None)
将graph_def的图导入到python中

* 图收集（Graph collections）
操作
描述

tf.add_to_collection(name, value)
基于默认的图，其功能便为Graph.add_to_collection()

tf.get_collection(key, scope=None)
基于默认的图，其功能便为Graph.get_collection()

* 定义新操作节点（Defining new operations）
tf.RegisterGradient
操作
描述

class tf.RegisterGradient
返回一个用于寄存op类型的梯度函数的装饰器

tf.NoGradient(op_type)
设置操作节点类型op_type的节点没有指定的梯度

class tf.RegisterShape
返回一个用于寄存op类型的shape函数的装饰器

class tf.TensorShape
表示tensor的shape

tf.TensorShape.merge_with(other)
与other合并shape信息，返回一个TensorShape类

tf.TensorShape.concatenate(other)
与other的维度相连结

tf.TensorShape.ndims
返回tensor的rank

tf.TensorShape.dims
返回tensor的维度

tf.TensorShape.as_list()
以list的形式返回tensor的shape

tf.TensorShape.is_compatible_with(other)
判断shape是否为兼容TensorShape(None)与其他任何shape值兼容

class tf.Dimension
 

tf.Dimension.is_compatible_with(other)
判断dims是否为兼容

tf.Dimension.merge_with(other)
与other合并dims信息

tf.op_scope(values, name, default_name=None)
在python定义op时，返回一个上下文管理器
