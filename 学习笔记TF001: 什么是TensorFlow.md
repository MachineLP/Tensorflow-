官网: https://www.tensorflow.org

Github: https://github.com/tensorflow

学习TensorFlow，如果你学过以下知识会更有帮助：一元、多元微积分，矩阵代数、乘法，编程原理，机器学习,Python 编程，模块组织,NumPy库,matplotlib库，前馈神经网络、卷积神经网络、循环神经网络。

机器学习，通用数学模型解决数据特定问题。垃圾邮件检测、产品推荐、预测商品价值都有应用。深度学习，多层神经网络。足够数量数据，通用GPU发展，更有效训练算法，使深度学习快速发展。深度学习接收所有参数，自动确定有用高阶组合。

在以下情况下可以考虑使用TensorFlow：研究、开发、迭代新机器学习架构、新模型。模型训练直接切换部署。快速迭代，快速实现产品。实现已有复杂架构。可视化计算图，使用TensorFlow构建，实现最新研究文献模型。大规模分布式模型。多台多种实虚设备用例。为移动嵌入式系统创建训练模型。预训练模型。

使用TensorFlow有以下优势：易用性，工作流易理解，API一致。API稳定，向下兼容。与NumPy无缝集成，数据科学家。不占编译时间，快速验证想法。多种高层接口，Keras、SkFlow。灵活性，不同类型、尺寸机器。大模型数据集模型训练完成时间合理。分别或同时使用CPU 、GPU。高效性，从低效到性能榜首，众多专家持续改进。谷歌推动支持。使用研究者与开发者通用语言。谷歌日常工作用到，并进行投资。社区知名成员众多。谷歌发布预训练模型，免费使用，迅速实现原型系统。TensorBoard调试、可视化模型，非常方便。TensorFlow Serving方便初创公司部署。

TensorFlow还存在的问题：分布式支持不成熟，手工定义每台设备角色。Kubernetes未完成。定制代码技巧强，创建用户运算，定制代码实现不易。特性缺失，因为比较新，有很多应用场景所需要的库还没有提供。

Google在2011年启动，开发了第一代分布式机器学习框架 DistBelief。第二代TensorFlow，6倍训练速度提升，在2015年11月开源，2016年4月发分布式版本，2017年1月发1.0版本。现已有2000多个项目建立深度学习模型。
TensorFlow的主要维护团队是Google Brain。开源，一个是借助社区力量，一个是回馈社区，推动机器学习应用。Google还有很多在IT界广泛使用的开源库：Android、Chromium、Go、JavaScript V8、Protobuf、Bazel、Tesseract等。

Tensorflow既是机器学习算法实现接口，又是机器学习算法执行框架。TensorFlow的前端支持Python、C++、Go、Java等，目前推荐优先使用Python。后端使用C++、CUDA等。TensorFlow算法在异构系统方便移植，Android、iOS、CPU服务器、GPU集群等。可以实现机器学习、线性回归、逻辑回归、随机森林等算法。在语音识别、自然语言处理、计算机视觉、机器人控制、信息抽取、药物研发、分子活动预测等方面应用广泛。

TensorFlow架构包括：C++、Python的前端接口，Core TensorFlow Execution System中间层，以及CPU、GPU、Android、iOS的底层实现。

TensorFlow用数据流图规划计算流程，映射到不同硬件、操作系统。同时满足大规模模型训练和小规模应用部署。计算用有状态的数据流图表示。简单实现并行计算，用不同硬件资源训练。同步或异步更新全局共享模型参数状态。
做了共同子图消除、异步核优化、通信优化、模型并行、数据并行、流水线等优化。

计算表示为有向图，叫计算图。每个运算操作一个节点。节点的连接叫边。计算图描述计算流程，维护更新状态，分支控制（条件、循环）。流动数据是多维数组，叫张量（tensor）。张量类型可先定义，可计算图结构推断。没有数据流动的边是依赖控制，起始节点执行后再执行目标节点。

运算操作代表一类抽象运算。运算操作预先设置或由计算图推断。设置运算操作属性支持不同张量元素类型。运算核是运算操作在具体硬件实现。新运算操作、核要注册。

TensorFlow有很多内建运算操作：标量运算（Add、Sub、Mul、Div、Exp、Log、Greater、Less、Equal）、向量运算（Concat、Slice、Split、Constant、Rank、Shape、Shuffle）、矩阵运算（MatMul、MatrixInverse、MatrixDeterminant）、带状态的运算（Variable、Assign、AssignAdd）、神经网络组件（SoftMax、Sigmoid、ReLU、Convolution2D、MaxPooling）、储存（Save）、恢复（Restore）、队列运算（Enqueue、Dequeue）、同步运算（MutexAcquire、MutexRelease）、控制流（Merge、Switch、Enter、Leave、NextIteration）。

Session是交互接口。Session.Extend添加新节点、边，创建计算图。Session.Run执行计算图。自动找所有需要计算节点，按依赖顺序执行。创建一次，反复执行整个或部分计算图。数据tensor不持续保留。Variable把tensor储存在内存或显存。tensor被保存被更新。

client通过Session接口连接master、worker。每个worker连接、管理多个硬件设置device。master指挥worker执行计算图。单机模式，client、master、worker在同一机器同一进程，设备name：/job:localhost/device:cpu:0。分布模式，client、master、worker在不同机器不同进程，集群调度系统管理，设备name：/job:worker/task:17/device:gpu:3。

管理设备对象接口，对象分配释放设备内存，执行节点运算核。tensor数据类型支持int、float、double、复数、字符串。tensor通过引用计数管理内存。支持x86 CPU、ARM CPU、GPU、TPU。

单机单设备，计算图按依赖关系顺序执行。节点所有上游依赖执行完，依赖为0,节点加入ready queue等待执行。下游所有节点依赖数减1。标准计算拓扑序。
单机多设备，要解决两个问题，指定执行节点的硬件设备和管理节点数据通信。
节点分配设备策略，计算代价模型，估算节点输入输出tensor大小及计算时间。人工经验制定启发式规则，加小部数据实际运算测量。模拟整个计算图执行，从起点，拓扑序执行，把能执行设备测试，代价模型计算时间估算加数据通信时间，内存使用峰值，选择综合时间最短设备。节点分配可设置限制条件。先找节点可用设备，再找同一设备的节点。
划分计算子图，同设备相邻节点同一子图。不同子图通过发送节点、接收节点及它们的边通信。同一tensor接收节点合成一个。with tf.device("/gpu:0"):实现单机多设备。
分布式（多机），发送接收节点实现不同。不同机器TCP、RDMA传输数据。发送、接收节点传输失败，worker心跳检测失败，计算图终止重启。Variable node 连接 Save、Restore node。检查点（checkpoint）保存数据，重启时从上一检查点恢复。

计算cost function梯度，TensorFlow原生支持自动求导，用的是反向传播算法（back propagation）。通过计算图拓展节点方式实现，求梯度节点对用户透明。计算两个tensor之间的梯度，先寻找正向路径，然后回溯，每个节点增加对应求解梯度节点，根据链式法则计算总梯度。新增节点计算梯度函数，tf.gradients()。自求求导带来计算优化的麻烦。正向执行计算图进行推断，确定执行顺序，使用经理规则容易取得好效果，tensor迅速被后续节点用掉，不持续占用内存。反向传播计算梯度，需要用计算图开头tensor，占用大量GPU显存，限制模型规模。现在通过更好的优化方法，重新计算tensor不保存，tensor从GPU显存移到CPU主内存等方式改善。

支持单独执行子图，用节点名加potr形式指定数据，也可以选择一组输入数据映射，并指定一组输出数据。整个计算图根据输入、输出调整，输入节点连接feed node。输出节点连接fetch node。根据输出数据自动求导执行节点。

支持计算图控制流。Swith、Merge根据布尔值跳过子图，把子图结果合并。Enter、Leave、NestIteration实现循环迭代。Loop每次循环有唯一tag，执行结果输出frame,可以查询结果日志。控制流支持分布式，循环节点划分到不同子图，子图连接控制节点，实现循环，终止信号发送其他子图。支持隐含梯度计算节点。通过feed node输入数据，从client读取，网络传到分布式系统其他节点，网络开销大。通过input node 输入文件系统路径，worker节点读取本地文件。

队列（queue）调度任务，节点导步执行。数据运算时，提前从磁盘读取下一batch数据，减少磁盘I/O阻塞。导步计算梯度，组合整个梯度。有先进先出（FIFO）队列、先牌(shuffling)队列。

容器（ Container）管理长期变量。每一进程一默认容器，一直存在到进程结束。共享不同计算图不同Session状态值。

识别高阶运算操作的重复计算，改写计算图，只执行一次。安排运算顺序改善数据传输、内存占用问题。提供异度计算，避免计算和I/O阻塞。支持线性代数计算库 Eigen、矩阵乘法计算库 BLAS、cuBLAS、深度学习计算库 cuda-convnet、cuDNN。支持数据压缩。

数据并行，mini-batch数据，不同设备计算，实现梯度计算并行。使用多个线程控制梯度计算，每一线程计算导步更新模型参数。同步没有梯度干扰，空错性差。异步容错性好，梯度干扰，梯度利用率下降。混合式，先大块异步，再块内同步。计算性能损耗小。不同mini-batch干扰少，可同时多份（replicas）数据并行。模型并行，计算图不同部分在不同设备运算。减少每轮训练迭代时间。需要模型有大量并行互不依赖子图。不同硬件性能损耗不同。流水线并行，计算做成流水线，在同一设备连续并行执行。未来，把任意子图封装函数，提供just-in-time编译器推断tensor类型大小，自动生成高度优化流水线，优化节点分配设备，优化节点执行排序。

开源深度学习框架，Google、微软、Facebook都参与。基本都支持Python。NumPy、SciPy、Pandas、Scikit-lean、XGboost等组件方便数据采集、预处理。

TensorFlow方便设计神经网络结构。不必写C++、CUDA。自动求导。C++核心方便线上部署。支持低配置嵌入式设备。通过SWIG添加其他脚本支持。每个mini-batch要从Python feed到网络，延迟大。内置TF.lean、TF.Slim快速设计网络。兼容scikit-lean estimator接口，实现evaluate、grid、search、cross、validation等功能。自动将分支转为子图。灵活移值，轻易部署任意数量设备。编译快。TensorBoard可视化网络结构、过程。支持常见网络结构（卷积神经网络CNN，循环神经网络RNN）、深度强化学习、计算密集科学计算等。缺点，计算图必须为静态图。异构支持各种硬件和操作系统。框架分布式性能是关键。TensorFlow 单机reduction只能用CPU，分布式通信使用socket RPC,不是RDMA，待优化。TensorFlow Serving 组件可导出训练好的模型，部署成对外提供预测是服务RESTful接口。实现应用机器学习全流程，训练模型、调试参数、打包模型、部署服务。TensorBoard，监控运行过程关键指标，支持标量、图片、直方图、计算图可视化。

Caffe广泛使用。网络结构以配置文件定义。训练速度快。组件模块化。每个神经网络模块一个Layer。Layer接收输入数据，内部计算产生输出数据。把各Layer拼接构成完整网络。每个Layer定义正向预测运算、反向传播运算。部分需要自己写C++或CUDA代码。只针对图像，没有考虑文本、语音、时间序列数据。有大量训练好的经典模型。工业界、学术界提供源码的深度学习论文都使用Caffe。人脸识别、图片分类、位置检测、目标追踪有很多应用。适合稳定性要求严格生产环境。第一个主流工业级深度学习框架。移植性好。Protobuf配置定义神经网络结构，command line 训练预测。GPU训练性能很好。CaffeOnSpark实现大规模分布式训练。

Theano开发较早，2008年。集成NumPy，直接使用ndarray。计算稳定性好。动态生成C或CUDA代码。部署困难，不适合生产。完全基于Python，自动求导。对卷积神经网络支持很，符号计算支持循环控制，RNN实现简单高性能。派生大量深度学习库，Keras可随意切换计算后端，适合快速探索网络结构，组件可插拔，Lasagne网络每层定义严谨，还有scikit-neuralnetwork、nolearn、blocks、deepy、pylearn2、Scikit-theano等库。在上层封装库里，不需要从最基础tensor粒度，可以从更上层layer粒度开始设计网络。

Torch，高效科学计算库，支持大量机器学习算法，GPU计算优先。Facebook开源深度学习组件。大量机器学习、计算机视觉、信号处理、并行运算、图像、视频、音频、网络处理库。大量训练好的深度学习模型。支持设计非常复杂的神经网络拓扑图结构。Lua性能高，达到C的80%。方便使用大量C库。方便实现复杂操作。

Lasagne基于Theano。轻度封装。简化操作又支持底层操作。

Keras高度模块化。最快速度原型实验。专精深度学习。降低编程开销、理解开销。模型越复杂，收益越大。

MXNet混合符事情编程模型和指令式编程模型。分布式性能高。动态依赖调度器，上层计算图优化算法使符号计算非常快，可以在小内存GPU训练模型，在移动设备运行任务。支持非常多的语言封装。

DIGITS封装Caffe。不需要也不可能写代码。浏览器进行操作，可视化界面，可视化统计报表，可视化结构图。上传图片到服务或输入url分类图片。

 CNTK由微软开源。语音识别领域使用广泛。丰富细粒度神经网络组件。性能导向。1-bit compression，降低通信代价，提升大规模分布并行效率。可以自定义计算节点。部署简单，但不支持ARM。唯一支持单机8GPU。

DeepLearning4J基于Java和Scala。即插即用解决方案原型。可以和Hadoop、Spark整合，根据集群节点、连接自动优化。大量Java分布式集群、JVM库。有商业版，付费用户电话技术支持。

Chainer日本。网络在实时运行中定义。存储历史运行计算结果。方便使用Python控制流。

Leaf基于Rust语言。移植性好。性能高。最简单的API。

DSSTNE由亚马逊开源。稀疏神经网络框架。支持自动模型并行。支持超大稀疏数据训练。


参考资料：
《TensorFlow实战》
《面向机器智能的TensorFlow实践》


