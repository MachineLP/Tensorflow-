TensorFlow目录结构。

    ACKNOWLEDGMENTS #TensorFlow版本声明
    ADOPTERS.md #使用TensorFlow的人员或组织列表
    AUTHORS #TensorFlow作者的官方列表
    BUILD
    CONTRIBUTING.md #TensorFlow贡献指导
    ISSUE_TEMPLATE.md #提ISSUE的模板
    LICENSE #版权许可
    README.md
    RELEASE.md #每次发版的change log
    WORKSPACE #配置移动端开发环境
    bower.BUILD
    configure
    models.BUILD
    tensorflow #主目录
    third_party #第三方库，包括eigen3(特征运算，SVD、LU分解等)、gpus(支持cuda)、hadoop、jpeg、llvm、py、sycl
    tools #构建cuda支持
    util

tensorflow目录结构：

    BUILD
    __init__.py
    c
    cc #采用C++进行训练的亲样例
    compiler
    contrib #将常用功能封装在一起高级API
    core #C++实现主要目录
    examples #各种示例
    g3doc #针对C++、Python版本代码文档
    go
    java
    opensource_only #声明目录
    python #Python实现主要目录
    stream_executor #流处理
    tensorboard #App、Web支持，以及脚本支持
    tensorflow.bzl
    tf_exported_symbols.lds
    tf_version_script.lds
    tools #工具杂项
    user_ops
    workspace.bzl

contirb目录。保存常用功能封装高级API。不是官方支持。高级API完善后被官方迁移到核心TensorFlow目录或去掉。部分包(package)在https://github.com/tensorflow/models 有更完整实现。
framework:很多函数在这里定义(get_varibles、get_global_step)，一些废弃或不推荐(deprecated)函数。
layers:initializers.py，变量初始化函数。layers.py，层操作和权重偏置变量函数。optimizers.py，损失函数和global_step张量优化器操作。regularizers.py，带权重正则化函数。summaries.py，摘要操作添加到tf.GraphKeys.SUMMARIES集合中的函数。
learn:使用TensorFlow进行深度学习高级API，训练模型、评估模型、读取批处理数据、队列功能API封装。
rnn:额外RNN Cell，对RNN隐藏层改进，LSTMBlockCell、GRUBlockCell、FusedRNNCell、GridLSTMCell、AttentionCellWrapper。
seq2seq:建立神经网络seq2seq层和损失函数操作。
slim:TensorFlow-Slim(TF-Slim)，定义、训练、评估TensorFlow复杂模型轻量级库。TF-Slim与TensorFlow原生函数和tf.contrib其他包自由组合。TF-Slim已逐渐迁移到TensorFlow开源Models，里面有广泛使用卷积神经网络图像分类模型代友，可以从头训练模型或预测训练模型开始微调。

core目录。C语言文件，TensorFlow原始实现。

    BUILD
    common_runtime #公共运行库
    debug
    ditributed_runtime #分布式执行模块，含有grpc session、grpc worker、grpc master
    example
    framework #基础功能模块
    graph
    kernels #核心操作在CPU、CUDA内核实现
    lib #公共基础库
    ops
    platform #操作系统实现相关文件
    protobuf #.proto文件，用于传输时结构序列化
    public #API头文件目录
    user_ops
    util
Protocol Buffers，谷歌公司创建的数据序列化(serialization)工具，结构化数据序列化，数据存储或RPC数据交换格式。定义协议缓冲区，生成.pb.h和.pb.cc文件。定义get、set、序列化、反序列化函数。TensorFlow核心proto文件graph_def.proto、node_def.proto、op_def.proto保存在framework目录。构图时先构建graph_def，存储下来，在实际计算时再转成图、节点、操作内存对象。
tensorflow-1.1.0/tensorflow/core/framework/node_def.proto，定义proto文件。node_def.proto定义指定设备(device)操作(op)、操作属性(attr)。
framework 目录还有node_def_builder.h、node_def_builder.cc、node_def_util.h、node_def_util_test.cc。在C++里操作node_def.proto的protobuf结构。

examples目录，深度学习例子，MNIST、Word2vec、Deepdream、Iris、HDF5。TensorFlow在Android系统上的移动端实现。扩展.ipynb文档教程，jupyter打开。

g3doc。存放Markdown维护的TensorFlow文档，离线手册。g3doc/api_docs目录内容从代码注释生成，不应该直接编辑。脚本tools/docs/gen_docs.sh生成API文档。无参数调用，只重新生成Python API文档，操作文档，包括Python、C++定义。传递-a，运行脚本重新生成C++ API文档，需要完装doxygen。必须从tools/docs目录调用。

python目录。激活函数、卷积函数、池化函数、损失函数、优化方法。

tensorboad目录。实现TensorFlow图表可视化工具代码，代码基于Tornado实现网页端可视化。http://www.tornadoweb.org/en/stable/ 。

TensorFlow源代码学习方法。
1)了解自己研究的基本领域，图像分类、物体检测、语音识别，了解领域所用技术，卷积神经网络(convolutional neural network,CNN)、循环神经网络(recurrent neural network,RNN)，知道实现基本原理。
2)运行GitHub对应基本模型，目录结构：

    AUTHORS
    CONTRIBTING.md
    LICENSE
    README.md
    WORKSPACE
    autoencoder
    compression
    differential_privacy
    im2txt
    inception
    lm_1b
    namignizer
    neural_gpu
    neural_programmer
    next_frame_prdiction
    resnet
    slim
    street
    swivel
    syntaxnet
    textsum
    transformer
    tutorials
    video_prediction
计算机视觉，compression(图像压缩)、im2txt(图像描述)、inception(对ImageNet数据集用Inception V3架构训练评估)、resnet(残差网络)、slim(图像分类)、street(路标识别或验证码识别)。
自然语言处理，lm_1b(语言模型)、namignizer(起名字)、swivel(Swivel算法转换词向量)、syntaxnet(分词和语法分析)、textsum(文本摘要)、tutorials目录word2vec(词转换向量)。
教科书式代码，看懂学懂有助今后自己实现模型。运行模型，调试、调参。完整读完MNIST或CIFAR10整个项目逻辑，就掌握TensorFlow项目架构。
slim目录。TF-Slim图像分类库。定义、训练、评估复杂模型轻量级高级API。训练、评估lenet、alexnet、vgg、inception_v1、inception_v2、inception_v3、inception_v4、resnet_v1、resnet_v2，模型位于slim/nets:

    alexnet.py
    alexnet_test.py
    cifarnet.py
    inception.py
    inception_resnet_v2.py
    inception_resnet_v2_test.py
    inception_utils.py
    inception_v1.py
    inception_v1_test.py
    inception_v2.py
    inception_v2_test.py
    inception_v3.py
    inception_v3_test.py
    inception_v4.py
    inception_v4_test.py
    lenet.py
    nets_factory.py
    nets_factory_test.py
    overfeat.py
    overfeat_test.py
    resnet_utils.py
    resnet_v1.py
    resnet_v1_test.py
    resnet_v2.py
    resnet_v3_test.py
    vgg.py
    vgg_test.py
TF-Slim包含脚本从头训练模型或从预先训练网络开始训练模型并微调，slim/scripts:

    finetune_inception_v1_on_flowers.sh
    finetune_inception_v3_on_flowers.sh
    train_cifarnet_on_cifar10.sh
    train_lenet_on_mnist.sh
TF-Slim包含下载标准图像数集，转换TensorFlow支持TFRecords格式脚本，slim/datasets:

    cifar10.py
    dataset_factory.py
    dataset_utils.py
    download_and_convert_cifar10.py
    download_and_convert_flowers.py
    download_and_convert_mnist.py
    flowers.py
    imagenet.py
    mnist.py
3)结合要做的项目，找到相关论文，自己用TensorFlow实现论文内容。质的飞跃。

参考资料：
《TensorFlow技术解析与实战》


