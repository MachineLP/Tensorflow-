PlayGround。http://playground.tensorflow.org 。教学目的简单神经网络在线演示、实验图形化平台。可视化神经网络训练过程。在浏览器训练神经网络。界面，数据(DATA)、特征(FEATURES)、神经网络隐藏层(HIDDEN LAYERS)、层中连接线、输出(OUTPUT)。

数据。二维平面，蓝色正值，黄色负值。数据形态，圆形、异或、高斯、螺旋。数据配置，调整噪声(noise)大小，改变训练、测试数据比例(ratio)，调整入输入每批(batch)数据数量1-30。

特征。特征提取(feature extraction)。每个点有X1､X2两个特征。衍生其他特征，X1X1､X2X2､X1X2､sin(X1)、sin(X2)。X1左边黄色是负，右边蓝色是正，表示横坐标值。X2上边蓝色是正，下边黄色是负，表示纵坐标值。X1X1横坐标抛物线信息。X2X2纵坐标抛物线信息。X1X2双曲抛物面信息。sin(X1)横坐标正弦函数信息。sin(X2)纵坐标正弦函数信息。分类器(classifier)结合特征，画出线，把原始颜色数据分开。

隐藏层。设置隐藏层数量、每个隐藏层神经元数量。隐藏层间连接线表示权重(weight)，蓝色表示神经元原始输出，浅色表示神经元负输出。连接线粗细、深浅表示权重绝对值大小。鼠标放线上可以看到具体值，修改值。修改值要考虑激活函数。Sigmoid，没有负向黄色区域，值域(0,1)。下层神经网络神经元对上层输出再组合。根据上次预测准确性，反向传播每个组合不同权重。组合连接线粗细深浅变化。越深越粗，权重越大。

输出。黄色点归于黄色背景，蓝色点归于蓝色背景。背景颜色深浅代表可能性强弱。选定螺旋形数据，7个特征全部输入，3个隐藏层，第一层8个神经元，第二层4个神经元，第三层2个神经元。训练2分钟，测试损失(test loss)和训练损失(training loss)不再下降。只输入最基本前4特征，6个隐藏层，前4层8个神经元，第五层6个神经元，第六层2个神经元。增加神经元个数和神经网络层数。

TensorBoard。TensorFlow自带可视化工具，Web应用程序套件。7种可视化，SCALARS(训练过程准确率、损失值、权重/偏置变化)、IMAGES(训练过程记录图像)、AUDIO(训练过程记录音频)、GRAPHS(模型数据流图、各设备消耗内存时间)、DISTRIBUTIONS(训练过程记数据分布图)、HISTOGRAMS(训练过程记录数据柱状图)、EMBEDDINGS(展示词向量投影分布)。

运行本地服务器，监听6006端口。浏览器发出请求，分析训练记录数据，绘制训练过程图像。运行手写数字识别入门例子。python tensorflow-1.1.0/tensorflow/examples/tutorials/mnist/mnist/mnist_with_summaries.py 。打开TensorBoard面板。tensorboard -logdir=/tmp/mnist/logs/mnist_with_summaries 。浏览器找开网址，查看面板各项功能。

SCALARS面板。左边，Split on undercores(用下划线分开显示)、Data downloadlinks(数据下载链接)、Smoothing(图像曲线平滑程度)、Horizontal Axis(水平轴)。水平轴3种，STEP迭代次数，RELATIVE训练集测试集相对值，WALL时间。右边，准确率、交叉熵损失函数值变化曲线(迭代次数1000次)。每层偏置(biases)、权重(weights)变化曲线，每次迭代最大值、最小值、平均值、标准差。

IMAGES面板。训练数据集、测试集预处理后图片。

AUDIO面板。训练过程处理音频数据。

GRAPHS面板。数据流图。节点间连线为数据流，连线越粗，两个节点间流动张量(tensor)越多。左边，选择迭代步骤。不同Color(颜色)不同Structure(整个数据流图结构)，不同Color不同Device(设备)。选择特定迭代step899,显示各个节点Compute time(计算时间)、Memory(内存消耗)。

DISTRIBUTIONS面板。平面表示特定层激活前后、权重、偏置分布。

HISTOGRAMS面板。立体表示特定层激活前后、权重、偏置分布。

EMBEDDINGS面板。词嵌入投影。

词嵌入(word embedding)，自然语言处理，推荐系统。Word2vec。TensorFlow Word2vec basic版、optimised版。

降维分析。代码 https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/word2vec/word2vec_basic.py 。Word2vec训练数据集text8 http://mattmahoney.net/dc/textdata 。只包含a~z字符和空格，27种字符。Skip-gram模型，根据目标词汇预测上下文。给定n个词围绕词w，w预测一个句子中一个缺漏词c，概率p(c|w)表示。生成t-SNE降给呈现词汇接近程度关系。

word2vec_basic.py。
下载文件读取数据，read_data函数，读取输入数据。输出list，每项一词。
建立词汇字典，对应词、编码。dictionary存储词、编码。reverse_dictionary是反过来的dictionary 编码、词。data是词list对应词编码上一步词list转编码。count存词汇、词频，重复数量少于49999,用'UNK'表示稀有词。
产生一个批次(batch)训练数据。定义generate_batch函数，输入batch_size、num_skip、skip_window，batch_size是每个batch大小，num_skips样本源端考虑次数，skip_window左右考虑词数，skip_window*2=num_skips。返回batch、label。batch形状[batch_size]，label形状[batch_size,1]，一个中心词预测周边词。
构建、训练模型。Skip-gram模型。
t-SNE降维呈现。Matplotlib绘制出图形。

t-SNE。流形学习(manifold Learning)。假设数据均匀采样于一个高维空间低维流形。流形学习，找到高维空间低维流形，求相庆嵌入映射，实现维数约简或数据可视化。线性流形学习如主成份分析(PCA)，非线性流形学习如特距特征映射(Isomap)、拉普拉斯特征映射(Laplacian eigenmaps,LE)、局部线性嵌入(Locally-linear embedding,LLE)。

嵌入投影。EMBEDDINGS面板，交互式可视化、分析高维数据。例子 https://github.com/tensorflow/models/blob/master/tutorials/embedding/word2vec_optimized.py 。

定义操作(operator,OP)，SkipgramWord2vec、NegTrainWord2vec。操作先编译，执行。TF_INC=$(python -c 'import tensorflow as tf;print(tf.sysconfig.get_include())') 。g++ -std=c++11 -shared word2vec_ops.cc word2vec_kernels.cc -o word2vec_ops.so -fPIC -I $TF_INC -O2 -D_GLIBCXX_USE_CXX11_ABI=0 。当前目录下生成word2vec_ops.so文件，执行word2vec_optimized.py，生成模型、日志文件，位于/tmp/，执行tensorboard --logdir=/tmp/ 。访问浏览器。

EMBEDDINGS面板左边工具栏，降维方式T-SNE、PCA、CUSTOM，二维、三维图像切换。t-SNE降维工具，手动调整Dimension(困惑度)、Learnign rate(学习率)，生成10000个点分布。右边，正则表达式匹配词，词间余弦距离、欧式距离关系。任意选择一个点，选择“isolate 101 points”按钮，展示100个空间上最近被选择点词，词数量。

参考资料：
《TensorFlow技术解析与实战》


