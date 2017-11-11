TF.Contrib，开源社区贡献，新功能，内外部测试，根据反馈意见改进性能，改善API友好度，API稳定后，移到TensorFlow核心模块。生产代码，以最新官方教程和API指南参考。

统计分布。TF.contrib.ditributions模块，Bernoulli、Beta、Binomial、Gamma、Ecponential、Normal、Poisson、Uniform等统计分布，统计研究、应用中常用，各种统计、机器学习模型基石，概率模型、图形模型依赖。

每个不同统计分布不同特征、函数，同样子类Distribution扩展。Distribution，建立和组织随机变量、统计分布基础类。is_continuous表明随机变量分布是否连续。allow_nan_states表示分布是否接受nan数据。sample()从分布取样。prob()计算随机变量密度函数。cdf()求累积分布函数。entropy()、mean()、std()、variance()得到统计分布平均值和方差等特征。自定义统计分布类要实现以上方程。

Gamma分布。contrib.distributions导入Gamma分布，初始化alpha、beta tf.constant，建立Gamma分布。batch_shap().eval()得到每个样本形状，get_batch_shape()，tf.TensorShape()。log_pdf()函数，值log转换概率密度函数。建立多维Gamma分布，传入多维alpha、beta参数。

Layer模块。Contrib.layer包含机器学习算法所需各种各样成份、部件，卷积层、批标准化层、机器学习指票、优化函数、初始器、特征列。

机器学习层。深度学习和计算机视觉二维平均池avg_pool2d。np.random.uniform建立宽高都是3几张图片，contrib.layers.avg_pool2d()对图片快速建立3x3二维平均池，outpu形状[5,1,1,3]，对每个3x3区域取计算平均值。

建立卷积层，contrib.layers.convolution2d()建立32个3x3过滤器卷积层，改stride、padding、activation_fn参数建立不同架构卷积层，使用不同卷咱们层激活函数。contrib.layers自动建立op名字，output.op.name值'Conv/Relu'，用了Conv层和ReLU激活函数。layer有自己对应op名字，每个op空间存储对应变量，contrib.framework.get_variables_by_name()得到对应op空间变量值。get_variables_by_name得到建立卷积层权重，权重形状是weights_shape值，[3,3,4,32]。

contrib.framework  arg_scope减少代码重复使用。layers.convolution2d及传入参数放到agr_scope，避免重复在多个地方传入。normalizer_fn和normalizer_params，标准化方程及参数。

len(tf.contrib.framework.get_variables('Conv/BatchNorm'))得到第一个Conv/BatchNorm层长度。

完全连接神经网络层fully_connected()。建立输入矩阵，fully_connected()建立输出7个神经单元神经网络层。tf.name_scope截下来运算放name_scope。fully_connected()传入scope。"fe/fc"层别号。传入outputs_collections，直接得到层输出。

repeat()重复用同样参数重复建立某个层。stack()用不同参数建立多个fully_connected()层。conv2d_transpose、conv2d_in_plane、separable_conv2d，参考官方文档。

损失函数。tf.contrib.losses模块，各种常用损失函数，二类分类、多类分类、回归模型等机器学习算法。

绝对差值。tf.constant建立predictions、targets数列。同样shape。选择性建立权重。losses.absolute_difference()计算预测损失值。

计算softmax交叉熵。多类分类机器学习模型。建立predictions、labels，多给。losses.softmax_cross_entropy()计算预测softmax交叉熵值。loss.eval()运行。loss.op.name得到TensorFlow自动赋值op名字，'softmax_cross_entropy_loss/value'。softmax_cross_entropy() label_smoothing平滑所有标识。

应用大部分分布稀疏，sparse_softmax_cross_entropy()提升计算效率。

特征列 Feature Column。tf.contrib.layers高阶特征列(Feature Column)API,和TF.Learn API结合使用，建立最适合自己数据的模型。

数据连续特征(continuous Feature)、类别特征(Categorical Feature)。连续数值特征称连续特征，可直接用在模型里。不连续类别特征，需要数值化，转换为一系列数值代表每个不同类别。learn.datasets API读入数据。

layers.FeatureColumn API定义特征列。real_valued_column()定义连续特征。

sparse_column_with_keys()处理类别特征，事先知道特征所有可能值。不知道所有可能值，用sparse_column_with_hash_bucket()转为特征列，哈希表。SparseColumn，直接在TF.Learn传入Estimator。

数据科学应用，连续特征可能需要被离散化，形成新类别特征，更好代表特征和目标分类类别之间关系。bucketized_column()将SparseColumn区间化。

部分应用，多个特征综合、交互与目标分类类别关系更紧密。多个特征相关，特征交互能建立更有效模型。crossed_column()建立交叉特征列。

特征列传入TF.Learn Estimator。fit()、predict()训练、评估模型。

取部分特征加权求和作新特征列，weighted_sum_from_feature_columns()实现。

Embeddings，嵌入向量。稀疏、高维类别特征向量，转换低维、稠密实数值向量，和连续特征向量联合，一起输入神经网络模型训练和优化损失函数。大部分文本识别，先将文本转换成嵌入向量。

contrib.layers模块 embedding_column()迅速把高维稀疏类别特征向量转为想要维数的嵌入向量。特征交互矩阵比较稀疏，级别比较高，转换后可以使模型更具有概括性更有效。传入TF.Learn Extimator进行模型建立、训练、评估。embedding_columns传入DNNLinearCombinedClassifier深度神经网络特征列。

许多实际稀疏高维数据，通常有空特征及无效ID，safe_enbedding_lookup_sparse()安全建立嵌入向量。tf.SparseTensor建立稀疏ID和稀疏权重。建立嵌入向量权重embedding_weights，取决词汇量大小、嵌入同量维数、shard数量。initializer.run()、eval()初始化嵌入向量权重。safe_embedding_lookup_sparse()将原来特征向量安全转换为低维、稠密特征向量。eval()收集到一个tuple。

性能分析器tfprof。分析模型架构、衡量系统性能。衡量模型参数、浮点运算、op执行时间、要求存储大小、探索模型结构。

命令安装tfprof命令行工具。bazel build -c opt tensorflow/contrib/trprof/...。

查询帮助文件。bazel-bin/tensorflow/contrib/tfprof/tools/tfprof/tfprof help。

执行互动模式，指定graph_path分析模型shape、参数。bazel-bin/tensorflow/contrib/tfprof/tools/tfprof/tfprof \--graph_path=graph.pbtxt。

graph_path、checkpoint_path查看checkpoint Tensor数据和对应值。bazel-bin/tensorflow/contrib/tfprof/tools/tfprof/tfprof \--graph_path=graph.pbtxt \--checkpoint_path=model.ckpt。

提供run_meta_path查看不同op请求存储、计时。bazel-bin/tensorflow/contrib/tfprof/tools/tfprof/tfprof \--graph_path=graph.pbtxt \--fun_meta_path=run_meta \--checkpoint_path=model.ckpt。

graph_path文件是GraphDef文本文件，用来在内存建立模型代表。tf.Supervisor写graph.pbtxt。tf.Graph.as_graph_def()或其他类似API存储模型定义到GraphDef文件。

run_meta_path文件是tensorflow::RunMetadata结果。得到模型每个op所需存储和时间消耗。

checkpoint_path是模型checkpoint包含所有checkpoint变量op类型、shape、值。

op_log_path是tensorflow::tfprof::OpLog结果，包含额外op信息，op组类别名字。

tfprof是CLI命令行工具，输入tfprof命令按回车，进入互动模式，再按回车看到命令行参数默认值。

调节参数，show_name_regexes查找符合正则式条件的scope名字。

tfprof提供两种类型分析：scope、graph。graph，查看op在graph里所花内存、时间。

参考资料：
《TensorFlow实战》


