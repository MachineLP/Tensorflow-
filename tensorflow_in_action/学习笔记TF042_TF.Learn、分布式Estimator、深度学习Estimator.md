TF.Learn，TensorFlow重要模块，各种类型深度学习及流行机器学习算法。TensorFlow官方Scikit Flow项目迁移，谷歌员工Illia Polosukhin、唐源发起。Scikit-learn代码风格，帮助数据科学从业者更好、更快适应接受TensorFlow代码。囊括许多TensorFlow代码、设计模式，用户更快搭建机器学习模型实现应用。避免大量代码重复，把精力放在搭建更精确模型。与其他contrib模块无逢结合。

分布式Estimator。Estimator，各种各样机器学习、深度学习类。可以直接用这些高阶类，也可以根据实际应用需求快速创建子类。graph_actions模块，Estimator在训练、评估模型复杂分布式逻辑被实现、浓缩，不需要复杂Supervisor、Coordinator分布式训练具体实现细节、逻辑。

Estimator接受自定义模型，函数答名(入参字段->返回字段):(1)(features,targets)->(predictions,loss,train_op)。(2)(features,targets,mode)->(predictions,loss,train_op)。(3)(features,targets,mode,params)->(predictions,loss,train_op)。

自定义模型接受两个参数：features和targets。features，数据特征。targets数据特征每行目标或分类标识。tf.one_hot对targets独热编码(One-hot Encoding)。layers.stack叠加多层layers.fully_connected完全连接深度神经网络，每层分别10､20､10个隐藏节点，不同层转换、训练，得到新数据特征。models.logistic_regression_zero_init加一层，0初始参数值逻辑回归模型，得到预测值、损失值。contrib.layers.optimize_loss函数优化损失值，根据需要选择不同优化函数和学习速率。optimize_loss训练算子(Training Operator)，每次训练迭代优化模型参数和决定模型发展方向。返回预测值、预测概率，或其中一个。

iris数据分类。Scikit-learn datasets引入数据，cross_validation数据分训练、评估。my_model放learn.Estimator，Scikit-learn风格fit、predict函数。快速定义自己的模型函数，直接利用Estimator各种功能，直接分布式模型训练，不用担心实现细节。

模式(Mode)定义函数，常用模式training、evaluation、prediction，可以在ModeKeys找到。加条件语句实现复杂逻辑。params调节参数，fit函数可以给更多参数。

建立机器学习Estimator。BaseEstimator最抽象最基本实现TensorFlow模型训练、评估类。fit()模型训练，partial_fit()线上训练，evaluate()评估模型，predict()使用模型预测新数据。graph_actions复杂逻辑模型训练、预测。SuperVisor、Coordinator、QueueRunner，分布式训练、预测。learn.DataFeeder、learn.DataFrame类自动识别、处理、迭代不同类型数据。estimators.tensor_signature对数据进行兼容性判断(稀疏张量Sparse Tensor)，数据读入更方便、稳定。BaseEstimator对learn.monitors及模型存储进行初始化设置。learn.monitors监测模型训练。

BaseEstimator，_get_train_ops()、_get_eval_ops()、_get_predict_ops()子类实现。

Estimator，_get_train_ops()接受features、targets参数，自定义模型函数返回Operation、损失Tensor Tuple，在每个训练迭代优化模型参数。非监督学习模型Estimator，忽略targets。

_get_eval_ops()，BaseEstimator子类自定义metrics评估每个模型训练迭代。contrib.metrics。自定义metrics函数返回一个Tensor对象Python字黄代表评估Operation，每次迭代用到。

自定义模型对新数据预测、计算损失值，ModeKeys EVAL表明函数只在评估用。contrib.metrics模块，streaming_mean对loss计算平均流，之前计算过平均值加这次迭代损失值再计算平均值。

_get_predict_ops()实现自定义预测。对预测结果进一步处理。预测概率转换简单预测结果，概率平滑加工(Smooting)。函数返回Tensor对象Python字典代表预测Operation。Estimator predict()函数，Estimator分布式功能。非监督模型，类似Sckkit-learn transform()函数。

逻辑回归(LogisticRegressor)，Estimator提供绝大部分实现，LogisticRegressor只需提供自己的metrics(AUC、accuracy、precision、recall，处理二分类问题)，快速在LogiticRegressor基础写子类实现个性化二分类Estimator，不需要关心其他逻辑实现。

TF.Learn 随机森林模型TensorForestEstimator许多细节实现在contrib.tensor_forest。只利用、暴露高阶需要用到成分到TensorForestEstimator。超参数通过contrib.tensor_forest.ForestHParams传到构造函数params，构造函数params.fill()建造随机森林TensorFlow图，tensor_forest.RandomForestGraphs。

实现复杂，需要高效率，细节用C++实现单独Kernel。_get_predict_ops()函数，tensor_forest内部C++实现data_ops.ParseDataTensorOrDict()函数检测、转换读入数据到可支持数据类型，RandomForestGraphs inference_graph函数得到预测Operation。

_get_train_ops()、_get_eval_ops()函数分别调用RandomForestGraphs.trainning_loss()、RandomForestGraphs.onference_graph()函数，data_ops.ParseDataTensorOrDict、data_ops.ParseLabelTensorOrDict分别检测、转换features、targets到兼容数据类型。

调节RunConfig运行时参数。RunConfig，TF.Learn类，调节程序运行时参数。num_cores选择使用核数量，num_ps_replicas调节参数服务器数量，gpu_memory_fraction控制使用GPU存储百分比。

RunConfig master参数，指定训练模型主服务器地址，task设置任务ID，每个任务ID控制一个训练模型参数服务器replica。

初始化一个RunConfig对象，传进Estimator。RunConfig参数默认值在本地运行简单模型，只用一个任务ID，80%GPU存储参数传进Estimator。运行时参数会自动运用，不用担心ConfigProto、GPUOptions细节。快速改变参数实现分布式模型训练、参数服务器使用。

Experiment，简单易用建立模型实验类，建模所需所有信息，Estimator、训练数据、评估数据、平估指标、监督器、评估频率。可以选择当地运行，可以和RunConfig配合分布式试验。LearnRunner，方便做实验。

tf.app.flags定义可以从命令行传入参数，数据、模型、输出文件路径、训练、评估步数。schedule试验类型。local_run()当地试验。run_std_server()标准服务器试验。master_grpc_url主要GRPC TensorFlow服务器。num_parameter_servers参数服务器数量。

建立Experiment对象函数，FLAGS建立RunConfig，机器学习模型Estimator，建立广度深度结合分类器(DNNLinearCombinedClassifier)。input_train_fn、input_test_fn，数据来源、提供训练、评估数据。

create_experiment_fn()函数传入LearnRunner进行不同类型试验，当地、服务器试验。试验结果存储到不同路径。

深度学习Estimator。TF.Learn包含简单易用深度神经网络Estimator。分类问题DNNClassifier。_input_fn()建立数据，layers模块建立特征列。

特征列、每层隐藏神经单元数、标识类别数传入DNNClassifier，迅速建立深度神经网络模型。

fit()、evaluate()方法模型训练、评估。

每行数据都有权重。DNNClassfier，指定一列为权重列，自动分配到训练过程。四行数据，每行不同权重。权重列、特征列放入features。

DNNClassifier表明权重列列名 w，特征列列名x(x转换特征列)。

传入自定义metrics方程_my_metric_op()，操作predictions、targets进行metrics计算。只考虑二分类问题，tf.slice()剪切predictions第二列作预测值。

tf.slice()传入输入矩阵input，剪切开始元素begin，剪切Tensor形状size，size[i]代表第i个维度想剪切矩阵shape。

根据需求任意在predictions、targets操作实现想要metrics计算，evaluate()传入metrics函数，TF.Learn根据metrics评估模型。

evaluate()可以提供多个metrics，_my_metric_op自定义，tr.contrib自带。

optimizer提供自定义函数，定义自己的优化函 ，包含指数递减学习率。

tf.contrib.framework.get_or_create_global_step()得到目前模型训练到达全局步数。tf.train.exponential_decay()对学习率指数递减，避免爆炸梯度。

广度深度模型，DNNLinearCombinedClassifier。谷歌广泛用在各种机器学习应用，深度神经网络和逻辑回归结合，不同特征通过两种不同方式结合，更能体现应用意义和更有效推荐结果。类似Kaggle竞赛Ensemble。

更多参数，与DNNClassifier、LinearClassifier不同特征列选择。

gender、education、relationship、workclass转换为FeatureColumn。分wide_columns、deep_columns。wide_columns用在LinearClassifier，deep_columns用在DNNClassifier。分别传入DNNLinearCombinedClassifier建立广度深度模型。具有线性特征，也具有深度神经网络特征。

参考资料：
《TensorFlow实战》


