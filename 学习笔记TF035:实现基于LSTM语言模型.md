神经结构进步、GPU深度学习训练效率突破。RNN，时间序列数据有效，每个神经元通过内部组件保存输入信息。

卷积神经网络，图像分类，无法对视频每帧图像发生事情关联分析，无法利用前帧图像信息。RNN最大特点，神经元某些输出作为输入再次传输到神经元，可以利用之前信息。

xt是RNN输入，A是RNN节点，ht是输出。对RNN输入数据xt，网络计算得输出结果ht，某些信息(state,状态)传到网络输入。输出ht与label比较得误差，用梯度下降(Gradient Descent)和Back-Propagation Through Time(BPTT)方法训练网络。BPTT，用反向传播求解梯度，更新网络参数权重。Real_Time Recurrent Learning(RTRL)，正向求解梯度，计算复杂度高。介于BPTT和RTRL之间混合方法，缓解时间序列间隔过长带来梯度弥散问题。

RNN循环展开串联结构，类似系列输入x和系列输出串联普通神经网络，上层神经网络传递信息给下层。适合时间序列数据处理分析。展开每层级神经网络，参数相同，只需要训练一层RNN参数。共享参数思想与卷积神经网络权值共享类似。

RNN处理整个时间序列信息，记忆最深是最后输入信号。前信号强度越来越低。Long Sort Term Memory(LSTM)突破，语音识别、文本分类、语言模型、自动对话、机器翻译、图像标注领域。

长程依赖(Long-term Dependencies)，传统RNN关键缺陷。LSTM，Schmidhuber教授1997年提出，解决长程依赖，不需要特别复杂调试超参数，默认记住长期信息。

LSTM内部结构，4层神经网络，小圆圈是point-wise操作(向量加法、点乘等)，小矩形是一层可学习参数神经网络。LSTM单元上直线代表LSTM状态state，贯穿所有串联LSTM单元，从第一个流向最后一个，只有少量线性干预和改变。状态state传递，LSTM单凶添加或删减信息，LSTM Gates控制信息流修改操作。Gates包含Sigmoid层和向量点乘操作。Sigmoid层输出0到1间值，直接控制信息传递比例。0不允许信息传递，1让信息全部通过。每个LSTM单元3个Gates，维护控制单元状态信息。状态信息储存、修改，LSTM单元实现长程记忆。

RNN变种，LSTM，Gated Recurrent Unit(GRU)。GRU结构，比LSTM少一个Gate。计算效率更高(每个单元计算节约几个矩阵运算)，占用内存少。GRU收敛所需迭代更少，训练速度更快。

循环神经网络，自然语言处理，语言模型。语言模型，预测语句概率模型，给定上下文语境，历史出现单词，预测下一个单词出现概率，NLP、语音识别、机器翻译、图片标注任务基础关键。Penn Tree Bank(PTB)常用数据集，质量高，不大，训练快。《Recurrent Neural Network Regularization》。

下载PTB数据集，解压。确保解压文件路径和Python执行路径一致。1万个不同单词，有句尾标记，罕见词汇统一处理为特殊字符。wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examplex.tgz 。tar xvf simple-examples.tgz 。

下载TensorFlow Models库(git clone https://github.com/tensorflow/models.git)，进入目录models/tutorials/rnn/ptb(cd)。载入常用库，TensorFlow Models PTB reader，读取数据内容。单词转唯一数字编码。

定义语言模型处理输入数据class，PTBInput。初始化方法__init__()，读取参数config的batch_size、num_steps到本地变量。num_steps，LSTM展开步数(unrolled steps of LSTM)。计算epoth size ，epoch内训练迭代轮数，数据长度整除batch_size、num_steps。reader.ptb_producer获取特征数据input_data、label数据targets。每次执行获取一个batch数据。

定义语言模型class，PTBModel。初始化函数__init__()，参数，训练标记is_training、配置参数config、PTBInput类实例input_。读取input_的batch_size、num_steps，读取config的hidden_size(LSTM节点数)、vocab_size(词汇表大小)到本地变量。

tf.contrib.rnn.BasicLSTMCell设置默认LSTM单元，隐含节点数hidden_size、gorget_bias(forget gate bias) 0，state_is_tuple True，接受返回state是2-tuple形式。训练状态且Dropout keep_prob小于1,1stm_cell接Dropout层，tf.contrib.rnn.DropoutWrapper函数。RNN堆叠函数 tf.contrib.rnn.MultiRNNCell 1stm_cell多层堆叠到cell，堆叠次数 config num_layers，state_is_truple设True,cell.zero_state设LSTM单元初始化状态0。LSTM单元读放单词，结合储存状态state计算下一单词出现概率分布，每次读取单词，状态state更新。

创建网络词嵌入embedding，将one-hot编码格式单词转向量表达形式。with tf.device("/cpu:0") 计算限定CPU进行。初始化embedding矩阵，行数设词汇表数vocab_size，列数(单词向量表达维数)hidden_size，和LST单元陷含节点数一致。训练过程，embedding参数优化更新。tf.nn.embedding_lookup查询单对应向量表达获得inputs。训练状态加一层Dropout。

定义输出outputs，tf.variable_scope设名RNN。控制训练过程，限制梯度反向传播展开步数固定值，num_steps.设置循环长度 num-steps，控制梯度传播。从第2次循环，tf.get_varible_scope.reuse_variables设置复用变量。每次循环，传入inputs、state到堆叠LSTM单元(cell)。inputs 3维度，第1维 batch第几个样本，第2维 样本第几个单词，第3维 单词向量表达维度。inputs[:,time_step,:] 所有样本第time_step个单词。输出cell_output和更新state。 结果cell_output添加输出列表outputs。

tf.concat串接output内容，tf.reshape转长一维向量。Softmax层，定义权重softmax_w、偏置softmax_b。tf.matmul 输出output乘权重加偏置得网络最后输出logits。定久损失loss，tf.contrib.legacy_seq2seq.sequence_loss_by_example计算输出logits和targets偏差。sequence_loss，target words average negative log probability，定义loss=1/N add i=1toN ln Ptargeti。tf.reduce_sum汇总batch误差，计算平均样本误差cost。保留最终状态final_state。不是训练状态直接返回。

定义学习速率变量lr，设不可训练。tf.trainable_variables获取全部可训练参数tvars。针对cost，计算tvars梯度，tf.clip_by_global_norm设梯度最大范数，起正则化效果。Gradient Clipping防止Gradient Explosion梯度爆炸问题。不限制梯度，迭代梯度过大，训练难收敛。定义优化器Gradient Descent。创建训练操作_train_op，optimizer.apply_gradients，clip过梯度用到所有可训练参数tvars，tf.contrib.framework.get_or_create_global_step生成全局统一训练步数。

设置_new_lr(new learning rate) placeholder控制学习速率。定义操作_lr_update，tf.assign 赋_new_lr值给当前学习速率_lr。定义assign_lr函数，外部控制模型学习速率，学习速率值传入_new_lr placeholder，执行_update_lr操作修改学习速率。

定义PTBModel class property。Python @property装饰器，返回变量设只读，防止修改变量引发问题。input、initial_state、cost、final_state、lr、train_op。

定义模型设置。init_scale，网络权重初始scale。learning_rate，学习速率初始值。max_grad_norm，梯度最大范数。num_lyers，LSTM堆叠层数。num_steps，LSTM梯度反向传播展开步数。hidden_size，LSTM内隐含节点数。max_epoch，初始学习速率可训练epoch数，需要调整学习速率。max_max_epoch，总共可训练epoch数。keep_prob，dropout层保留节点比例。lr_decay学习速率衰减速度。batch_size，每个batch样本数量。

MediumConfig中型模型，减小init_scale，希望权重初值不要过大，小有利温和训练。学习速率、最大梯度范数不变，LSTM层数不变。梯度反向传播展开步数num_steps从20增大到35。hidden_size、max_max_epoch增大3倍。设置dropout keep_prob 0.5。学习迭代次数增大，学习速率衰减速率lr_decay减小。batch_size、词汇表vocab_size不变。

LargeConfig大型模型，进一步缩小init_scale。放宽最大梯度范数max_grad_norm到10。hidden_size提升到1500。max_epoch、max_max_epoch增大。keep_prob因模型复杂度上升继续下降。学习速率衰减速率lr_decay进一步减小。

TestConfig测试用。参数尽量最小值。

定义训练epoch数据函数run_epoch。记录当前时间，初始化损失costs、迭代数据iters，执行model.initial_state初始化状态，获得初始状态。创建输出结果字典表fetches，包括cost、final_state。如果有评测操作，也加入fetches。训练循环，次数epoch_size。循环，生成训练feed_dict，全部LSTM单元state加入feed_dict，传入feed_dict，执行fetches训练网络，拿到cost、state。累加cost到costs，累加num_steps到iters。每完成10%epoch，展示结果，当前epoch进度，perplexity(平均cost自然常数指数，语言模型比较性能重要指标，越低模型输出概率分布在预测样本越好)，训练速度(单词数每秒)。返回perplexity函数结果。

reader.ptb_raw_data读取解压后数据，得训练数据、验证数据、测试数据。定义训练模型配置SmallConfig。测试配置eval_config需和训练配置一致。测试配置batch_size、num_steps 1。

创建默认Graph，tf.random_uniform_initializer设置参数初始化器，参数范围在[-init_scale,init_scale]之间。PTBInput和PTBModel创建训练模型m，验证模型mvalid，测试模型mtest。训练、验证模型用config，测试模型用测试配置eval_config。

tf.train.supervisor()创建训练管理器sv，sv.managed_session创建默认session，执行训练多个epoch数据循环。每个epoch循环，计算累计学习速率衰减值，只需计算超过max_epoch轮数，求lr_decay超出轮数次幂。初始学习速率乘累计衰减速，更新学习速率。循环内执行epoch训练和验证，输出当前学习速率、训练验证集perplexity。完成全部训练，计算输出模型测试集perplexity。

SmallConfig小型模型，i7 6900K GTX 1080 训练速率21000单词每秒，最后epoch，训练集36.9 perplexity，验证集122.3、测试集116.7。

中型模型，训练集48.45，验证集86.16、测试集82.07。大型模型，训练集37.87，验证集82.62、测试集78.29。

LSTM存储状态，依靠状态对当前输入处理分析预测。RNN、LSTM赋预神经网络记忆和储存过往信息能力，模仿人类简单记忆、推理功能。注意力(attention)机制是RNN、NLP领域研究热点，机器更好模拟人脑功能。图像标题生成任务，注意力机制RNN对区域图像分析，生成对应文字描述。《Show,Attend and Tell:Neural Image Caption Generation with Visual Attention》。

    import time
    import numpy as np
    import tensorflow as tf
    import reader
    #flags = tf.flags
    #logging = tf.logging
    #flags.DEFINE_string("save_path", None,
    #                    "Model output directory.")
    #flags.DEFINE_bool("use_fp16", False,
    #                  "Train using 16-bit floats instead of 32bit floats")
    #FLAGS = flags.FLAGS
    #def data_type():
    #  return tf.float16 if FLAGS.use_fp16 else tf.float32
    class PTBInput(object):
      """The input data."""
      def __init__(self, config, data, name=None):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
        self.input_data, self.targets = reader.ptb_producer(
        data, batch_size, num_steps, name=name)
    class PTBModel(object):
      """The PTB model."""
      def __init__(self, is_training, config, input_):
        self._input = input_
        batch_size = input_.batch_size
        num_steps = input_.num_steps
        size = config.hidden_size
        vocab_size = config.vocab_size
        # Slightly better results can be obtained with forget gate biases
        # initialized to 1 but the hyperparameters of the model would need to be
        # different than reported in the paper.
        def lstm_cell():
          return tf.contrib.rnn.BasicLSTMCell(
              size, forget_bias=0.0, state_is_tuple=True)
    attn_cell = lstm_cell
        if is_training and config.keep_prob < 1:
          def attn_cell():
            return tf.contrib.rnn.DropoutWrapper(
                lstm_cell(), output_keep_prob=config.keep_prob)
        cell = tf.contrib.rnn.MultiRNNCell(
            [attn_cell() for _ in range(config.num_layers)], state_is_tuple=True)
        self._initial_state = cell.zero_state(batch_size, tf.float32)
        with tf.device("/cpu:0"):
          embedding = tf.get_variable(
              "embedding", [vocab_size, size], dtype=tf.float32)
      inputs = tf.nn.embedding_lookup(embedding, input_.input_data)
    if is_training and config.keep_prob < 1:
      inputs = tf.nn.dropout(inputs, config.keep_prob)
    # Simplified version of models/tutorials/rnn/rnn.py's rnn().
    # This builds an unrolled LSTM for tutorial purposes only.
    # In general, use the rnn() or state_saving_rnn() from rnn.py.
    #
    # The alternative version of the code below is:
    #
    # inputs = tf.unstack(inputs, num=num_steps, axis=1)
    # outputs, state = tf.nn.rnn(cell, inputs,
    #                            initial_state=self._initial_state)
    outputs = []
    state = self._initial_state
    with tf.variable_scope("RNN"):
      for time_step in range(num_steps):
        if time_step > 0: tf.get_variable_scope().reuse_variables()
        (cell_output, state) = cell(inputs[:, time_step, :], state)
        outputs.append(cell_output)
        output = tf.reshape(tf.concat(outputs, 1), [-1, size])
        softmax_w = tf.get_variable(
        "softmax_w", [size, vocab_size], dtype=tf.float32)
        softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=tf.float32)
        logits = tf.matmul(output, softmax_w) + softmax_b
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [logits],
            [tf.reshape(input_.targets, [-1])],
            [tf.ones([batch_size * num_steps], dtype=tf.float32)])
        self._cost = cost = tf.reduce_sum(loss) / batch_size
        self._final_state = state
        if not is_training:
          return
        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                      config.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self._lr)
        self._train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=tf.contrib.framework.get_or_create_global_step())
        self._new_lr = tf.placeholder(
            tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)
      def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})
      @property
      def input(self):
        return self._input
      @property
      def initial_state(self):
        return self._initial_state
      @property
      def cost(self):
        return self._cost
      @property
      def final_state(self):
        return self._final_state
      @property
      def lr(self):
        return self._lr
      @property
      def train_op(self):
        return self._train_op
    class SmallConfig(object):
      """Small config."""
      init_scale = 0.1
      learning_rate = 1.0
      max_grad_norm = 5
      num_layers = 2
      num_steps = 20
      hidden_size = 200
      max_epoch = 4
      max_max_epoch = 13
      keep_prob = 1.0
      lr_decay = 0.5
      batch_size = 20
      vocab_size = 10000
    class MediumConfig(object):
      """Medium config."""
      init_scale = 0.05
      learning_rate = 1.0
      max_grad_norm = 5
      num_layers = 2
      num_steps = 35
      hidden_size = 650
      max_epoch = 6
      max_max_epoch = 39
      keep_prob = 0.5
      lr_decay = 0.8
      batch_size = 20
      vocab_size = 10000
    class LargeConfig(object):
      """Large config."""
      init_scale = 0.04
      learning_rate = 1.0
      max_grad_norm = 10
      num_layers = 2
      num_steps = 35
      hidden_size = 1500
      max_epoch = 14
      max_max_epoch = 55
      keep_prob = 0.35
      lr_decay = 1 / 1.15
      batch_size = 20
      vocab_size = 10000
    class TestConfig(object):
      """Tiny config, for testing."""
      init_scale = 0.1
      learning_rate = 1.0
      max_grad_norm = 1
      num_layers = 1
      num_steps = 2
      hidden_size = 2
      max_epoch = 1
      max_max_epoch = 1
      keep_prob = 1.0
      lr_decay = 0.5
      batch_size = 20
      vocab_size = 10000
    def run_epoch(session, model, eval_op=None, verbose=False):
      """Runs the model on the given data."""
      start_time = time.time()
      costs = 0.0
      iters = 0
      state = session.run(model.initial_state)
      fetches = {
          "cost": model.cost,
          "final_state": model.final_state,
      }
      if eval_op is not None:
        fetches["eval_op"] = eval_op
      for step in range(model.input.epoch_size):
        feed_dict = {}
        for i, (c, h) in enumerate(model.initial_state):
          feed_dict[c] = state[i].c
          feed_dict[h] = state[i].h
        vals = session.run(fetches, feed_dict)
        cost = vals["cost"]
        state = vals["final_state"]
        costs += cost
        iters += model.input.num_steps
        if verbose and step % (model.input.epoch_size // 10) == 10:
          print("%.3f perplexity: %.3f speed: %.0f wps" %
                (step * 1.0 / model.input.epoch_size, np.exp(costs / iters),
                 iters * model.input.batch_size / (time.time() - start_time)))
      return np.exp(costs / iters)
    raw_data = reader.ptb_raw_data('simple-examples/data/')
    train_data, valid_data, test_data, _ = raw_data
    config = SmallConfig()
    eval_config = SmallConfig()
    eval_config.batch_size = 1
    eval_config.num_steps = 1
    with tf.Graph().as_default():
      initializer = tf.random_uniform_initializer(-config.init_scale,
                                              config.init_scale)
      with tf.name_scope("Train"):
        train_input = PTBInput(config=config, data=train_data, name="TrainInput")
        with tf.variable_scope("Model", reuse=None, initializer=initializer):
          m = PTBModel(is_training=True, config=config, input_=train_input)
          #tf.scalar_summary("Training Loss", m.cost)
          #tf.scalar_summary("Learning Rate", m.lr)
      with tf.name_scope("Valid"):
        valid_input = PTBInput(config=config, data=valid_data, name="ValidInput")
        with tf.variable_scope("Model", reuse=True, initializer=initializer):
          mvalid = PTBModel(is_training=False, config=config, input_=valid_input)
          #tf.scalar_summary("Validation Loss", mvalid.cost)
      with tf.name_scope("Test"):
        test_input = PTBInput(config=eval_config, data=test_data, name="TestInput")
        with tf.variable_scope("Model", reuse=True, initializer=initializer):
          mtest = PTBModel(is_training=False, config=eval_config,
                       input_=test_input)
      sv = tf.train.Supervisor()
      with sv.managed_session() as session:
        for i in range(config.max_max_epoch):
          lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
          m.assign_lr(session, config.learning_rate * lr_decay)
          print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
          train_perplexity = run_epoch(session, m, eval_op=m.train_op,
                                   verbose=True)
          print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
          valid_perplexity = run_epoch(session, mvalid)
          print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))
        test_perplexity = run_epoch(session, mtest)
        print("Test Perplexity: %.3f" % test_perplexity)
         # if FLAGS.save_path:
         #   print("Saving model to %s." % FLAGS.save_path)
         #   sv.saver.save(session, FLAGS.save_path, global_step=sv.global_step)
    #if __name__ == "__main__":
    #  tf.app.run()

参考资料：
《TensorFlow实战》


