预测编码(predictive coding)，向RNN输入大量序列，训练预测序列下一帧能力。语言建模(language modelling),预测一个句子中下一个单词的似然。生成文本，依据网络下一个单词分布抽样，训练结束，种子单词(seed word)送入RNN，观察预测的下一个单词，最可能单词输入RNN，重复，生成新内容。预测编码压缩训练网络任意序列所有重要信息。网络捕捉语法、语言规则，精确预测是语言下一个字符。

字符级语言建模，网络不仅学会构词，还学会拼写，网络输入维数更低，不必考虑未知单词，可以发明新单词。Andrew Karpathy 2015年应用RNN于字符级语言建模。https://github.com/karpathy/char-rnn 。

ArXiv.org托管计算机科学、数学、物理学、生物学等领域研究论文。提供基于Web可检索文献API。

依据给定搜索查询从ArXiv获取摘要，在构造方法，检查是否有旧摘要转储文件。有，直接使用，不调ArXiv API。执行新查询，删除或转移旧转储文件。可以优化检查已有文件与新类别、新关键词是否匹配。没有转储文件，调_fetch_all，生成行写入磁盘。

只在Machine Learning、Neural and Evolutionary Computing、Optimization and Control，搜索机器学习论文。只返回包含单词neural、network、deep元数据结果。

_fetch_all完成分页。每次查询，返回定量摘要，指定偏移量获到指定页结果。_fetch_page传入指定页面尺寸参数。参数很大，尝试一次性得到全部结果，严重影响查询效率。页面获取容错性更强，减小ArXiv API负载。

抓取结果XML格式，BeautifulSoup库提取摘要。执行命令 sudo -H pip3 install beautifulsoup4 安装。查看文章<entry> 标签，读取<summary>标签摘要文本。

定义任务，编写解析器获取数据集。预测编码模型，预测输入序列下一个字符，只有一个输入，构造方法sequence参数。参数对象，修改重要选项，复现实验。initial参数，默认值None，循环连接层初始内部活性值。TensorFlow隐状态初始化为零张量，语言模型采样时需要再定义。

数据处理，构造办玫数据、目标序列，引入时域差。时间步t，St输入，St+1输出。提供序列切片，切除第一帧或最后一帧。tf.slice切片运算，参数序列、各维起始索引元组、各维大小元组。sizes-1保持维度起始索引到终止索引所有元素不变。只关心第2维。

mask，尺寸batch_size*max_length张量，分量非0即1，取决帧是否被使用。属性length沿时间轴对mask求和，得到每个序列长度。mask、length属性对数据序列合法，与目标序列长度相同，不在数据序列上计算，包含最后一帧，没有下一字母可预测。数据张量最后一帧切除，包含填序帧，不包含大多数序列实际最后一帧。用mask对代价函数掩膜处理。

同时获得预测和最后循环活性值。之前仅返回预测值。最后活性值有效生成序列。forward返回两个张量元组，prediction、state只是方便外部访问。

每个时间步，模型从词汇表预测下一字母。分类问题，采用交叉熵代价函数，计算字符预测错误率。logprob属性，刻画模型对数空间正确下一字母分配概率。变换到对数空间取均值负交叉熵。结果返回线性空间，得到混淆度(perplexity)。混淆度表示模型在每个时间步猜测选项数目。完美模型，混淆度1。每个类别输出相同概率模型，混淆度为n。如果下一字母零概率，混淆度会变无穷大。预测概率箝位在很小正数和1之间。

固定长度序列，结果tf.reduce_mean。变长序列，与掩码相乘，屏蔽填充帧，沿帧尺寸聚合，每帧只有一个元素集，tf.reduce_sum聚合各帧为一个标量。

序列实际长度取平均每个序列各帧。使用每个序列长度最大值和1，避免空序列除数为0。tf.reduce_mean取平均批数据样本。

已构建模块整合，数据集、预处理步聚、网络模型。打印混淆度度量，周期性保存训练进展。加载数据集，数据流图定义输入，预处理数据集训练模型，追踪对数几率，相邻两次训练epoch评价时间计算打印混淆度。_init_or_load_session，tf.train.Saver保存数据流图tf.Variable当前值到检查点文件。实际点检查(checkpointing)在_evalution完成。寻找已有检查点文件另载。tf.train.get_checkpoint_state从检查点文件目录查找TensorFlow元数据文件。检查点文件，通过指定数字(epoch数)预先准备。加载检查点文件，Python正则表达式包re提取epoch数。

调用Training(get_params())()。20 epoch 1 小时。20 epochs*200 batches*100 examples*50 characters = 20M个字母。模型在混淆度1.5/字母时收敛。每个字母只需1.5位，可实现文本压缩。单词级语言模型，依据单词数取平均。乘以每个单词平均字符数。

利用训练好模型生成新的相似序列。从磁盘加载最新模型检查点，定义占位符，数据输入数据流图，生成新数据。

构造方法，创建预处理类实例，转化当前生成序列为NumPy向量，输入数据流图。占位符sequencec预留每批数据一个序列空间。序列长度为2。模型将除最后字符外所有字符作为输入，除首字符外所有字符作为目标。当前文本最后字符和序列任意第二字符输入模型。网络为第一字符预测结果，第二字符作目标值。获取循环神经网络最后活性值，初始化网络下次运行时状态。模型初始状态参数，使用过的GRUCell状态，尺寸rnn_layers*rnn_units向量。

__call__函数，采样文本序列逻辑。从一个采样种子开始，每次预测一个字符，当前文本送入网络。相同预处理类转换当前文本为填充NumPy块送入网络。批数据只有一个序列和一个输出帧，只关心索引[0, 0]预测结果。

_sample函数对softmaxl输出采样。选取序列最优预测，作为下一帧传入网络生成序列。实际不是只选择最可能下一帧，从RNN输出概率分布随机采样。高输出概率高单词更可能选中，输出概率低单词也可能被选中。

引入温度参数T，使softmax层输出分布预测更相似或更不同。在线性空间缩放输出，变换至指数空间并再次归一化。运用自然对数撤销。每个值除以选择温度值，得新应用softmax函数。

调用Sampling(get_params())('We', 500) 。捕捉数据内部统计依赖性。

    import requests
    import os
    from bs4 import BeautifulSoup

    from helpers import ensure_directory

    class ArxivAbstracts:

        ENDPOINT = 'http://export.arxiv.org/api/query'
        PAGE_SIZE = 100

        def __init__(self, cache_dir, categories, keywords, amount=None):
            self.categories = categories
            self.keywords = keywords
            cache_dir = os.path.expanduser(cache_dir)
            ensure_directory(cache_dir)
            filename = os.path.join(cache_dir, 'abstracts.txt')
            if not os.path.isfile(filename):
                with open(filename, 'w') as file_:
                    for abstract in self._fetch_all(amount):
                        file_.write(abstract + '\n')
            with open(filename) as file_:
                self.data = file_.readlines()

        def _fetch_all(self, amount):
            page_size = type(self).PAGE_SIZE
            count = self._fetch_count()
            if amount:
                count = min(count, amount)
            for offset in range(0, count, page_size):
                print('Fetch papers {}/{}'.format(offset + page_size, count))
                yield from self._fetch_page(page_size, count)

        def _fetch_page(self, amount, offset):
            url = self._build_url(amount, offset)
            response = requests.get(url)
            soup = BeautifulSoup(response.text)
            for entry in soup.findAll('entry'):
                text = entry.find('summary').text
                text = text.strip().replace('\n', ' ')
                yield text

        def _fetch_count(self):
            url = self._build_url(0, 0)
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'lxml')
            count = int(soup.find('opensearch:totalresults').string)
            print(count, 'papers found')
            return count

        def _build_url(self, amount, offset):
            categories = ' OR '.join('cat:' + x for x in self.categories)
            keywords = ' OR '.join('all:' + x for x in self.keywords)
            url = type(self).ENDPOINT
            url += '?search_query=(({}) AND ({}))'.format(categories, keywords)
            url += '&max_results={}&offset={}'.format(amount, offset)
            return url

    import random
    import numpy as np

    class Preprocessing:

        VOCABULARY = \
            " $%'()+,-./0123456789:;=?ABCDEFGHIJKLMNOPQRSTUVWXYZ" \
            "\\^_abcdefghijklmnopqrstuvwxyz{|}"

        def __init__(self, texts, length, batch_size):
            self.texts = texts
            self.length = length
            self.batch_size = batch_size
            self.lookup = {x: i for i, x in enumerate(self.VOCABULARY)}

        def __call__(self, texts):
            batch = np.zeros((len(texts), self.length, len(self.VOCABULARY)))
            for index, text in enumerate(texts):
                text = [x for x in text if x in self.lookup]
                assert 2 <= len(text) <= self.length
                for offset, character in enumerate(text):
                    code = self.lookup[character]
                    batch[index, offset, code] = 1
            return batch

        def __iter__(self):
            windows = []
            for text in self.texts:
                for i in range(0, len(text) - self.length + 1, self.length // 2):
                    windows.append(text[i: i + self.length])
            assert all(len(x) == len(windows[0]) for x in windows)
            while True:
                random.shuffle(windows)
                for i in range(0, len(windows), self.batch_size):
                    batch = windows[i: i + self.batch_size]
                    yield self(batch)

    import tensorflow as tf
    from helpers import lazy_property

    class PredictiveCodingModel:

        def __init__(self, params, sequence, initial=None):
            self.params = params
            self.sequence = sequence
            self.initial = initial
            self.prediction
            self.state
            self.cost
            self.error
            self.logprob
            self.optimize

        @lazy_property
        def data(self):
            max_length = int(self.sequence.get_shape()[1])
            return tf.slice(self.sequence, (0, 0, 0), (-1, max_length - 1, -1))

        @lazy_property
        def target(self):
            return tf.slice(self.sequence, (0, 1, 0), (-1, -1, -1))

        @lazy_property
        def mask(self):
            return tf.reduce_max(tf.abs(self.target), reduction_indices=2)

        @lazy_property
        def length(self):
            return tf.reduce_sum(self.mask, reduction_indices=1)

        @lazy_property
        def prediction(self):
            prediction, _ = self.forward
            return prediction

        @lazy_property
        def state(self):
            _, state = self.forward
            return state

        @lazy_property
        def forward(self):
            cell = self.params.rnn_cell(self.params.rnn_hidden)
            cell = tf.nn.rnn_cell.MultiRNNCell([cell] * self.params.rnn_layers)
            hidden, state = tf.nn.dynamic_rnn(
                inputs=self.data,
                cell=cell,
                dtype=tf.float32,
                initial_state=self.initial,
                sequence_length=self.length)
            vocabulary_size = int(self.target.get_shape()[2])
            prediction = self._shared_softmax(hidden, vocabulary_size)
            return prediction, state

        @lazy_property
        def cost(self):
            prediction = tf.clip_by_value(self.prediction, 1e-10, 1.0)
            cost = self.target * tf.log(prediction)
            cost = -tf.reduce_sum(cost, reduction_indices=2)
            return self._average(cost)

        @lazy_property
        def error(self):
            error = tf.not_equal(
                tf.argmax(self.prediction, 2), tf.argmax(self.target, 2))
            error = tf.cast(error, tf.float32)
            return self._average(error)

        @lazy_property
        def logprob(self):
            logprob = tf.mul(self.prediction, self.target)
            logprob = tf.reduce_max(logprob, reduction_indices=2)
            logprob = tf.log(tf.clip_by_value(logprob, 1e-10, 1.0)) / tf.log(2.0)
            return self._average(logprob)

        @lazy_property
        def optimize(self):
            gradient = self.params.optimizer.compute_gradients(self.cost)
            if self.params.gradient_clipping:
                limit = self.params.gradient_clipping
                gradient = [
                    (tf.clip_by_value(g, -limit, limit), v)
                    if g is not None else (None, v)
                    for g, v in gradient]
            optimize = self.params.optimizer.apply_gradients(gradient)
            return optimize

        def _average(self, data):
            data *= self.mask
            length = tf.reduce_sum(self.length, 0)
            data = tf.reduce_sum(data, reduction_indices=1) / length
            data = tf.reduce_mean(data)
            return data

        def _shared_softmax(self, data, out_size):
            max_length = int(data.get_shape()[1])
            in_size = int(data.get_shape()[2])
            weight = tf.Variable(tf.truncated_normal(
                [in_size, out_size], stddev=0.01))
            bias = tf.Variable(tf.constant(0.1, shape=[out_size]))
            # Flatten to apply same weights to all time steps.
            flat = tf.reshape(data, [-1, in_size])
            output = tf.nn.softmax(tf.matmul(flat, weight) + bias)
            output = tf.reshape(output, [-1, max_length, out_size])
            return output

    import os
    import re
    import tensorflow as tf
    import numpy as np

    from helpers import overwrite_graph
    from helpers import ensure_directory
    from ArxivAbstracts import ArxivAbstracts
    from Preprocessing import Preprocessing
    from PredictiveCodingModel import PredictiveCodingModel

    class Training:

        @overwrite_graph
        def __init__(self, params, cache_dir, categories, keywords, amount=None):
            self.params = params
            self.texts = ArxivAbstracts(cache_dir, categories, keywords, amount).data
            self.prep = Preprocessing(
                self.texts, self.params.max_length, self.params.batch_size)
            self.sequence = tf.placeholder(
                tf.float32,
                [None, self.params.max_length, len(self.prep.VOCABULARY)])
            self.model = PredictiveCodingModel(self.params, self.sequence)
            self._init_or_load_session()

        def __call__(self):
            print('Start training')
            self.logprobs = []
            batches = iter(self.prep)
            for epoch in range(self.epoch, self.params.epochs + 1):
                self.epoch = epoch
                for _ in range(self.params.epoch_size):
                    self._optimization(next(batches))
                self._evaluation()
            return np.array(self.logprobs)

        def _optimization(self, batch):
            logprob, _ = self.sess.run(
                (self.model.logprob, self.model.optimize),
                {self.sequence: batch})
            if np.isnan(logprob):
                raise Exception('training diverged')
            self.logprobs.append(logprob)

        def _evaluation(self):
            self.saver.save(self.sess, os.path.join(
                self.params.checkpoint_dir, 'model'), self.epoch)
            self.saver.save(self.sess, os.path.join(
                self.params.checkpoint_dir, 'model'), self.epoch)
            perplexity = 2 ** -(sum(self.logprobs[-self.params.epoch_size:]) /
                            self.params.epoch_size)
            print('Epoch {:2d} perplexity {:5.4f}'.format(self.epoch, perplexity))

        def _init_or_load_session(self):
            self.sess = tf.Session()
            self.saver = tf.train.Saver()
            checkpoint = tf.train.get_checkpoint_state(self.params.checkpoint_dir)
            if checkpoint and checkpoint.model_checkpoint_path:
                path = checkpoint.model_checkpoint_path
                print('Load checkpoint', path)
                self.saver.restore(self.sess, path)
                self.epoch = int(re.search(r'-(\d+)$', path).group(1)) + 1
            else:
                ensure_directory(self.params.checkpoint_dir)
                print('Randomly initialize variables')
                self.sess.run(tf.initialize_all_variables())
                self.epoch = 1

    from Training import Training
    from get_params import get_params

    Training(
        get_params(),
        cache_dir = './arxiv',
        categories = [
            'Machine Learning',
            'Neural and Evolutionary Computing',
            'Optimization'
        ],
        keywords = [
            'neural',
            'network',
            'deep'
        ]
        )()

    import tensorflow as tf
    import numpy as np

    from helpers import overwrite_graph
    from Preprocessing import Preprocessing
    from PredictiveCodingModel import PredictiveCodingModel

    class Sampling:

        @overwrite_graph
        def __init__(self, params):
            self.params = params
            self.prep = Preprocessing([], 2, self.params.batch_size)
            self.sequence = tf.placeholder(
                tf.float32, [1, 2, len(self.prep.VOCABULARY)])
            self.state = tf.placeholder(
                tf.float32, [1, self.params.rnn_hidden * self.params.rnn_layers])
            self.model = PredictiveCodingModel(
                self.params, self.sequence, self.state)
            self.sess = tf.Session()
            checkpoint = tf.train.get_checkpoint_state(self.params.checkpoint_dir)
            if checkpoint and checkpoint.model_checkpoint_path:
                tf.train.Saver().restore(
                    self.sess, checkpoint.model_checkpoint_path)
            else:
               print('Sampling from untrained model.')
            print('Sampling temperature', self.params.sampling_temperature)

        def __call__(self, seed, length=100):
            text = seed
            state = np.zeros((1, self.params.rnn_hidden * self.params.rnn_layers))
            for _ in range(length):
                feed = {self.state: state}
                feed[self.sequence] = self.prep([text[-1] + '?'])
                prediction, state = self.sess.run(
                    [self.model.prediction, self.model.state], feed)
                text += self._sample(prediction[0, 0])
            return text

        def _sample(self, dist):
            dist = np.log(dist) / self.params.sampling_temperature
            dist = np.exp(dist) / np.exp(dist).sum()
            choice = np.random.choice(len(dist), p=dist)
            choice = self.prep.VOCABULARY[choice]
            return choice

参考资料：
《面向机器智能的TensorFlow实践》


