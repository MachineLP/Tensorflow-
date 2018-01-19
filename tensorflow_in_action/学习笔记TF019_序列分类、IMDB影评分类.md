序列分类，预测整个输入序列的类别标签。情绪分析，预测用户撰写文字话题态度。预测选举结果或产品、电影评分。

国际电影数据库(International Movie Database)影评数据集。目标值二元，正面或负面。语言大量否定、反语、模糊，不能只看单词是否出现。构建词向量循环网络，逐个单词查看每条评论，最后单词话性值训练预测整条评论情绪分类器。

斯担福大学人工智能实验室的IMDB影评数据集: http://ai.stanford.edu/~amaas/data/sentiment/ 。压缩tar文档，正面负面评论从两个文件夹文本文件获取。利用正则表达式提取纯文本，字母全部转小写。

词向量嵌入表示，比独热编码词语语义更丰富。词汇表确定单词索引，找到正确词向量。序列填充相同长度，多个影评数据批量送入网络。

序列标注模型，传入两个占位符，一输入数据data或序列，二目标值target或情绪。传入配置参数params对象，优化器。

动态计算当前批数据序列长度。数据单个张量形式，各序列以最长影评长度补0。绝对值最大值缩减词向量。零向量，标量0。实型词向量，标量大于0实数。tf.sign()离散为0或1。结果沿时间步相加，得到序列长度。张量长度与批数据容量相同，标量表示序列长度。

使用params对象定义单元类型和单元数量。length属性指定向RNN提供批数据最多行数。获取每个序列最后活性值，送入softmax层。因每条影评长度不同，批数据每个序列RNN最后相关输出活性值有不同索引。在时间步维度(批数据形状sequences*time_steps*word_vectors)建立索引。tf.gather()沿第1维建立索引。输出活性值形状sequences*time_steps*word_vectors前两维扁平化(flatten)，添加序列长度。添加length-1,选择最后有效时间步。

梯度裁剪，梯度值限制在合理范围内。可用任何中分类有意义代价函数，模型输出可用所有类别概率分布。增加梯度裁剪(gradient clipping)改善学习结果，限制最大权值更新。RNN训练难度大，不同超参数搭配不当，权值极易发散。

TensorFlow支持优化器实例compute_gradients函数推演，修改梯度，apply_gradients函数应用权值变化。梯度分量小于-limit，设置-limit；梯度分量在于limit，设置limit。TensorFlow导数可取None，表示某个变量与代价函数没有关系，数学上应为零向量但None利于内部性能优化，只需传回None值。

影评逐个单词送入循环神经网络，每个时间步由词向量构成批数据。batched函数查找词向量，所有序列长度补齐。训练模型，定义超参数、加载数据集和词向量、经过预处理训练批数据运行模型。模型成功训练，取决网络结构、超参数、词向量质量。可从skip-gram模型word2vec项目(https://code.google.com/archive/p/word2vec/ )、斯坦福NLP研究组Glove模型(https://nlp.stanford.edu/projects/glove )，加载预训练词向量。

Kaggle 开放学习竞赛(https://kaggle.com/c/word2vec-nlp-tutorial )，IMDB影评数据，与他人比较预测结果。


    import tarfile
    import re

    from helpers import download


    class ImdbMovieReviews:

        DEFAULT_URL = \
        'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
        TOKEN_REGEX = re.compile(r'[A-Za-z]+|[!?.:,()]')

    def __init__(self, cache_dir, url=None):
        self._cache_dir = cache_dir
        self._url = url or type(self).DEFAULT_URL

        def __iter__(self):
            filepath = download(self._url, self._cache_dir)
            with tarfile.open(filepath) as archive:
                for filename in archive.getnames():
                    if filename.startswith('aclImdb/train/pos/'):
                        yield self._read(archive, filename), True
                    elif filename.startswith('aclImdb/train/neg/'):
                        yield self._read(archive, filename), False

        def _read(self, archive, filename):
            with archive.extractfile(filename) as file_:
                data = file_.read().decode('utf-8')
                data = type(self).TOKEN_REGEX.findall(data)
                data = [x.lower() for x in data]
                return data

    import bz2
    import numpy as np


    class Embedding:

        def __init__(self, vocabulary_path, embedding_path, length):
            self._embedding = np.load(embedding_path)
            with bz2.open(vocabulary_path, 'rt') as file_:
                self._vocabulary = {k.strip(): i for i, k in enumerate(file_)}
            self._length = length

        def __call__(self, sequence):
            data = np.zeros((self._length, self._embedding.shape[1]))
            indices = [self._vocabulary.get(x, 0) for x in sequence]
            embedded = self._embedding[indices]
            data[:len(sequence)] = embedded
            return data

        @property
        def dimensions(self):
            return self._embedding.shape[1]

    import tensorflow as tf

    from helpers import lazy_property


    class SequenceClassificationModel:

        def __init__(self, data, target, params):
            self.data = data
            self.target = target
            self.params = params
            self.prediction
            self.cost
            self.error
            self.optimize

        @lazy_property
        def length(self):
            used = tf.sign(tf.reduce_max(tf.abs(self.data), reduction_indices=2))
            length = tf.reduce_sum(used, reduction_indices=1)
            length = tf.cast(length, tf.int32)
            return length

        @lazy_property
        def prediction(self):
            # Recurrent network.
            output, _ = tf.nn.dynamic_rnn(
                self.params.rnn_cell(self.params.rnn_hidden),
                self.data,
                dtype=tf.float32,
                sequence_length=self.length,
            )
            last = self._last_relevant(output, self.length)
            # Softmax layer.
            num_classes = int(self.target.get_shape()[1])
            weight = tf.Variable(tf.truncated_normal(
                [self.params.rnn_hidden, num_classes], stddev=0.01))
            bias = tf.Variable(tf.constant(0.1, shape=[num_classes]))
            prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)
            return prediction

        @lazy_property
        def cost(self):
            cross_entropy = -tf.reduce_sum(self.target * tf.log(self.prediction))
            return cross_entropy

        @lazy_property
        def error(self):
            mistakes = tf.not_equal(
                tf.argmax(self.target, 1), tf.argmax(self.prediction, 1))
            return tf.reduce_mean(tf.cast(mistakes, tf.float32))

        @lazy_property
        def optimize(self):
            gradient = self.params.optimizer.compute_gradients(self.cost)
            try:
                limit = self.params.gradient_clipping
                gradient = [
                    (tf.clip_by_value(g, -limit, limit), v)
                    if g is not None else (None, v)
                    for g, v in gradient]
            except AttributeError:
                print('No gradient clipping parameter specified.')
            optimize = self.params.optimizer.apply_gradients(gradient)
            return optimize

        @staticmethod
        def _last_relevant(output, length):
            batch_size = tf.shape(output)[0]
            max_length = int(output.get_shape()[1])
            output_size = int(output.get_shape()[2])
            index = tf.range(0, batch_size) * max_length + (length - 1)
            flat = tf.reshape(output, [-1, output_size])
            relevant = tf.gather(flat, index)
            return relevant

    import tensorflow as tf

    from helpers import AttrDict

    from Embedding import Embedding
    from ImdbMovieReviews import ImdbMovieReviews
    from preprocess_batched import preprocess_batched
    from SequenceClassificationModel import SequenceClassificationModel

    IMDB_DOWNLOAD_DIR = './imdb'
    WIKI_VOCAB_DIR = '../01_wikipedia/wikipedia'
    WIKI_EMBED_DIR = '../01_wikipedia/wikipedia'


    params = AttrDict(
        rnn_cell=tf.contrib.rnn.GRUCell,
        rnn_hidden=300,
        optimizer=tf.train.RMSPropOptimizer(0.002),
        batch_size=20,
    )

    reviews = ImdbMovieReviews(IMDB_DOWNLOAD_DIR)
    length = max(len(x[0]) for x in reviews)

    embedding = Embedding(
        WIKI_VOCAB_DIR + '/vocabulary.bz2',
        WIKI_EMBED_DIR + '/embeddings.npy', length)
    batches = preprocess_batched(reviews, length, embedding, params.batch_size)

    data = tf.placeholder(tf.float32, [None, length, embedding.dimensions])
    target = tf.placeholder(tf.float32, [None, 2])
    model = SequenceClassificationModel(data, target, params)

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    for index, batch in enumerate(batches):
        feed = {data: batch[0], target: batch[1]}
        error, _ = sess.run([model.error, model.optimize], feed)
        print('{}: {:3.1f}%'.format(index + 1, 100 * error))

参考资料：
《面向机器智能的TensorFlow实践》


