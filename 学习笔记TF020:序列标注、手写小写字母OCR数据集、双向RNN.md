序列标注(sequence labelling)，输入序列每一帧预测一个类别。OCR(Optical Character Recognition 光学字符识别)。

MIT口语系统研究组Rob Kassel收集，斯坦福大学人工智能实验室Ben Taskar预处理OCR数据集(http://ai.stanford.edu/~btaskar/ocr/ )，包含大量单独手写小写字母，每个样本对应16X8像素二值图像。字线组合序列，序列对应单词。6800个，长度不超过14字母的单词。gzip压缩，内容用Tab分隔文本文件。Python csv模块直接读取。文件每行一个归一化字母属性，ID号、标签、像素值、下一字母ID号等。

下一字母ID值排序，按照正确顺序读取每个单词字母。收集字母，直到下一个ID对应字段未被设置为止。读取新序列。读取完目标字母及数据像素，用零图像填充序列对象，能纳入两个较大目标字母所有像素数据NumPy数组。

时间步之间共享softmax层。数据和目标数组包含序列，每个目标字母对应一个图像帧。RNN扩展，每个字母输出添加softmax分类器。分类器对每帧数据而非整个序列评估预测结果。计算序列长度。一个softmax层添加到所有帧：或者为所有帧添加几个不同分类器，或者令所有帧共享同一个分类器。共享分类器，权值在训练中被调整次数更多，训练单词每个字母。一个全连接层权值矩阵维数batch_size*in_size*out_size。现需要在两个输入维度batch_size、sequence_steps更新权值矩阵。令输入(RNN输出活性值)扁平为形状batch_size*sequence_steps*in_size。权值矩阵变成较大的批数据。结果反扁平化(unflatten)。

代价函数，序列每一帧有预测目标对，在相应维度平均。依据张量长度(序列最大长度)归一化的tf.reduce_mean无法使用。需要按照实际序列长度归一化，手工调用tf.reduce_sum和除法运算均值。

损失函数，tf.argmax针对轴2非轴1,各帧填充，依据序列实际长度计算均值。tf.reduce_mean对批数据所有单词取均值。

TensorFlow自动导数计算，可使用序列分类相同优化运算，只需要代入新代价函数。对所有RNN梯度裁剪，防止训练发散，避免负面影响。

训练模型，get_sataset下载手写体图像，预处理，小写字母独热编码向量。随机打乱数据顺序，分偏划分训练集、测试集。

单词相邻字母存在依赖关系(或互信息)，RNN保存同一单词全部输入信息到隐含活性值。前几个字母分类，网络无大量输入推断额外信息，双向RNN(bidirectional RNN)克服缺陷。
两个RNN观测输入序列，一个按照通常顺序从左端读取单词，另一个按照相反顺序从右端读取单词。每个时间步得到两个输出活性值。送入共享softmax层前，拼接。分类器从每个字母获取完整单词信息。tf.modle.rnn.bidirectional_rnn已实现。

实现双向RNN。划分预测属性到两个函数，只关注较少内容。_shared_softmax函数，传入函数张量data推断输入尺寸。复用其他架构函数，相同扁平化技巧在所有时间步共享同一个softmax层。rnn.dynamic_rnn创建两个RNN。
序列反转，比实现新反向传递RNN运算容易。tf.reverse_sequence函数反转帧数据中sequence_lengths帧。数据流图节点有名称。scope参数是rnn_dynamic_cell变量scope名称，默认值RNN。两个参数不同RNN，需要不同域。
反转序列送入后向RNN，网络输出反转，和前向输出对齐。沿RNN神经元输出维度拼接两个张量，返回。双向RNN模型性能更优。

    import gzip
    import csv
    import numpy as np

    from helpers import download

    class OcrDataset:

        URL = 'http://ai.stanford.edu/~btaskar/ocr/letter.data.gz'

        def __init__(self, cache_dir):
            path = download(type(self).URL, cache_dir)
            lines = self._read(path)
            data, target = self._parse(lines)
            self.data, self.target = self._pad(data, target)

        @staticmethod
        def _read(filepath):
            with gzip.open(filepath, 'rt') as file_:
                reader = csv.reader(file_, delimiter='\t')
                lines = list(reader)
                return lines

        @staticmethod
        def _parse(lines):
            lines = sorted(lines, key=lambda x: int(x[0]))
            data, target = [], []
            next_ = None
            for line in lines:
                if not next_:
                    data.append([])
                    target.append([])
                else:
                    assert next_ == int(line[0])
                next_ = int(line[2]) if int(line[2]) > -1 else None
                pixels = np.array([int(x) for x in line[6:134]])
                pixels = pixels.reshape((16, 8))
                data[-1].append(pixels)
                target[-1].append(line[1])
            return data, target

        @staticmethod
        def _pad(data, target):
            max_length = max(len(x) for x in target)
            padding = np.zeros((16, 8))
            data = [x + ([padding] * (max_length - len(x))) for x in data]
            target = [x + ([''] * (max_length - len(x))) for x in target]
            return np.array(data), np.array(target)

    import tensorflow as tf

    from helpers import lazy_property

    class SequenceLabellingModel:

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
            output, _ = tf.nn.dynamic_rnn(
                tf.nn.rnn_cell.GRUCell(self.params.rnn_hidden),
                self.data,
                dtype=tf.float32,
                sequence_length=self.length,
            )
            # Softmax layer.
            max_length = int(self.target.get_shape()[1])
            num_classes = int(self.target.get_shape()[2])
            weight = tf.Variable(tf.truncated_normal(
                [self.params.rnn_hidden, num_classes], stddev=0.01))
            bias = tf.Variable(tf.constant(0.1, shape=[num_classes]))
            # Flatten to apply same weights to all time steps.
            output = tf.reshape(output, [-1, self.params.rnn_hidden])
            prediction = tf.nn.softmax(tf.matmul(output, weight) + bias)
            prediction = tf.reshape(prediction, [-1, max_length, num_classes])
            return prediction

        @lazy_property
        def cost(self):
            # Compute cross entropy for each frame.
            cross_entropy = self.target * tf.log(self.prediction)
            cross_entropy = -tf.reduce_sum(cross_entropy, reduction_indices=2)
            mask = tf.sign(tf.reduce_max(tf.abs(self.target), reduction_indices=2))
            cross_entropy *= mask
            # Average over actual sequence lengths.
            cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices=1)
            cross_entropy /= tf.cast(self.length, tf.float32)
            return tf.reduce_mean(cross_entropy)

        @lazy_property
        def error(self):
            mistakes = tf.not_equal(
                tf.argmax(self.target, 2), tf.argmax(self.prediction, 2))
            mistakes = tf.cast(mistakes, tf.float32)
            mask = tf.sign(tf.reduce_max(tf.abs(self.target), reduction_indices=2))
            mistakes *= mask
            # Average over actual sequence lengths.
            mistakes = tf.reduce_sum(mistakes, reduction_indices=1)
            mistakes /= tf.cast(self.length, tf.float32)
            return tf.reduce_mean(mistakes)

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

    import random

    import tensorflow as tf
    import numpy as np

    from helpers import AttrDict

    from OcrDataset import OcrDataset
    from SequenceLabellingModel import SequenceLabellingModel
    from batched import batched

    params = AttrDict(
        rnn_cell=tf.nn.rnn_cell.GRUCell,
        rnn_hidden=300,
        optimizer=tf.train.RMSPropOptimizer(0.002),
        gradient_clipping=5,
        batch_size=10,
        epochs=5,
        epoch_size=50
    )

    def get_dataset():
        dataset = OcrDataset('./ocr')
        # Flatten images into vectors.
        dataset.data = dataset.data.reshape(dataset.data.shape[:2] + (-1,))
        # One-hot encode targets.
        target = np.zeros(dataset.target.shape + (26,))
        for index, letter in np.ndenumerate(dataset.target):
            if letter:
                target[index][ord(letter) - ord('a')] = 1
        dataset.target = target
        # Shuffle order of examples.
        order = np.random.permutation(len(dataset.data))
        dataset.data = dataset.data[order]
        dataset.target = dataset.target[order]
        return dataset

    # Split into training and test data.
    dataset = get_dataset()
    split = int(0.66 * len(dataset.data))
    train_data, test_data = dataset.data[:split], dataset.data[split:]
    train_target, test_target = dataset.target[:split], dataset.target[split:]

    # Compute graph.
    _, length, image_size = train_data.shape
    num_classes = train_target.shape[2]
    data = tf.placeholder(tf.float32, [None, length, image_size])
    target = tf.placeholder(tf.float32, [None, length, num_classes])
    model = SequenceLabellingModel(data, target, params)
    batches = batched(train_data, train_target, params.batch_size)

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    for index, batch in enumerate(batches):
        batch_data = batch[0]
        batch_target = batch[1]
        epoch = batch[2]
        if epoch >= params.epochs:
            break
        feed = {data: batch_data, target: batch_target}
        error, _ = sess.run([model.error, model.optimize], feed)
        print('{}: {:3.6f}%'.format(index + 1, 100 * error))

    test_feed = {data: test_data, target: test_target}
    test_error, _ = sess.run([model.error, model.optimize], test_feed)
    print('Test error: {:3.6f}%'.format(100 * error))

    import tensorflow as tf

    from helpers import lazy_property

    class BidirectionalSequenceLabellingModel:

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
            output = self._bidirectional_rnn(self.data, self.length)
            num_classes = int(self.target.get_shape()[2])
            prediction = self._shared_softmax(output, num_classes)
            return prediction

        def _bidirectional_rnn(self, data, length):
            length_64 = tf.cast(length, tf.int64)
            forward, _ = tf.nn.dynamic_rnn(
                cell=self.params.rnn_cell(self.params.rnn_hidden),
                inputs=data,
                dtype=tf.float32,
                sequence_length=length,
                scope='rnn-forward')
            backward, _ = tf.nn.dynamic_rnn(
            cell=self.params.rnn_cell(self.params.rnn_hidden),
            inputs=tf.reverse_sequence(data, length_64, seq_dim=1),
            dtype=tf.float32,
            sequence_length=self.length,
            scope='rnn-backward')
            backward = tf.reverse_sequence(backward, length_64, seq_dim=1)
            output = tf.concat(2, [forward, backward])
            return output

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

        @lazy_property
        def cost(self):
            # Compute cross entropy for each frame.
            cross_entropy = self.target * tf.log(self.prediction)
            cross_entropy = -tf.reduce_sum(cross_entropy, reduction_indices=2)
            mask = tf.sign(tf.reduce_max(tf.abs(self.target), reduction_indices=2))
            cross_entropy *= mask
            # Average over actual sequence lengths.
            cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices=1)
            cross_entropy /= tf.cast(self.length, tf.float32)
            return tf.reduce_mean(cross_entropy)

        @lazy_property
        def error(self):
            mistakes = tf.not_equal(
                tf.argmax(self.target, 2), tf.argmax(self.prediction, 2))
            mistakes = tf.cast(mistakes, tf.float32)
            mask = tf.sign(tf.reduce_max(tf.abs(self.target), reduction_indices=2))
            mistakes *= mask
            # Average over actual sequence lengths.
            mistakes = tf.reduce_sum(mistakes, reduction_indices=1)
            mistakes /= tf.cast(self.length, tf.float32)
            return tf.reduce_mean(mistakes)

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

参考资料：
《面向机器智能的TensorFlow实践》


