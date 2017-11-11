卷积神经网络发展趋势。Perceptron(感知机)，1957年，Frank Resenblatt提出，始祖。Neocognitron(神经认知机)，多层级神经网络，日本科学家Kunihiko fukushima，20世纪80年代提出，一定程度视觉认知功能，启发卷积神经网络。LeNet-5，CNN之父，Yann LeCun，1997年提出，首次多层级联卷积结构，手写数字有效识别。2012年，Hinton学生Alex，8层卷积神经网络，ILSVRC 2012比赛冠军。AlexNet 成功应用ReLU激活函数、Dropout、最大覆盖池化、LRN层、GPU加速，启发后续技术创新，卷积神经网络研究进入快车道。

AlexNetx后，卷积神经网络，一类网络结构改进调整，一类网络深度增加。

               Perceptron(1957)
              Neocognitron(198x)
     NIN(2013)                  VGG(2014)
     Incepiton V1(2014)    MSRANet(2014)
     Incepiton V2(2015)    ResNet(2015)
     Incepiton V3(2015)    ResNet V2(2015)
            Inception ResNet V2(2016)

2013年，颜水成教授，Network in Network首次发表，优化卷积神经网络结构，推广1x1卷积结构。2014年，Google Incepiton Net V1,Inception Module，反复堆叠高效卷积网络结构，ILSVRC 2014冠军。2015年初，Incepiton V2,Batch Normalization，加速训练过程，提升网络性能。2015年末，Inception V3,Factorization in Small Convolutions思想，分解大尺寸卷积为多个小卷积或一维卷积。

2014年，ILSVRC亚军，VGGNet，全程3x3卷积，19层网络。季军MSRA-Net(微软)也是深层网络。2015年，微软ResNet，152层网络，ILSVRC 2015冠军，top-5错误率3.46%。ResNet V2,Batch Normalization，去除激活层，用Identity Mapping或Preactivation，提升网络性能。Inception ResNet V2,融合Inception Net网络结构,和ResNet训练极深网络残差学习模块。

GPU计算资源，开源工具。

循环神经网络(RNN)，NLP(Nature Language Processing，自然语言处理)最常用神经网络结构。Word2Vec，语言字词转化稠密向量(Dense Vector)。

Word2Vec，Word Embeddings，词向量或词嵌入。语言字词转向量形式表达(Vector Representations)模型。图片，像素点稠密矩阵，音频，声音信号频谱数据。

One-Hot Encoder，字词转离散单独符号。一个词对应一个向量，整篇文章对应一个稀疏矩阵。文本分类模型，Bag of Words，稀疏矩阵合并为一个向量，每个词对应向量计数，统计词出现次数，作为特征。特征编码随机，没有关联信息，没有字词关系。稀疏向量需要更多数据训练，训练效率低，计算麻烦。

向量表达(Vector Representations)，向量空间模型(Vector Space Models)，字词转连续值向量表达，意思相近词映射向量空量空间相近位置。向量空间模型在NLP依赖假设Distributional Hypothesis，相同语境词语义相近。向量空间模型，分两类，计数模型(Latent Semantic Analysis)，预测模型(Neural Probabilistic Language Models)。计数模型统计语料库相邻词频率，计数统计结果转小稠密矩阵，预测模型根据词周围相邻词推测出这个词和空间向量。

Word2Vec，计算非常高效，从原始语料学习字词空间向量预测模型。CBOW(Continuous Bag of Words)模式从原始语句推测目标字词，适合小型数据。Skip-Gram从目标字词推测原始语句，适合大型语料。意思相近词向量空间位置接近。

预测模型(Neural Probabilistic Language Models)，用最大似然方法，给定前语句h，最大化目标词汇Wt概率。计算量大，需计算词汇表所有单词出现可能性。Word2Vec CBOw模型，只需训练二元分类模型，区分真实目标词汇、编造词汇(噪声)两类。少量噪声词汇估计，类似蒙特卡洛模拟。

模型预测真实目标词汇高概率，预测其他噪声词汇低概率，训练学习目标最优化。编造噪声词汇训练，Negative Sampling，计算loss fuction效率非常高，只需计算随机选择k个词汇，训练速度快。Noise_contrastive Estimation(NCE) Loss，TensorFlow tf.nn.nce_loss。

Word2Vec Skip-Gram模式。构造语境与目标词汇映射关系。语境包括单词左边和右边词汇。滑窗尺寸 1。Skip-Gram模型，从目标词汇预测语境。制造随机词汇作负样本(噪声)。预测概率分布，正样本尽可能大，随机产生负样本尺可能小。优化算法(SGD)更新模型Word Embedding参数，概率分布损失函数(NCE Loss)尽可能小。单词Embedded Vector随训练过程调整，直到最适合语料空间位置。损失函数最小，最符合语料，预测正确单词概率最高。

载入依赖库。

定义下载广西数据函数，urllib.request.urlretrieve下载数据压缩文件核文件尺寸。已下载跳过。

解压下载压缩文件，tf.compat.as_str 数据转单词列表。数据转为17005207单词列表。

创建vocabulary词汇表，collections.Counter统计单词列表单词频数，most_common方法取top 50000频数单词作vocabulary。创建dict，top 50000词汇vocabulary放入dictionary，快速查询。Python dict查询复杂度O(1)，性能好。全部单词转编号(频数排序编号)。top50000以外单词，认定为Unkown(未知)，编号0,统计数量。遍历单词列表，每个单词，判断是否出现在dictionary，是转编号，不是编0。返回转换编码(data)、单词频数统计count、词汇表(dictionary)、反转形式(reverse_dictionary)。

删除原始单词列表，节约内存。打印vocabulary最高频词汇、数量(包括Unknow词汇)。“UNK”类418391个。“the”1061396个。“of”593677个。data前10单词['anarchism','originated','as','a','term','of','abuse','first','used','against'],编号[5235,3084,12,6,195,2,3137,46,59,156]。

生成Word2Vec训练样本。Skip-Gram模式(从目标单词反推语境)。定义函数generate_batch生成训练batch数据。参数batch_size为batch大小。skip_window单词最远可联系距离，设1只能跟紧邻两个单词生成样本。num_skips单词生成样本个数，不能大于skip_window两倍，batch_size是它的整数倍，确保batch包含词汇所有样本。

单词序号data_index为global变量，反复调用generate_batch，确保data_index可以在函数genetate_batch修改。assert确保num_skips、batch_size满足条件。np.ndarray初始化batch、labels为数组。定义span 单词创建相关样本单词数量，包括目标单词和前后单词，span=2*skip_window+1。创建最大容量span deque，双向队列，deque append方法添加变量，只保留最后插入span个变量。

从序号data_index开始，span个单词顺序读入buffer作初始值。buffer容量为span deque，已填满，后续数据替换前面数据。

第一层循环(次数batch_size//num_skips)，循环内目标单词生成样本。buffer目标单词和所有相关单词，定义target-skip_window，buffer第skip_window个变量为目标单词。定义生成样本需避免单词列表，tagets_to_avoid，列表开始包括第skip_window个单词(目标单词)，预测语境单词，不包括目标单词。

第二层循环(次数num_skips)，循环语境单词生成样本，先产生随机数，直到随机数不在targets_to_avoid中，代表可用语境单词，生成样本，feature目标词汇buffer[skip_window]，label是buffer[target]。语境单词使用，添加到targets_to_avoid过滤。目标单词所有样本生成完(num_skips个)，读入下一个单词，抛掉buffer第一个单词，滑窗向后移动一位，目标单词向后移动一个，语境单词整体后移，开始生成下一个目标单词训练样本。

两层循环完成，获得batch_size个训练样本。返回batch、labels。

调用generate_batch函数测试。参数batch_size设8,num_skips设2,skip_window设1,执行generate_batch获得batch、labels，打印。

定义训练batch_size 128,embedding_size 128。embedding_size，单词转稠密向量维度，50〜1000。skip_window单词间最远联系距离设1,num_skips目标单词提取样本数设2.生成验证数据valid_examples。随机抽取频数最高单词，看向量空间最近单词是否相关性高。valid_size设16抽取验证单词数。valid_window设100验证单词频为最高100个单词抽取。np.random.choice函数随机抽取。num_sampled训练负样本噪声单词数量。

定义Skip_Gram Word2Vec模型网络结构。创建f.Graph，设置为默认graph。创建训练数据inputs、labels placeholder，随机产生valid_examples转TensorFlow constant。with tf.device('/cpu:0')限定所有计算在CPU执行。tf.random_uniform随机生成所有单词词向量embeddings，单词表大小50000,向量维度128，tf.nn.embedding_lookup查找输入train_inputs对应赂理embed。tf.truncated_normal初始化训练优化目标NCE Loss的权重参数nce_weights，nce_biases初始化0。tf.nn.nce_loss计算学习词向量embedding训练数据loss，tf.reduce_mean汇总。

定义优化器SGD ,学习速率1.0。计算嵌入向量embeddings L2范数norm，embeddings除L2范数得标准化normalized_embeddings。tf.nn.embedding_lookup查询验证单词嵌入向量，计算验证单词嵌入同与词汇表所有单词相似性。tf.global_variables_initializer初始化所有模型参数。

定义最大迭代次数10万次，创建设置默认session，执行参数初始化。迭代中，generate_batch生成batch inputs、labels数据，创建feed_dict。session.run()执行优化器运算(参数更新)和损失计算，训练loss累积到avegage_loss。

每2000次循环，计算平均loss，显示。

每10000次循环，计算验证单词和全部单词相似度，验证单词最相似8个单词展示。

训练模型对名词、动词、形容词类型单词相似词汇识别非常准确。Skip-Gram Word2Vec 向量空间表达(Vetor Representations)质量非常高，近义词在向量空间位置非常靠近。

定义可视化Word2Vec效果函数。low_dim_embs降给到2维单词空间向量，图表展示单词位置。plt.scatter(matplotlib.pyplot)显示散点图(单词位置)，plt.annotate展示单词本身。plt.savefig保存图片到本地文件。

sklearn.manifold.TSNe实现降维，原始128维嵌入同量降到2维，plot_sith_labels函数展示。只展示词频最高100个单词可视化结果。

距离相近单词，语义高相似性。左上角单个字母聚集地。冠词聚集在左边中部。Word2Vec性能评价，可视化观察，Analogical Reasoning直接预测语义、语境关系。回答填空问题。大规模语料库，参数调试选取最适合值。


     import collections
     import math
     import os
     import random
     import zipfile
     import numpy as np
     import urllib
     import tensorflow as tf
     # Step 1: Download the data.
     url = 'http://mattmahoney.net/dc/'
     def maybe_download(filename, expected_bytes):
       if not os.path.exists(filename):
         filename, _ = urllib.request.urlretrieve(url + filename, filename)
       statinfo = os.stat(filename)
       if statinfo.st_size == expected_bytes:
         print('Found and verified', filename)
       else:
         print(statinfo.st_size)
         raise Exception(
        'Failed to verify ' + filename + '. Can you get to it with a browser?')
       return filename
     filename = maybe_download('text8.zip', 31344016)
     # Read the data into a list of strings.
     def read_data(filename):
       with zipfile.ZipFile(filename) as f:
         data = tf.compat.as_str(f.read(f.namelist()[0])).split()
       return data
     words = read_data(filename)
     print('Data size', len(words))
     # Step 2: Build the dictionary and replace rare words with UNK token.
     vocabulary_size = 50000
     def build_dataset(words):
       count = [['UNK', -1]]
       count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
       dictionary = dict()
       for word, _ in count:
         dictionary[word] = len(dictionary)
       data = list()
       unk_count = 0
       for word in words:
         if word in dictionary:
           index = dictionary[word]
         else:
           index = 0  # dictionary['UNK']
           unk_count += 1
         data.append(index)
       count[0][1] = unk_count
       reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
       return data, count, dictionary, reverse_dictionary
     data, count, dictionary, reverse_dictionary = build_dataset(words)
     del words  # Hint to reduce memory.
     print('Most common words (+UNK)', count[:5])
     print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])
     data_index = 0
     # Step 3: Function to generate a training batch for the skip-gram model.
     def generate_batch(batch_size, num_skips, skip_window):
       global data_index
       assert batch_size % num_skips == 0
       assert num_skips <= 2 * skip_window
       batch = np.ndarray(shape=(batch_size), dtype=np.int32)
       labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
       span = 2 * skip_window + 1 # [ skip_window target skip_window ]
       buffer = collections.deque(maxlen=span)
       for _ in range(span):
         buffer.append(data[data_index])
         data_index = (data_index + 1) % len(data)
       for i in range(batch_size // num_skips):
         target = skip_window  # target label at the center of the buffer
         targets_to_avoid = [ skip_window ]
         for j in range(num_skips):
           while target in targets_to_avoid:
             target = random.randint(0, span - 1)
      targets_to_avoid.append(target)
      batch[i * num_skips + j] = buffer[skip_window]
      labels[i * num_skips + j, 0] = buffer[target]
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
       return batch, labels
     batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
     for i in range(8):
       print(batch[i], reverse_dictionary[batch[i]],
           '->', labels[i, 0], reverse_dictionary[labels[i, 0]])
     # Step 4: Build and train a skip-gram model.
     batch_size = 128
     embedding_size = 128  # Dimension of the embedding vector.
     skip_window = 1       # How many words to consider left and right.
     num_skips = 2         # How many times to reuse an input to generate a label.
     valid_size = 16     # Random set of words to evaluate similarity on.
     valid_window = 100  # Only pick dev samples in the head of the distribution.
     valid_examples = np.random.choice(valid_window, valid_size, replace=False)
     num_sampled = 64    # Number of negative examples to sample.
     graph = tf.Graph()
     with graph.as_default():
       # Input data.
       train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
       train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
       valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
       # Ops and variables pinned to the CPU because of missing GPU implementation
       with tf.device('/cpu:0'):
    # Look up embeddings for inputs.
         embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
         embed = tf.nn.embedding_lookup(embeddings, train_inputs)
         # Construct the variables for the NCE loss
         nce_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)))
         nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
       loss = tf.reduce_mean(
           tf.nn.nce_loss(weights=nce_weights,
                          biases=nce_biases,
                          labels=train_labels,
                          inputs=embed,
                          num_sampled=num_sampled,
                          num_classes=vocabulary_size))
       # Construct the SGD optimizer using a learning rate of 1.0.
       optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
       # Compute the cosine similarity between minibatch examples and all embeddings.
       norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
       normalized_embeddings = embeddings / norm
       valid_embeddings = tf.nn.embedding_lookup(
      normalized_embeddings, valid_dataset)
       similarity = tf.matmul(
      valid_embeddings, normalized_embeddings, transpose_b=True)
       # Add variable initializer.
       init = tf.global_variables_initializer()
     # Step 5: Begin training.
     num_steps = 100001
     with tf.Session(graph=graph) as session:
       init.run()
       print("Initialized")
       average_loss = 0
       for step in range(num_steps):
         batch_inputs, batch_labels = generate_batch(
             batch_size, num_skips, skip_window)
         feed_dict = {train_inputs : batch_inputs, train_labels : batch_labels}
         _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
         average_loss += loss_val
         if step % 2000 == 0:
           if step > 0:
             average_loss /= 2000
           # The average loss is an estimate of the loss over the last 2000 batches.
           print("Average loss at step ", step, ": ", average_loss)
           average_loss = 0
         # Note that this is expensive (~20% slowdown if computed every 500 steps)
         if step % 10000 == 0:
           sim = similarity.eval()
           for i in range(valid_size):
             valid_word = reverse_dictionary[valid_examples[i]]
             top_k = 8 # number of nearest neighbors
             nearest = (-sim[i, :]).argsort()[1:top_k+1]
             log_str = "Nearest to %s:" % valid_word
             for k in range(top_k):
               close_word = reverse_dictionary[nearest[k]]
               log_str = "%s %s," % (log_str, close_word)
             print(log_str)
       final_embeddings = normalized_embeddings.eval()
     # Step 6: Visualize the embeddings.
     def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
       assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
       plt.figure(figsize=(18, 18))  #in inches
       for i, label in enumerate(labels):
         x, y = low_dim_embs[i,:]
         plt.scatter(x, y)
         plt.annotate(label,
                      xy=(x, y),
                      xytext=(5, 2),
                      textcoords='offset points',
                      ha='right',
                      va='bottom')
       plt.savefig(filename)
       #%%
     try:
       from sklearn.manifold import TSNE
       import matplotlib.pyplot as plt
       tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
       plot_only = 200
       low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only,:])
       labels = [reverse_dictionary[i] for i in range(plot_only)]
       plot_with_labels(low_dim_embs, labels)
     except ImportError:
       print("Please install sklearn, matplotlib, and scipy to visualize embeddings.")

参考资料：
《TensorFlow实战》


