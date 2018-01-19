我们在使用tensorflow时，会发现tf.nn，tf.layers， tf.contrib模块有很多功能是重复的，尤其是卷积操作，在使用的时候，我们可以根据需要现在不同的模块。但有些时候可以一起混用。
        下面是对三个模块的简述：
        （1）tf.nn ：提供神经网络相关操作的支持，包括卷积操作（conv）、池化操作（pooling）、归一化、loss、分类操作、embedding、RNN、Evaluation。
        （2）tf.layers：主要提供的高层的神经网络，主要和卷积相关的，个人感觉是对tf.nn的进一步封装，tf.nn会更底层一些。
        （3）tf.contrib：tf.contrib.layers提供够将计算图中的  网络层、正则化、摘要操作、是构建计算图的高级操作，但是tf.contrib包含不稳定和实验代码，有可能以后API会改变。

以上三个模块的封装程度是逐个递进的。 
