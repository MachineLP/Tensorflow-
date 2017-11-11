TensorFlow并行，模型并行，数据并行。模型并行根据不同模型设计不同并行方式，模型不同计算节点放在不同硬伯上资源运算。数据并行，比较通用简便实现大规模并行方式，同时使用多个硬件资源计算不同batch数据梯度，汇总梯度全局参数更新。

数据并行，多块GPU同时训练多个batch数据，运行在每块GPU模型基于同一神经网络，网络结构一样，共享模型参数。

同步数据并行，所有GPU计算完batch数据梯度，统计将多个梯度合在一起，更新共享模型参数，类似使用较大batch。GPU型号、速度一致时，效率最高。
异步数据并行，不等待所有GPU完成一次训练，哪个GPU完成训练，立即将梯度更新到共享模型参数。
同步数据并行，比异步收敛速度更快，模型精度更高。

同步数据并行，数据集CIFAR-10。载入依赖库，TensorFlow Models cifar10类，下载CIFAR-10数据预处理。

设置batch大小 128,最大步数100万步(中间随时停止，模型定期保存)，GPU数量4。

定义计算损失函数tower_loss。cifar10.distorted_inputs产生数据增强images、labels，调用cifar10.inference生成卷积网络，每个GPU生成单独网络，结构一致，共享模型参数。根据卷积网络、labels，调用cifar10.loss计算损失函数(loss储存到collection)，tf.get_collection('losses',scope)获取当前GPU loss(scope限定范围)，tf.add_n 所有损失叠加一起得total_loss。返回total_loss作函数结果。

定义函数average_gradients，不同GPU计算梯度合成。输入参数tower_grads梯度双层列表，外层列表不同GPU计算梯度，内层列表GPU计算不同Variable梯度。最内层元素(grads,variable)，tower_grads基本元素二元组(梯度、变量)，具体形式[[(grad0_gpu0,var0_gpu0),(grad1_gpu0,var1_gpu0)……],[(grad0_gpu1,var0_gpu1),(grad1_gpu1,var1_gpu1)……]……]。创建平均梯度列表average_grads，梯度在不同GPU平均。zip(*tower_grads)双层列表转置，变[[(grad0_gpu0,var0_gpu0),(grad0_gpu1,var0_gpu1)……],[(grad1_gpu0,var1_gpu0),(grad1_gpu1,var1_gpu1)……]……]形式，循环遍历元素。循环获取元素grad_and_vars，同Variable梯度在不同GPU计算结果。同Variable梯度不同GPU计算副本，计算梯度均值。梯度N维向量，每个维度平均。tf.expand_dims给梯度添加冗余维度0,梯度放列表grad。tf.concat 维度0上合并。tf.reduce_mean维度0平均，其他维度全部平均。平均梯度，和Variable组合得原有二元组(梯度、变量)格式，添加到列表average_grads。所有梯度求均后，返回average_grads。

定义训练函数。设置默认计算设备CPU。global_step记录全局训练步数，计算epoch对应batch数，学习速率衰减需要步数decay_steps。tf.train.exponential_decay创建随训练步数衰减学习速率，第一参数初始学习速率，第二参数全局训练步数，第三参数每次衰减需要步数，第四参数衰减率，staircase设true，阶梯式衰减。设置优化算法GradientDescent，传入随机步数衰减学习速率。

定义储存GPU计算结果列表tower_grads。创建循环，循环次数GPU数量。循环中tf.device限定使用哪个GPU。tf.name_scope命名空间。

GPU用tower_loss获取损失。tf.get_variable_scope().reuse_variables()重用参数。GPU共用一个模型入完全相同参数。opt.compute_gradients(loss)计算单个GPU梯度，添加到梯度列表tower_grads。average_gradients计算平均梯度，opt.apply_gradients更新模型参数。

创建模型保存器saver，Session allow_soft_placement 参数设True。有些操作只能在CPU上进行，不使用soft_placement。初始化全部参数，tf.train.start_queue_runner()准备大量数据增强训练样本，防止训练被阻塞在生成样本。

训练循环，最大迭代次数max_steps。每步执行一次更新梯度操作apply_gradient_op(一次训练操作)，计算损失操作loss。time.time()记录耗时。每隔10步，展示当前batch loss。每秒钟可训练样本数和每个batch训练花费时间。每隔1000步，Saver保存整个模型文件。

cifar10.maybe_download_and_extract()下载完整CIFAR-10数据，train()开始训练。

loss从最开始4点几，到第70万步，降到0.07。平均每个batch耗时0.021s，平均每秒训练6000个样本，单GPU 4倍。


    import os.path
    import re
    import time
    import numpy as np
    import tensorflow as tf
    import cifar10
    batch_size=128
    #train_dir='/tmp/cifar10_train'
    max_steps=1000000
    num_gpus=4
    #log_device_placement=False
    def tower_loss(scope):
      """Calculate the total loss on a single tower running the CIFAR model.
      Args:
        scope: unique prefix string identifying the CIFAR tower, e.g. 'tower_0'
      Returns:
         Tensor of shape [] containing the total loss for a batch of data
      """
      # Get images and labels for CIFAR-10.
      images, labels = cifar10.distorted_inputs()
      # Build inference Graph.
      logits = cifar10.inference(images)
      # Build the portion of the Graph calculating the losses. Note that we will
      # assemble the total_loss using a custom function below.
      _ = cifar10.loss(logits, labels)
      # Assemble all of the losses for the current tower only.
      losses = tf.get_collection('losses', scope)
      # Calculate the total loss for the current tower.
      total_loss = tf.add_n(losses, name='total_loss')
      # Compute the moving average of all individual losses and the total loss.
      # loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
      # loss_averages_op = loss_averages.apply(losses + [total_loss])
      # Attach a scalar summary to all individual losses and the total loss; do the
      # same for the averaged version of the losses.
      # for l in losses + [total_loss]:
        # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
        # session. This helps the clarity of presentation on tensorboard.
        # loss_name = re.sub('%s_[0-9]*/' % cifar10.TOWER_NAME, '', l.op.name)
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        # tf.scalar_summary(loss_name +' (raw)', l)
        # tf.scalar_summary(loss_name, loss_averages.average(l))
        # with tf.control_dependencies([loss_averages_op]):
        # total_loss = tf.identity(total_loss)
      return total_loss
    def average_gradients(tower_grads):
      """Calculate the average gradient for each shared variable across all towers.
      Note that this function provides a synchronization point across all towers.
      Args:
        tower_grads: List of lists of (gradient, variable) tuples. The outer list
          is over individual gradients. The inner list is over the gradient
          calculation for each tower.
      Returns:
         List of pairs of (gradient, variable) where the gradient has been averaged
         across all towers.
      """
      average_grads = []
      for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
          # Add 0 dimension to the gradients to represent the tower.
          expanded_g = tf.expand_dims(g, 0)
          # Append on a 'tower' dimension which we will average over below.
          grads.append(expanded_g)
        # Average over the 'tower' dimension.
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)
        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
      return average_grads
    def train():
      """Train CIFAR-10 for a number of steps."""
      with tf.Graph().as_default(), tf.device('/cpu:0'):
        # Create a variable to count the number of train() calls. This equals the
        # number of batches processed * FLAGS.num_gpus.
        global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0), trainable=False)
        # Calculate the learning rate schedule.
        num_batches_per_epoch = (cifar10.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN /
                             batch_size)
        decay_steps = int(num_batches_per_epoch * cifar10.NUM_EPOCHS_PER_DECAY)
        # Decay the learning rate exponentially based on the number of steps.
        lr = tf.train.exponential_decay(cifar10.INITIAL_LEARNING_RATE,
                                        global_step,
                                        decay_steps,
                                        cifar10.LEARNING_RATE_DECAY_FACTOR,
                                        staircase=True)
        # Create an optimizer that performs gradient descent.
        opt = tf.train.GradientDescentOptimizer(lr)
        # Calculate the gradients for each model tower.
        tower_grads = []
        for i in range(num_gpus):
          with tf.device('/gpu:%d' % i):
            with tf.name_scope('%s_%d' % (cifar10.TOWER_NAME, i)) as scope:
              # Calculate the loss for one tower of the CIFAR model. This function
              # constructs the entire CIFAR model but shares the variables across
              # all towers.
              loss = tower_loss(scope)
              # Reuse variables for the next tower.
              tf.get_variable_scope().reuse_variables()
              # Retain the summaries from the final tower.
              # summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
              # Calculate the gradients for the batch of data on this CIFAR tower.
              grads = opt.compute_gradients(loss)
              # Keep track of the gradients across all towers.
              tower_grads.append(grads)
        # We must calculate the mean of each gradient. Note that this is the
        # synchronization point across all towers.
        grads = average_gradients(tower_grads)
        # Add a summary to track the learning rate.
        # summaries.append(tf.scalar_summary('learning_rate', lr))
        # Add histograms for gradients.
        # for grad, var in grads:
        #     if grad is not None:
        #         summaries.append(
        #             tf.histogram_summary(var.op.name + '/gradients', grad))
        # Apply the gradients to adjust the shared variables.
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
        # Add histograms for trainable variables.
        # for var in tf.trainable_variables():
        #     summaries.append(tf.histogram_summary(var.op.name, var))
        # Track the moving averages of all trainable variables.
        # variable_averages = tf.train.ExponentialMovingAverage(
        #     cifar10.MOVING_AVERAGE_DECAY, global_step)
        # variables_averages_op = variable_averages.apply(tf.trainable_variables())
        # Group all updates to into a single train op.
        # train_op = tf.group(apply_gradient_op, variables_averages_op)
        # Create a saver.
        saver = tf.train.Saver(tf.all_variables())
        # Build the summary operation from the last tower summaries.
        # summary_op = tf.merge_summary(summaries)
        # Build an initialization operation to run below.
        init = tf.global_variables_initializer()
        # Start running operations on the Graph. allow_soft_placement must be set to
        # True to build towers on GPU, as some of the ops do not have GPU
        # implementations.
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        sess.run(init)
        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)
        # summary_writer = tf.train.SummaryWriter(train_dir, sess.graph)
        for step in range(max_steps):
          start_time = time.time()
          _, loss_value = sess.run([apply_gradient_op, loss])
          duration = time.time() - start_time
          assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
          if step % 10 == 0:
            num_examples_per_step = batch_size * num_gpus
            examples_per_sec = num_examples_per_step / duration
            sec_per_batch = duration / num_gpus
            format_str = ('step %d, loss = %.2f (%.1f examples/sec; %.3f '
                          'sec/batch)')
            print (format_str % (step, loss_value,
                                 examples_per_sec, sec_per_batch))
            # if step % 100 == 0:
            #     summary_str = sess.run(summary_op)
            #     summary_writer.add_summary(summary_str, step)
          # Save the model checkpoint periodically.
          if step % 1000 == 0 or (step + 1) == max_steps:
          # checkpoint_path = os.path.join(train_dir, 'model.ckpt')
            saver.save(sess, '/tmp/cifar10_train/model.ckpt', global_step=step)
    cifar10.maybe_download_and_extract()
    #if tf.gfile.Exists(train_dir):
    #  tf.gfile.DeleteRecursively(train_dir)
    #tf.gfile.MakeDirs(train_dir)
    train()

参考资料：
《TensorFlow实战》


