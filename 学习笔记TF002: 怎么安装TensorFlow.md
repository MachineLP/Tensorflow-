官网：
https://www.tensorflow.org/install/install_mac
https://www.tensorflow.org/install/install_windows

考虑到软件依赖项，依赖冲突。单机建议用Virturalenv，分布式用Docker。解决依赖冲突有以下方式：代码库内部的软件包依赖，依赖库放到代码中，局部引用。重复占用空间，手工更改。用户无法修改。使用依赖环境，虚拟环境。Virturalenv、Anaconda。使用容器，软件、文件系统、运行时、依赖库打包轻量级方案。典型应用有Docker。

TensorFlow需要用到两个经典库：Jupyter(iPython) Notebook、matplotlib。Jupyter Notebook可以交互式编写可视化结果文档，代码展示，Markdown单元，设计原型，代码写入逻辑块，方便调试脚本特定部分。matplotlib是绘图库，可以实现数据可视化，典型应用Seaborn。

Virtualenv环境安装（看网络情况，我装了四小时，重试了无数次这两个命令，尤其是第二个）
sudo easy_install pip
sudo pip install --upgrade virtualenv

创建虚拟环境目录：
sudo mkdir ~/env

创建虚拟环境：
virtualenv --system-site-packages ~/env/tensorflow

激活虚拟环境：
source ~/env/tensorflow/bin/activate

关闭虚拟环境：
deactivate

安装TensorFlow(装了2小时，还是不行):
Python 2.7: pip install --upgrade tensorflow
Python 3.4: pip3 install --upgrade tensorflow

最后是把需要的whl下载下来，直接通过pip install 装本地的文件。

安装Jupyter、matplotlib(又是两小时。。。)
sudo pip install jupyter
sudo pip install matplotlib

其中widgetsnbextension没有办法下载下来装，因为下下来的是3.0.0的，需要的是2.0.0的。

装完之后，在Jupyter上跑一个。

mkdir tf-notebooks

cd tf-notebooks
jupyter notebook

测试一下：

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

%matplotlib inline
a = tf.random_normal([2,20])
sess = tf.Session()
out = sess.run(a)
x,y = out

plt.scatter(x,y)
plt.show()

参考资料：
《TensorFlow实战》
《面向机器智能的TensorFlow实践》


