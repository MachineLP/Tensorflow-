下载TensorFlow https://github.com/tensorflow/tensorflow/tree/v1.1.0 。Tags选择版本，下载解压。

pip安装。pip，Python包管理工具，PyPI(Python Packet Index) https://pypi.python.org/pypi 。

Mac环境。
安装virtualenv。virtualenv，Python沙箱工具，创建独立Python环境。pip install virtralenv --upgrade 安装virtualenv。virtualenv --system-site-packages ~/tensorflow 创建tensorflow文件夹。cd ~/tensorflow 进入目录。source bin/activate 激活沙箱。pip install tensorflow==1.1.0 安装TensorFlow。

Ubuntu/Linux环境。
先安装virtualenv沙盒环境，再用pip安装TensorFlow。CPU版 pip install tensorflow==1.1.0 。GPU版 pip install tensorflow-gpu==1.1.0 。

Windows环境。
Windows 7､Windows 10、Server 2016。PowerShell。64位Python3.5.x。

Java安装。
下载JAR(Java ARchive)libtensorflow-1.1.0-rc2.jar。本地库，libtensorflow_jni-cpu-darwin-x86_64-1.1.0-rc2.tar.gz解压到jni目录。编译 javac -cpu libtensorflow-1.1.0-rc2.jar MyClass.java 。

源代码安装。
Bazel编译工具，JDK 8，0.44。brew install bazel 。其他操作系统，apt-get。进入tensorflow-1.1.0源代码目录，运行./configure，Python路径、是否用HDFS、是否用Google Cloud Platform。bazel编译命令，加入--local_resources 2048,4,1.0限制内存大小。bazel build --local_resources 2048,4,1.0 -c opt //tensorflow/tools/pip_package:build_pip_package bazel-bin/tensorflow/tools/pip_package/build_pip_package /tem/tensorflow_pkg 。进入/tem/tensorflow_pkg，pip install /tmp/tensorflow_pkg/tensorflow-1.1.0-cp27-cp27m-macosx_10_12_intel.whl 。GPU版本需要配置选择使用CUDA、CUDA SDK版本。

依赖模块。
numpy。存储、处理大型矩阵科学计算包。比Python嵌套列表结构(nested list structure)高效。强大N维数组对象Array。成熟函数库。整合C/C++、Fortran代码工具包。线性代数、傅里叶变换、随机数生成函数。pip install numpy --upgrade 。
matplotlib。绘图库。一整套和MATLAB相似命令API。适合交互式制图。线图、散点图、等高线图、条形图、柱状图、3D图。绘图控件，嵌入GUI应用。可视化训练结果、特征映射。pip install matplotlib --upgrade 。
jupyter notebook。Ipython升级版。浏览器创建、共享代码、方程、文档。基于Tornado框架Web应用，MQ消息管理。pip install jupyter --upgrade 。打开 jupyter notebook 。浏览器自动打开，启动成功。
scikit-image。图像处理算法，过滤图片。pip install scikit-image --upgrade 。
librosa。音频特征提取。pip install librosa --upgrade 。
nltk。语料库。自然语言处理，分词、词性标注、命名实体识别(NER)、句法分析。pip install nltk --upgrade 。nltk.download()下载nltk数据源。
keras。第一个TensorFlow核心高级别框架，默认API。pip install keras --upgrade 。
tflearn。pip install git+https://github.com/tflearn/tflearn.git 。

参考资料：
《TensorFlow技术解析与实战》

