**[TF API数学计算](https://www.tensorflow.org/api_guides/python/math_ops)**
tf...... ：math
（1）刚开始先给一个运行实例。
        tf是基于图（Graph）的计算系统。而图的节点则是由操作（Operation）来构成的，而图的各个节点之间则是由张量（Tensor）作为边来连接在一起的。所以Tensorflow的计算过程就是一个Tensor流图。Tensorflow的图则是必须在一个Session中来计算。
![](http://upload-images.jianshu.io/upload_images/4618424-c3f09c6002bea64a?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

 ```
import tensorflow as tf  
import numpy as np  
#定义占位符，此处是一个二维的tensor；这个在之后构建神经时候，对于model输入数据是要用到的。  
X = tf.placeholder(tf.float32, [None, 1])  
# 构建图。  
out  = tf.abs(X)。  
# 在session中启动图。  
sess = tf.Session()  
# 给定输入数据，直接给定数组。  
# b = np.array([-23.51])  
# 给定的b是一维数组，注意下面要加[]  
# b_abs = sess.run(out, feed_dict={X: [b]})  
# 给定的输入是一个列表。  
a = []  
a.append(12.3)  
a.append(-34.4)  
b = np.array(a)  
b = b.reshape([-1,1])  
b_abs = sess.run(out, feed_dict={X:b})  
print (a_abs)  
sess.close()  
```
注意多个Graph的使用：
```
import tensorflow as tf  
g1 = tf.Graph()  
with g1.as_default():  
    c1 = tf.constant([1.0])  
with tf.Graph().as_default() as g2:  
    c2 = tf.constant([2.0])  
  
with tf.Session(graph=g1) as sess1:  
    print sess1.run(c1)  
with tf.Session(graph=g2) as sess2:  
    print sess2.run(c2)  
  
# result:  
# [ 1.0 ]  
# [ 2.0 ]  
```
2）tf.a...API：
tensor可以是一维、二维、.....可以到多维
Arithmetic Operators

1.1 tf.add(x,y,name=None)

功能：对应位置元素的加法运算。
输入：x,y具有相同尺寸的tensor，可以为`half`, `float32`, `float64`, `uint8`, `int8`, `int16`, `int32`, `int64`, 
`complex64`, `complex128`, `string‘类型。
例：
x=tf.constant(1.0)
y=tf.constant(2.0)
z=tf.add(x,y)

z==>(3.0)
1.2 tf.subtract(x,y,name=None)

功能：对应位置元素的减法运算。
输入：x,y具有相同尺寸的tensor，可以为`half`, `float32`, `float64`,  `int32`, `int64`, `complex64`, `complex128`, 
`string‘类型。
例：
x=tf.constant([[1.0,-1.0]],tf.float64)
y=tf.constant([[2.2,2.3]],tf.float64)
z=tf.subtract(x,y)

z==>[[-1.2,-3.3]]
1.3 tf.multiply(x,y,name=None)

功能：对应位置元素的乘法运算。
输入：x,y具有相同尺寸的tensor，可以为`half`, `float32`, `float64`, `uint8`, `int8`, `uint16`,`int16`, `int32`, `int64`, 
`complex64`, `complex128`, `string‘类型。
例：
x=tf.constant([[1.0,-1.0]],tf.float64)
y=tf.constant([[2.2,2.3]],tf.float64)
z=tf.multiply(x,y)

z==>[[2.2,-2.3]]
1.4 tf.scalar_mul(scalar,x)

功能：固定倍率缩放。
输入：scalar必须为0维元素，x为tensor。
例：
scalar=2.2
x=tf.constant([[1.2,-1.0]],tf.float64)
z=tf.scalar_mul(scalar,x)

z==>[[2.64,-2.2]]
1.5 tf.div(x,y,name=None)[推荐使用tf.divide(x,y)]

功能：对应位置元素的除法运算（使用python2.7除法算法，如果x,y有一个为浮点数，结果为浮点数;否则为整数，但使用该函数会报错）。
输入：x,y具有相同尺寸的tensor，x为被除数，y为除数。
例：
x=tf.constant([[1,4,8]],tf.int32)
y=tf.constant([[2,3,3]],tf.int32)
z=tf.div(x,y)

z==>[[0,1,2]]

x=tf.constant([[1,4,8]],tf.int64)
y=tf.constant([[2,3,3]],tf.int64)
z=tf.divide(x,y)

z==>[[0.5,1.33333333,2.66666667]]

x=tf.constant([[1,4,8]],tf.float64)
y=tf.constant([[2,3,3]],tf.float64)
z=tf.div(x,y)

z==>[[0.5,1.33333333,2.66666667]]
1.6 tf.truediv(x,y,name=None)

功能：对应位置元素的除法运算。（使用python3除法算法，又叫真除，结果为浮点数，推荐使用tf.divide）
输入：x,y具有相同尺寸的tensor，x为被除数，y为除数。
1.7 tf.floordiv(x,y,name=None)

功能：对应位置元素的地板除法运算。返回不大于结果的最大整数
输入：x,y具有相同尺寸的tensor，x为被除数，y为除数。
例：
x=tf.constant([[2,4,-1]],tf.int64) #float类型运行结果一致，只是类型为浮点型
y=tf.constant([[3,3,3]],tf.int64)
z=tf.floordiv(x,y)

z==>[[0,1,-1]]
1.8 tf.realdiv(x,y,name=None)

功能：对应位置元素的实数除法运算。实际情况不非官方描述，与divide结果没区别，
输入：x,y具有相同尺寸的tensor，可以为`half`, `float32`, `float64`, `uint8`, `int8`, `int16`, `int32`, `int64`, 
`complex64`, `complex128`, `string‘类型。
例：
x=tf.constant([[2+1j,4+2j,-1+3j]],tf.complex64)
y=tf.constant([[3+3j,3+1j,3+2j]],tf.complex64)
z=tf.realdiv(x,y)

z==>[[0.50000000-0.16666667j 1.39999998+0.2j 0.23076922+0.84615386j]]
1.9 tf.truncatediv(x,y,name=None)

功能：对应位置元素的截断除法运算，获取整数部分。（和手册功能描述不符，符号位并不能转为0）
输入：x,y具有相同尺寸的tensor，可以为`uint8`, `int8`, `int16`, `int32`, `int64`,类型。(只能为整型，浮点型等并未注册，和手册不符)
例：
x=tf.constant([[2,4,-7]],tf.int64)
y=tf.constant([[3,3,3]],tf.int64)
z=tf.truncatediv(x,y)

z==>[[0 1 -2]]
1.10 tf.floor_div(x,y,name=None)

功能：对应位置元素的地板除法运算。（和tf.floordiv运行结果一致，只是内部实现方式不一样）
输入：x,y具有相同尺寸的tensor，可以为`half`, `float32`, `float64`, `uint8`, `int8`, `int16`, `int32`, `int64`, 
`complex64`, `complex128`, `string‘类型。
1.11 tf.truncatemod(x,y,name=None)

功能：对应位置元素的截断除法取余运算。
输入：x,y具有相同尺寸的tensor，可以为float32`, `float64`,  `int32`, `int64`类型。
例：
x=tf.constant([[2.1,4.1,-1.1]],tf.float64)
y=tf.constant([[3,3,3]],tf.float64)
z=tf.truncatemod(x,y)

z==>[[2.1 1.1 -1.1]]
1.12 tf.floormod(x,y,name=None)

功能：对应位置元素的地板除法取余运算。
输入：x,y具有相同尺寸的tensor，可以为float32`, `float64`,  `int32`, `int64`类型。
例：
x=tf.constant([[2.1,4.1,-1.1]],tf.float64)
y=tf.constant([[3,3,3]],tf.float64)
z=tf.truncatemod(x,y)

z==>[[2.1 1.1 1.9]]
1.13 tf.mod(x,y,name=None)

功能：对应位置元素的除法取余运算。若x和y只有一个小于0，则计算‘floor（x/y）*y+mod(x,y)’。
输入：x,y具有相同尺寸的tensor，可以为`float32`, `float64`,  `int32`, `int64`类型。
例：
x=tf.constant([[2.1,4.1,-1.1]],tf.float64)
y=tf.constant([[3,3,3]],tf.float64)
z=tf.mod(x,y)

z==>[[2.1 1.1 1.9]]
1.14 tf.cross(x,y,name=None)

功能：计算叉乘。最大维度为3。
输入：x,y具有相同尺寸的tensor，包含3个元素的向量
例：
x=tf.constant([[1,2,-3]],tf.float64)
y=tf.constant([[2,3,4]],tf.float64)
z=tf.cross(x,y)

z==>[[17. -10. -1]]#2×4-（-3）×3=17，-（1×4-（-3）×2）=-10，1×3-2×2=-1。
Basic Math Functions

1.15 tf.add_n(inputs,name=None)

功能：将所有输入的tensor进行对应位置的加法运算
输入：inputs：一组tensor，必须是相同类型和维度。
例：
x=tf.constant([[1,2,-3]],tf.float64)
y=tf.constant([[2,3,4]],tf.float64)
z=tf.constant([[1,4,3]],tf.float64)
xyz=[x,y,z]
z=tf.add_n(xyz)

z==>[[4. 9. 4.]]
1.16 tf.abs(x,name=None)

功能：求x的绝对值。
输入：x为张量或稀疏张量，可以为`float32`, `float64`,  `int32`, `int64`类型。
例：
x=tf.constant([[1.1,2,-3]],tf.float64)
z=tf.abs(x)

z==>[[1.1 2. 3.]]
1.17 tf.negative(x,name=None)

功能：求x的负数。
输入：x为张量或稀疏张量，可以为`half`,`float32`, `float64`,  `int32`, `int64`,`complex64`,`complex128`类型。
例：
x=tf.constant([[1.1,2,-3]],tf.float64)
z=tf.negative(x)

z==>[[-1.1. -2. 3.]]
1.18 tf.sign(x,name=None)

功能：求x的符号，x>0,则y=1;x<0则y=-1;x=0则y=0。
输入：x,为张量，可以为`half`,`float32`, `float64`,  `int32`, `int64`,`complex64`,`complex128`类型。
例：
x=tf.constant([[1.1,0,-3]],tf.float64)
z=tf.sign(x)

z==>[[1. 0. -1.]]
1.19 tf.reciprocal(x,name=None)

功能：求x的倒数。
输入：x为张量，可以为`half`,`float32`, `float64`,  `int32`, `int64`,`complex64`,`complex128`类型。
例：
x=tf.constant([[2,0,-3]],tf.float64)
z=tf.reciprocal(x)

z==>[[0.5 inf -0.33333333]]
1.20 tf.square(x,name=None)

功能：计算x各元素的平方。
输入：x为张量或稀疏张量，可以为`half`,`float32`, `float64`,  `int32`, `int64`,`complex64`,`complex128`类型。
例：
x=tf.constant([[2,0,-3]],tf.float64)
z=tf.square(x)

z==>[[4. 0. 9.]]
1.21 tf.round(x,name=None)

功能：计算x各元素的距离其最近的整数，若在中间，则取偶数值。
输入：x为张量，可以为`float32`, `float64`类型。
例：
x=tf.constant([[0.9,1.1,1.5,-4.1,-4.5,-4.9]],tf.float64)
z=tf.round(x)

z==>[[1. 1. 2. -4. -4. -5.]]
1.22 tf.sqrt(x,name=None)

功能：计算x各元素的平方。
输入：x为张量，可以为`half`,`float32`, `float64`,`complex64`,`complex128`类型。
例：
x=tf.constant([[2,3,-5]],tf.float64)
z=tf.sqrt(x)

z==>[[1.41421356 1.73205081 nan]]
1.23 tf.rsqrt(x,name=None)

功能：计算x各元素的平方根的倒数。
输入：x为张量或稀疏张量，可以为`half`,`float32`, `float64`,`complex64`,`complex128`类型。
例：
x=tf.constant([[2,3,5]],tf.float64)
z=tf.rsqrt(x)

z==>[[0.70710678 0.57735027 0.4472136]]
1.24 tf.pow(x,y,name=None)

功能：计算x各元素的y次方。
输入：x，y为张量，可以为`float32`, `float64`, `int32`, `int64`,`complex64`,`complex128`类型。
例：
x=tf.constant([[2,3,5]],tf.float64)
y=tf.constant([[2,3,4]],tf.float64)
z=tf.pow(x,y)

z==>[[4. 27. 625.]]
1.25 tf.exp(x,name=None)

功能：计算x各元素的自然指数，即e^x。
输入：x为张量，可以为`half`,`float32`, `float64`,`complex64`,`complex128`类型。
例：
x=tf.constant([[0,1,-1]],tf.float64)
z=tf.exp(x)

z==>[[1. 2.71828183 0.36787944]]
1.26 tf.expm1(x,name=None)

功能：计算x各元素的自然指数减1，即e^x-1。
输入：x为张量，可以为`half`,`float32`, `float64`,`complex64`,`complex128`类型。
例：
x=tf.constant([[0,1,-1]],tf.float64)
z=tf.expm1(x)

z==>[[0. 1.71828183 -0.63212056]]
1.27 tf.log(x,name=None)

功能：计算x各元素的自然对数。
输入：x为张量，可以为`half`,`float32`, `float64`,`complex64`,`complex128`类型。
例：
x=tf.constant([[1,2.71828183,10]],tf.float64)
z=tf.log(x)

z==>[[0. 1. 2.30258509]]
1.28 tf.log1p(x,name=None)

功能：计算x各元素加1后的自然对数。
输入：x为张量，可以为`half`,`float32`, `float64`,`complex64`,`complex128`类型。
例：
x=tf.constant([[0,1.71828183,9]],tf.float64)
z=tf.log1p(x)

z==>[[0. 1. 2.30258509]]
1.29 tf.ceil(x,name=None)

功能：计算x各元素比x大的最小整数。
输入：x为张量，可以为`half`,`float32`, `float64`类型。
例：
x=tf.constant([[0.2，0.8，-0.7]],tf.float64)
z=tf.ceil(x)

z==>[[1. 1. -0.]]
1.30 tf.floor(x,name=None)

功能：计算x各元素比其小的最大整数。
输入：x为张量，可以为`half`,`float32`, `float64`类型。
例：
x=tf.constant([[0.2，0.8，-0.7]],tf.float64)
z=tf.floor(x)

z==>[[0. 0. -1.]]
1.31 tf.maximum(x,y,name=None)

功能：计算x,y对应位置元素较大的值。
输入：x，y为张量，可以为`half`,`float32`, `float64`,  `int32`, `int64`类型。
例：
x=tf.constant([[0.2,0.8,-0.7]],tf.float64)
y=tf.constant([[0.2,0.5,-0.3]],tf.float64)
z=tf.maximum(x,y)

z==>[[0.2 0.8 -0.3]]
1.32 tf.minimum(x,y,name=None)

功能：计算x,y对应位置元素较小的值。
输入：x，y为张量，可以为`half`,`float32`, `float64`,  `int32`, `int64`类型。
例：
x=tf.constant([[0.2,0.8,-0.7]],tf.float64)
y=tf.constant([[0.2,0.5,-0.3]],tf.float64)
z=tf.maximum(x,y)

z==>[[0.2 0.5 -0.7]]
1.33 tf.cos(x,name=None)

功能：计算x的余弦值。
输入：x为张量，可以为`half`,`float32`, `float64`,  `complex64`, `complex128`类型。
例：
x=tf.constant([[0,3.1415926]],tf.float64)
z=tf.cos(x)

z==>[[1. -1.]]
1.34 tf.sin(x,name=None)

功能：计算x的正弦值。
输入：x为张量，可以为`half`,`float32`, `float64`,  `complex64`, `complex128`类型。
例：
x=tf.constant([[0,1.5707963]],tf.float64)
z=tf.sin(x)

z==>[[0. 1.]]
1.35 tf.lbeta(x,name=None)

功能：计算`ln(|Beta(x)|)`,并以最末尺度进行归纳。
          最末尺度`z = [z_0,...,z_{K-1}]`，则Beta(z) = \prod_j Gamma(z_j) / Gamma(\sum_j z_j)
输入：x为秩为n+1的张量，可以为'float','double'类型。
例：
x=tf.constant([[4,3,3],[2,3,2]],tf.float64)
z=tf.lbeta(x)

z==>[-9.62377365 -5.88610403]
#ln(gamma(4)*gamma(3)*gamma(3)/gamma(4+3+3))=ln(6*2*2/362880)=-9.62377365
#ln(gamma(2)*gamma(3)*gamma(2)/gamma(2+3+2))=ln(2/720)=-5.88610403
1.36 tf.tan(x,name=None)

功能：计算tan(x)。
输入：x为张量，可以为`half`,`float32`, `float64`,  `int32`, `int64`,`complex64`, `complex128`类型。
例：
x=tf.constant([[0,0.785398163]],tf.float64)
z=tf.tan(x)

z==>[[0. 1.]]
1.37 tf.acos(x,name=None)

功能：计算acos(x)。
输入：x为张量，可以为`half`,`float32`, `float64`,  `int32`, `int64`,`complex64`, `complex128`类型。
例：
x=tf.constant([[0,1,-1]],tf.float64)
z=tf.acos(x)

z==>[[1.57079633 0. 3.14159265]]
1.38 tf.asin(x,name=None)

功能：计算asin(x)。
输入：x为张量，可以为`half`,`float32`, `float64`,  `int32`, `int64`,`complex64`, `complex128`类型。
例：
x=tf.constant([[0,1,-1]],tf.float64)
z=tf.asin(x)

z==>[[0. 1.57079633 -1.57079633]]
1.39 tf.atan(x,name=None)

功能：计算atan(x)。
输入：x为张量，可以为`half`,`float32`, `float64`,  `int32`, `int64`,`complex64`, `complex128`类型。
例：
x=tf.constant([[0,1,-1]],tf.float64)

z=tf.atan(x)

z==>[[0. 0.78539816 -0.78539816]]
1.40 tf.lgamma(x,name=None)

功能：计算ln(gamma(x))。
输入：x为张量，可以为`half`,`float32`, `float64`类型。
例：
x=tf.constant([[1,2,3]],tf.float64)
z=tf.lgamma(x)

z==>[[0. 0. 0.69314718]]
1.41 tf.digamma(x,name=None)

功能：计算lgamma的导数，即gamma‘/gamma。
输入：x，y为张量，可以为`half`,`float32`, `float64`类型。
例：
x=tf.constant([[1,2,3]],tf.float64)
z=tf.digamma(x)

z==>[[-0.57721566 0.42278434 0.92278434]]
1.42 tf.erf(x,name=None)

功能：计算x的高斯误差。
输入：x为张量或稀疏张量，可以为`half`,`float32`, `float64`类型。
例：
x=tf.constant([[-1,0,1,2,3]],tf.float64)
z=tf.erf(x)

z==>[[-0.84270079 0. 0.84270079 0.99532227 0.99997791]]
1.43 tf.erfc(x,name=None)

功能：计算x高斯互补误差。
输入：x为张量，可以为`half`,`float32`, `float64`类型。
例：
x=tf.constant([[-1,0,1,2,3]],tf.float64)
z=tf.erfc(x)

z==>[[1.84270079 1.00000000 0.15729920 4.67773498e-03 2.20904970e-05]]
1.44 tf.squared_difference(x,y,name=None)

功能：计算(x-y)(x-y)。
输入：x为张量，可以为`half`,`float32`, `float64`类型。
例：
x=tf.constant([[-1,0,2]],tf.float64)
y=tf.constant([[2,3,4,]],tf.float64)
z=tf.squared_difference(x,y)

z==>[[9. 9. 4.]]
1.45 tf.igamma(a,x,name=None)

功能：计算gamma(a,x)/gamma(a),gamma(a,x)=\intergral_from_0_to_x t^(a-1)*exp^(-t)dt。
输入：x为张量，可以为`float32`, `float64`类型。
例：
a=tf.constant(1,tf.float64)
x=tf.constant([[1,2,3,4]],tf.float64)
z=tf.igamma(a,x)

z==>[[0.63212056 0.86466472 0.95021293 0.98168436]]
1.46 tf.igammac(a,x,name=None)

功能：计算gamma(a,x)/gamma(a),gamma(a,x)=\intergral_from_x_to_inf t^(a-1)*exp^(-t)dt。
输入：x为张量，可以为`float32`, `float64`类型。
例：
x=tf.constant([[-1,0,1,2,3]],tf.float64)
z=tf.erf(x)

z==>[[-0.84270079 0. 0.84270079 0.99532227 0.99997791]]
1.47 tf.zeta(x,q,name=None)

功能：计算Hurwitz zeta函数。
输入：x为张量或稀疏张量，可以为`float32`, `float64`类型。
例：
a=tf.constant(1,tf.float64)
x=tf.constant([[1,2,3,4]],tf.float64)
z=tf.zeta(x,a)

z==>[[inf 1.64493407 1.2020569 1.08232323]]
1.48 tf.polygamma(a,x,name=None)

功能：计算psi^{(a)}(x),psi^{(a)}(x) = ({d^a}/{dx^a})*psi(x),psi即为polygamma。    
输入：x为张量，可以为`float32`, `float64`类型。a=tf.constant(1,tf.float64) 
例：
x=tf.constant([[1,2,3,4]],tf.float64)z=tf.polygamma(a,x) 

z==>[[1.64493407 0.64493407 0.39493407 0.28382296]]
1.49 tf.betainc(a,b,x,name=None)

功能：计算I_x(a, b)。I_x(a, b) = {B(x; a, b)}/{B(a, b)}。
                    B(x; a, b) = \intergral_from_0_to_x t^{a-1} (1 - t)^{b-1} dt。
                    B(a, b) = \intergral_from_0_to_1 t^{a-1} (1 - t)^{b-1} dt。即完全beta函数。          
输入：x为张量，可以为`float32`, `float64`类型。a,b与x同类型。
例：
a=tf.constant(1,tf.float64)b=tf.constant(1,tf.float64)x=tf.constant([[0,0.5,1]],tf.float64) 

z==>[[0. 0.5 1.]]
1.50 tf.rint(x,name=None)

功能：计算离x最近的整数，若为中间值，取偶数值。
输入：x为张量，可以为`half`,`float32`, `float64`类型。
例：
x=tf.constant([[-1.7,-1.5,-1.1,0.1,0.5,0.4,1.5]],tf.float64)
z=tf.rint(x)

z==>[[-2. -2. -1. 0. 0. 0. 2.]]

矩阵数学函数：
1.51 tf.diag(diagonal, name=None)

功能：返回对角阵。
输入：tensor，秩为k<=3。
例：
a=tf.constant([1,2,3,4])
z=tf.diag(a)

z==>[[1 0 0 0
      0 2 0 0
      0 0 3 0
      0 0 0 4]]
1.52 tf.diag_part(input,name=None)

功能：返回对角阵的对角元素。
输入：tensor,且维度必须一致。
例：
a=tf.constant([[1,5,0,0],[0,2,0,0],[0,0,3,0],[0,0,0,4]])
z=tf.diag_part(a)

z==>[1,2,3,4]
1.53 tf.trace(x,name=None)

功能：返回矩阵的迹。
输入：tensor
例：
a=tf.constant([[[1,2,3],[4,5,6],[7,8,9]],[[10,11,12],[13,14,15],[16,17,18]]])
z=tf.trace(a)

z==>[15 42]
1.54 tf.transpose(a,perm=None,name='transpose')

功能：矩阵转置。
输入：tensor，perm代表转置后的维度排列，决定了转置方法，默认为[n-1,....,0]，n为a的维度。
例：
a=tf.constant([[1,2,3],[4,5,6]])
z=tf.transpose(a)#perm为[1,0]，即0维和1维互换。

z==>[[1 4]
     [2 5]
     [3 6]]
1.55 tf.eye(num_rows, num_columns=None, batch_shape=None, dtype=tf.float32, name=None)

功能：返回单位阵。
输入：num_rows:矩阵的行数;num_columns:矩阵的列数，默认与行数相等
            batch_shape:若提供值，则返回batch_shape的单位阵。
例：
z=tf.eye(2,batch_shape=[2])

z==>[[[1. 0.]
      [0. 1.]]
     [[1. 0.]
      [0. 1.]]]
1.56 tf.matrix_diag(diagonal,name=None)

功能：根据对角值返回一批对角阵
输入：对角值
例：
a=tf.constant([[1,2,3],[4,5,6]])
z=tf.matrix_diag(a)

z==>[[[1 0 0]
      [0 2 0]
      [0 0 3]]
     [[4 0 0]
      [0 5 0]
      [0 0 6]]]
1.57 tf.matrix_diag_part(input,name=None)

功能：返回批对角阵的对角元素
输入：tensor,批对角阵
例：
a=tf.constant([[[1,3,0],[0,2,0],[0,0,3]],[[4,0,0],[0,5,0],[0,0,6]]])
z=tf.matrix_diag_part(a)

z==>[[1 2 3]
     [4 5 6]]
1.58 tf.matrix_band_part(input,num_lower,num_upper,name=None)

功能：复制一个矩阵，并将规定带之外的元素置为0。
     假设元素坐标为（m，n），则in_band(m, n) = (num_lower < 0 || (m-n) <= num_lower)) &&
                                          (num_upper < 0 || (n-m) <= num_upper)。
    band（m,n）=in_band(m,n)*input(m,n)。
    特殊情况：
          tf.matrix_band_part(input, 0, -1) ==> 上三角阵.
          tf.matrix_band_part(input, -1, 0) ==> 下三角阵.
          tf.matrix_band_part(input, 0, 0) ==> 对角阵.
输入：num_lower:如果为负，则结果右上空三角阵;
     num_lower:如果为负，则结果左下为空三角阵。
例：
a=tf.constant([[0,1,2,3],[-1,0,1,2],[-2,-1,0,1],[-3,-2,-1,0]])
z=tf.matrix_band_part(a,1,-1)

z==>[[0 1 2 3]
     [-1 0 1 2]
     [0 -1 0 1]
     [0 0 -1 0]]
1.59 tf.matrix_set_diag(input,diagonal,name=None)

功能：将输入矩阵的对角元素置换为对角元素。
输入：input：矩阵，diagonal：对角元素。
例：
a=tf.constant([[0,1,2,3],[-1,0,1,2],[-2,-1,0,1],[-3,-2,-1,0]])
z=tf.matrix_set_diag(a,[10,11,12,13])

z==>[[10  1  2  3]
     [-1 11  1  2]
     [0  -1 12  1]
     [0   0 -1 13]]
1.60 tf.matrix_transpose(a,name='matrix_transpose')

功能：进行矩阵转置。只对低维度的2维矩阵转置，功能同tf.transpose(a,perm=[0,1,3,2])。(若a为4维)
输入：矩阵。
1.61 tf.matmul(a, b, transpose_a=False, transpose_b=False, adjoint_a=False, adjoint_b=False, a_is_sparse=False, b_is_sparse=False, name=None)

功能：矩阵乘法。配置后的矩阵a，b必须满足矩阵乘法对行列的要求。
输入：transpose_a,transpose_b:运算前是否转置;
      adjoint_a,adjoint_b:运算前进行共轭;
     a_is_sparse,b_is_sparse:a，b是否当作稀疏矩阵进行运算。
例：
a = tf.constant([1, 2, 3, 4, 5, 6], shape=[2, 3])
b = tf.constant([7, 8, 9, 10, 11, 12], shape=[3, 2])
z = tf.matmul(a, b)

z==>[[ 58  64]
     [139 154]]
1.62 tf.norm(tensor, ord='euclidean', axis=None, keep_dims=False, name=None)

功能：求取范数。
输入：ord：范数类型，默认为‘euclidean’，支持的有‘fro’，‘euclidean’，‘0’，‘1’，‘2’，‘np.inf’;
         axis：默认为‘None’，tensor为向量。
     keep_dims:默认为‘None’，结果为向量，若为True，保持维度。
例：
a = tf.constant([1, 2, 3, 4, 5, 6], shape=[2, 3],dtype=tf.float32)
z = tf.norm(a)
z2=tf.norm(a,ord=1)
z3=tf.norm(a,ord=2)
z4=tf.norm(a,ord=1,axis=0)
z5=tf.norm(a,ord=1,axis=1)
z6=tf.norm(a,ord=1,axis=1，keep_dims=True)

z==>9.53939
z2==>21.0
z3==>9.53939
z4==>[5.  7.  9.]
z5==>[6.  15.]
z6==>[[6.]
      [15.]]
1.63 tf.matrix_determinant(input, name=None)

功能：求行列式。
输入：必须是float32，float64类型。
例：
a = tf.constant([1, 2, 3, 4],shape=[2,2],dtype=tf.float32)
z = tf.matrix_determinant(a)

z==>-2.0
1.64 tf.matrix_inverse(input, adjoint=None, name=None)

功能：求矩阵的逆。
输入：输入必须是float32，float64类型。adjoint表示计算前求转置 
例：
a = tf.constant([1, 2, 3, 4],shape=[2,2],dtype=tf.float64)
z = tf.matrix_inverse(a)

z==>[[-2.    1.]
     [1.5  -0.5]]
1.65 tf.cholesky(input, name=None)

功能：进行cholesky分解。
输入：注意输入必须是正定矩阵。
例：
a = tf.constant([2, -2, -2, 5],shape=[2,2],dtype=tf.float64)
z = tf.cholesky(a)

z==>[[ 1.41421356  0.        ]
     [-1.41421356  1.73205081]]
1.66 tf.cholesky_solve(chol, rhs, name=None)

功能：对方程‘AX=RHS’进行cholesky求解。
输入：chol=tf.cholesky(A)。
例：
a = tf.constant([2, -2, -2, 5],shape=[2,2],dtype=tf.float64)
chol = tf.cholesky(a)
RHS=tf.constant([3,10],shape=[2,1],dtype=tf.float64)
z=tf.cholesky_solve(chol,RHS)

z==>[[5.83333333]
     [4.33333333]] #A*X=RHS
1.67 tf.matrix_solve(matrix, rhs, adjoint=None, name=None)

功能：求线性方程组，matrix*X=rhs。
输入：adjoint:是否对matrix转置。
例：
a = tf.constant([2, -2, -2, 5],shape=[2,2],dtype=tf.float64)
RHS=tf.constant([3,10],shape=[2,1],dtype=tf.float64)
z=tf.matrix_solve(a,RHS)

z==>[[5.83333333]
     [4.33333333]]
1.68 tf.matrix_triangular_solve(matrix, rhs, lower=None, adjoint=None, name=None)

功能：求解matrix×X=rhs，matrix为上三角或下三角阵。
输入：lower：默认为None，matrix上三角元素为0;若为True，matrix下三角元素为0;
     adjoint：转置
例：
a = tf.constant([2, 4, -2, 5],shape=[2,2],dtype=tf.float64)
RHS=tf.constant([3,10],shape=[2,1],dtype=tf.float64)
z=tf.matrix_triangular_solve(a,RHS)

z==>[[1.5]
     [2.6]]
1.69 tf.matrix_solve_ls(matrix, rhs, l2_regularizer=0.0, fast=True, name=None)

功能：求解多个线性方程的最小二乘问题。
输入：。
例：
a = tf.constant([2, 4, -2, 5],shape=[2,2],dtype=tf.float64)
RHS=tf.constant([3,10],shape=[2,1],dtype=tf.float64)
z=tf.matrix_solve_ls(a,RHS)

z==>[[-1.38888889]
     [1.44444444]]
1.70 tf.qr(input, full_matrices=None, name=None)

功能：对矩阵进行qr分解。
输入：。
例：
a = tf.constant([1,2,2,1,0,2,0,1,1],shape=[3,3],dtype=tf.float64)
q,r=tf.qr(a)

q==>[[-0.70710678   0.57735027   -0.40824829]
     [-0.70710678  -0.57735027    0.40824829]
     [0.            0.57735027   0.81649658 ]]
r==>[[-1.41421356  -1.41421356   -2.82842712]
     [0.            1.73205081    0.57735027]
     [0.            0.            0.81649658]]
1.71 tf.self_adjoint_eig(tensor, name=None)

功能：求取特征值和特征向量。
输入：
例：
a = tf.constant([3,-1,-1,3],shape=[2,2],dtype=tf.float64)
e,v=tf.self_adjoint_eig(a)

e==>[2.  4.]
v==>[[0.70710678 0.70710678]
    [0.70710678 -0.70710678]]
1.72 tf.self_adjoint_eigvals(tensor, name=None)

功能：计算多个矩阵的特征值。
输入：shape=[....,N,N]。
1.73 tf.svd(tensor, full_matrices=False, compute_uv=True, name=None)

功能：进行奇异值分解。tensor=u×diag（s）×transpose（v）
输入：
例：
a = tf.constant([3,-1,-1,3],shape=[2,2],dtype=tf.float64)
s,u,v=tf.svd(a)

s==>[4. 2.]
u==>[[0.70710678 0.70710678]
    [-0.70710678 0.70710678]]
v==>[[0.70710678 0.70710678]
    [-0.70710678 0.70710678]]

Tensor Math Function：

1.74 tf.tensordot(a, b, axes, name=None)

功能：同numpy.tensordot，根据axis计算点乘。
输入：axes=1或axes=[[1],[0]]，即为矩阵乘。
例：
a = tf.constant([1,2,3,4],shape=[2,2],dtype=tf.float64)
b = tf.constant([1,2,3,4],shape=[2,2],dtype=tf.float64)
z=tf.tensordot(a,b,axes=[[1],[1]])

z==>[[5.  11.]
     [11. 25.]]
Complex Number Functions

1.75 tf.complex(real, imag, name=None)

功能：将实数转化为复数。
输入：real，imag：float32或float64。
例：
real = tf.constant([1,2],dtype=tf.float64)
imag = tf.constant([3,4],dtype=tf.float64)
z=tf.complex(real,imag)

z==>[1.+3.j  2.+4.j]
1.76 tf.conj(x, name=None)

功能：返回x的共轭复数。
输入：
例：
a = tf.constant([1+2j,2-3j])
z=tf.conj(a)

z==>[1.-2.j  2.+3.j]
1.77 tf.imag(input, name=None)

功能：返回虚数部分。
输入：`complex64`,`complex128`类型。
例：
a = tf.constant([1+2j,2-3j])
z=tf.imag(a)

z==>[2.  -3.]
1.78 tf.real(input,name=None)

功能：返回实数部分。
输入：`complex64`,`complex128`类型。
例：
a = tf.constant([1+2j,2-3j])
z=tf.real(a)

z==>[1.  2.]
1.79 ～1.84 fft变换

函数：
tf.fft(input, name=None)
tf.ifft(input, name=None)
tf.fft2d(input, name=None)
tf.ifft2d(input, name=None)
tf.fft3d(input, name=None)
tf.ifft3d(input, name=None)
Reduction

1.85 tf.reduce_sum(input_tensor, axis=None, keep_dims=False, name=None, reduction_indices=None)

功能：沿着维度axis计算元素和，除非keep_dims=True，输出tensor保持维度为1。
输入：axis：默认为None，即沿所有维度求和。
例：
a = tf.constant([[1,2,3],[4,5,6]])
z=tf.reduce_sum(a)
z2=tf.reduce_sum(a,0)
z3=tf.reduce_sum(a,1)

z==>21
z2==>[5 7 9]
z3==>[6 15]
1.86 tf.reduce_prod(input_tensor, axis=None, keep_dims=False, name=None, reduction_indices=None)

功能：沿着维度axis计算元素积，除非keep_dims=True，输出tensor保持维度为1。
输入：axis：默认为None，即沿所有维度求和。
例：
a = tf.constant([[1,2,3],[4,5,6]])
z=tf.reduce_prod(a)
z2=tf.reduce_prod(a,0)
z3=tf.reduce_prod(a,1)

z==>720
z2==>[4 10 18]
z3==>[6 120]
1.87 tf.reduce_min(input_tensor, axis=None, keep_dims=False, name=None, reduction_indices=None)

功能：沿着维度axis计算最小值，除非keep_dims=True，输出tensor保持维度为1。
输入：axis：默认为None，即沿所有维度求和。
例：
a = tf.constant([[1,2,3],[4,5,6]])
z=tf.reduce_min(a)
z2=tf.reduce_min(a,0)
z3=tf.reduce_min(a,1)

z==>1
z2==>[1 2 3]
z3==>[1 4]
1.88 tf.reduce_max(input_tensor, axis=None, keep_dims=False, name=None, reduction_indices=None)

功能：沿着维度axis计算最大值，除非keep_dims=True，输出tensor保持维度为1。
输入：axis：默认为None，即沿所有维度求和。
例：
a = tf.constant([[1,2,3],[4,5,6]])
z=tf.reduce_max(a)
z2=tf.reduce_max(a,0)
z3=tf.reduce_max(a,1)

z==>6
z2==>[4 5 6]
z3==>[3 6]
1.89 tf.reduce_mean(input_tensor, axis=None, keep_dims=False, name=None, reduction_indices=None)

功能：沿着维度axis计算平均值，除非keep_dims=True，输出tensor保持维度为1。
输入：axis：默认为None，即沿所有维度求和。
例：
a = tf.constant([[1,2,3],[4,5,6]],dtype=tf.float64)
z=tf.reduce_mean(a)
z2=tf.reduce_mean(a,0)
z3=tf.reduce_mean(a,1)

z==>3.5
z2==>[2.5 3.5 4.5]
z3==>[2. 5.]
1.90 tf.reduce_all(input_tensor, axis=None, keep_dims=False, name=None, reduction_indices=None)

功能：沿着维度axis计算逻辑与，除非keep_dims=True，输出tensor保持维度为1。
输入：axis：默认为None，即沿所有维度求和。
例：
a = tf.constant([[True,True,False,False],[True,False,False,True]])
z=tf.reduce_all(a)
z2=tf.reduce_all(a,0)
z3=tf.reduce_all(a,1)

z==>False
z2==>[True False False False]
z3==>[False False]
1.91 tf.reduce_any(input_tensor, axis=None, keep_dims=False, name=None, reduction_indices=None)

功能：沿着维度axis计算逻辑或，除非keep_dims=True，输出tensor保持维度为1。
输入：axis：默认为None，即沿所有维度求和。
例：
a = tf.constant([[True,True,False,False],[True,False,False,True]])
z=tf.reduce_any(a)
z2=tf.reduce_any(a,0)
z3=tf.reduce_any(a,1)

z==>True
z2==>[True True False True]
z3==>[True True]
1.92 tf.reduce_logsumexp(input_tensor, axis=None, keep_dims=False, name=None, reduction_indices=None)

功能：沿着维度axis计算log(sum(exp()))，除非keep_dims=True，输出tensor保持维度为1。
输入：axis：默认为None，即沿所有维度求和。
例：
a = tf.constant([[0,0,0],[0,0,0]],dtype=tf.float64)
z=tf.reduce_logsumexp(a)
z2=tf.reduce_logsumexp(a,0)
z3=tf.reduce_logsumexp(a,1)

z==>1.79175946923#log(6)
z2==>[0.69314718 0.69314718 0.69314718]#[log(2) log(2) log(2)]
z3==>[1.09861229 1.09861229]#[log(3) log(3)]
1.93 tf.count_nonzero(input_tensor, axis=None, keep_dims=False, dtype=tf.int64, name=None, reduction_indices=None)

功能：沿着维度axis计算非0个数，除非keep_dims=True，输出tensor保持维度为1。
输入：axis：默认为None，即沿所有维度求和。
例：
a = tf.constant([[0,0,0],[0,1,2]],dtype=tf.float64)
z=tf.count_nonzero(a)
z2=tf.count_nonzero(a,0)
z3=tf.count_nonzero(a,1)

z==>2
z2==>[0 1 1]
z3==>[0 2]
1.94 tf.accumulate_n(inputs, shape=None, tensor_dtype=None, name=None)

功能：对应位置元素相加。如果输入是训练变量，不要使用，应使用tf.add_n。
输入：shape，tensor_dtype:类型检查
例：
a = tf.constant([[1,2],[3,4]])
b = tf.constant([[5,6],[7,8]])
z=tf.accumulate_n([a,b])

z==>[[6 8]
     [10 12]]
1.95 tf.einsum(equation, *inputs)

功能：通过equation进行矩阵乘法。
输入：equation：乘法算法定义。
# 矩阵乘
>>> einsum('ij,jk->ik', m0, m1)  # output[i,k] = sum_j m0[i,j] * m1[j, k]
# 点乘
>>> einsum('i,i->', u, v)  # output = sum_i u[i]*v[i]
# 向量乘
>>> einsum('i,j->ij', u, v)  # output[i,j] = u[i]*v[j]
# 转置
>>> einsum('ij->ji', m)  # output[j,i] = m[i,j]
# 批量矩阵乘
>>> einsum('aij,ajk->aik', s, t)  # out[a,i,k] = sum_j s[a,i,j] * t[a, j, k]
例：
a = tf.constant([[1,2],[3,4]])
b = tf.constant([[5,6],[7,8]])
z=tf.einsum('ij,jk->ik',a,b)

z==>[[19 22]
     [43 50]]
Scan

1.96 tf.cumsum(x, axis=0, exclusive=False, reverse=False, name=None)

功能：沿着维度axis进行累加。
输入：axis:默认为0
    reverse：默认为False，若为True，累加反向相反。
例：
a = tf.constant([[1,2,3],[4,5,6],[7,8,9]])
z=tf.cumsum(a)
z2=tf.cumsum(a,axis=1)
z3=tf.cumsum(a,reverse=True)

z==>[[1 2 3]
     [5 7 9]
     [12 15 18]]
z2==>[[1 3 6]
      [4 9 15]
      [7 15 24]]
z3==>[[12 15 18]
      [11 13 15]
      [7 8 9]]
1.97 tf.cumprod(x, axis=0, exclusive=False, reverse=False, name=None)

功能：沿着维度axis进行累积。
输入：axis:默认为0
    reverse：默认为False，若为True，累加反向相反。
例：
a = tf.constant([[1,2,3],[4,5,6],[7,8,9]])
z=tf.cumprod(a)
z2=tf.cumprod(a,axis=1)
z3=tf.cumprod(a,reverse=True)

z==>[[ 1  2   3]
     [ 5 10  18]
     [28 80 162]]
z2==>[[  1   2   6]
      [  4  20 120]
      [  7  56 504]]
z3==>[[ 28  80 162]
      [ 28  40  54]
      [  7   8   9]]
Segmentation

1.98 tf.segment_sum(data, segment_ids, name=None)

功能：tensor进行拆分后求和。
输入：segment_ids:必须是整型，1维向量，向量数目与data第一维的数量一致。
                必须从0开始，且以1进行递增。 
例：
a = tf.constant([[1,2,3],[4,5,6],[7,8,9]])
z=tf.segment_sum(a,[0,0,1])

z==>[[5 7 9]
     [7 8 9]]
1.99 tf.segment_prod(data, segment_ids, name=None)

功能：tensor进行拆分后求积。
输入：segment_ids:必须是整型，1维向量，向量数目与data第一维的数量一致。
                必须从0开始，且以1进行递增。 
例：
a = tf.constant([[1,2,3],[4,5,6],[7,8,9]])
z=tf.segment_prod(a,[0,0,1])

z==>[[4 10 18]
     [7  8  9]]
1.100 tf.segment_min(data, segment_ids, name=None)

功能：tensor进行拆分后求最小值。
输入：segment_ids:必须是整型，1维向量，向量数目与data第一维的数量一致。
                必须从0开始，且以1进行递增。 
例：
a = tf.constant([[1,2,3],[4,5,6],[7,8,9]])
z=tf.segment_min(a,[0,0,1])

z==>[[1 2 3]
     [7 8 9]]
1.101 tf.segment_max(data, segment_ids, name=None)

功能：tensor进行拆分后求最大值。
输入：segment_ids:必须是整型，1维向量，向量数目与data第一维的数量一致。
                必须从0开始，且以1进行递增。 
例：
a = tf.constant([[1,2,3],[4,5,6],[7,8,9]])
z=tf.segment_max(a,[0,0,1])

z==>[[4 5 6]
     [7 8 9]]
1.102 tf.segment_mean(data, segment_ids, name=None)

功能：tensor进行拆分后求平均值。
输入：segment_ids:必须是整型，1维向量，向量数目与data第一维的数量一致。
                必须从0开始，且以1进行递增。 
例：
a = tf.constant([[1,2,3],[4,5,6],[7,8,9]])
z=tf.segment_mean(a,[0,0,1])

z==>[[2 3 4]
     [7 8 9]]
1.103 tf.unsorted_segment_sum(data, segment_ids, num_segments, name=None)

功能：tensor进行拆分后求和。不同于sugementsum，segmentids不用按照顺序排列
输入：segment_ids:必须是整型，1维向量，向量数目与data第一维的数量一致。
    num_segments:分类总数，若多余ids匹配的数目，则置0。 
例：
a = tf.constant([[1,2,3],[4,5,6],[7,8,9]])
z=tf.unsorted_segment_sum(a,[0,1,0],2)
z2=tf.unsorted_segment_sum(a,[0,0,0],2)

z==>[[8 10 12]
     [4  5  6]]
z2==>[[12 15 18]
      [0  0  0]]
1.104 tf.unsorted_segment_max(data, segment_ids, num_segments, name=None)

功能：tensor进行拆分后求最大值。不同于sugementmax，segmentids不用按照顺序排列
输入：segment_ids:必须是整型，1维向量，向量数目与data第一维的数量一致。
    num_segments:分类总数，若多余ids匹配的数目，则置0。 
例：
a = tf.constant([[1,2,3],[4,5,6],[7,8,9]])
z=tf.unsorted_segment_max(a,[0,1,0],2)
z2=tf.unsorted_segment_max(a,[0,0,0],2)

z==>[[8 10 12]
     [4  5  6]]
z2==>[[12 15 18]
      [0  0  0]]
1.105 tf.sparse_segment_sum(data, indices, segment_ids, name=None)

功能：tensor进行拆分后求和。和segment_sum类似，只是segment_ids的rank数可以小于‘data’第0维度数。
输入：indices:选择第0维度参与运算的编号。
例：
a = tf.constant([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
z=tf.sparse_segment_sum(a, tf.constant([0, 1]), tf.constant([0, 0]))
z2=tf.sparse_segment_sum(a, tf.constant([0, 1]), tf.constant([0, 1]))
z3=tf.sparse_segment_sum(a, tf.constant([0, 2]), tf.constant([0, 1]))
z4=tf.sparse_segment_sum(a, tf.constant([0, 1,2]), tf.constant([0, 0,1]))

z==>[[6 8 10 12]]
z2==>[[1 2 3 4]
      [5 6 7 8]]
z3==>[[1 2 3 4]
      [9 10 11 12]]
z4==>[[6 8 10 12]
      [9 10 11 12]]
1.106 tf.sparse_segment_mean(data, indices, segment_ids, name=None)

功能：tensor进行拆分后求平均值。和segment_mean类似，只是segment_ids的rank数可以小于‘data’第0维度数。
输入：indices:选择第0维度参与运算的编号。
例：
a = tf.constant([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
z=tf.sparse_segment_mean(a, tf.constant([0, 1]), tf.constant([0, 0]))
z2=tf.sparse_segment_mean(a, tf.constant([0, 1]), tf.constant([0, 1]))
z3=tf.sparse_segment_mean(a, tf.constant([0, 2]), tf.constant([0, 1]))
z4=tf.sparse_segment_mean(a, tf.constant([0, 1,2]), tf.constant([0, 0,1]))

z==>[[3. 4. 5. 6.]]
z2==>[[1. 2. 3. 4.]
      [5. 6. 7. 8.]]
z3==>[[1. 2. 3. 4.]
      [9. 10. 11. 12.]]
z4==>[[3. 4. 5. 6.]
      [9. 10. 11. 12.]]
1.107 tf.sparse_segment_sqrt_n(data, indices, segment_ids, name=None)

功能：tensor进行拆分后求和再除以N的平方根。N为reduce segment数量。
     和segment_mean类似，只是segment_ids的rank数可以小于‘data’第0维度数。
输入：indices:选择第0维度参与运算的编号。
例：
a = tf.constant([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
z=tf.sparse_segment_sqrt_n(a, tf.constant([0, 1]), tf.constant([0, 0]))
z2=tf.sparse_segment_sqrt_n(a, tf.constant([0, 1]), tf.constant([0, 1]))
z3=tf.sparse_segment_sqrt_n(a, tf.constant([0, 2]), tf.constant([0, 1]))
z4=tf.sparse_segment_sqrt_n(a, tf.constant([0, 1,2]), tf.constant([0, 0,1]))

z==>[[4.24264069 5.65685424 7.07106781 8.48528137]]
z2==>[[1. 2. 3. 4.]
      [5. 6. 7. 8.]]
z3==>[[1. 2. 3. 4.]
      [9. 10. 11. 12.]]
z4==>[[4.24264069 5.65685424 7.07106781 8.48528137]
      [9. 10. 11. 12.]]
Sequence Comparison and Indexing

1.108 tf.argmin(input, axis=None, name=None, dimension=None)

功能：返回沿axis维度最小值的下标。
输入：
例：
a = tf.constant([[1,2,3,4], [5,6,7,8], [9,10,11,12]],tf.float64)
z1=tf.argmin(a,axis=0)
z2=tf.argmin(a,axis=1)

z1==>[0 0 0 0]
z2==>[0 0 0]
1.109 tf.argmax(input, axis=None, name=None, dimension=None)

功能：返回沿axis维度最大值的下标。
输入：
例：
a = tf.constant([[1,2,3,4], [5,6,7,8], [9,10,11,12]],tf.float64)
z1=tf.argmin(a,axis=0)
z2=tf.argmax(a,axis=1)

z1==>[2 2 2 2]
z2==>[3 3 3]
1.110 tf.setdiff1d(x, y, index_dtype=tf.int32, name=None)

功能：返回在x里不在y里的元素值和下标，
输入：
例：
a = tf.constant([1,2,3,4])
b=tf.constant([1,4])
out,idx=tf.setdiff1d(a,b)

out==>[2 3]
idx==>[1 2]
1.111 tf.where(condition, x=None, y=None, name=None)

功能：若x,y都为None，返回condition值为True的坐标;
    若x,y都不为None，返回condition值为True的坐标在x内的值，condition值为False的坐标在y内的值
输入：condition:bool类型的tensor;
例：
a=tf.constant([True,False,False,True])
x=tf.constant([1,2,3,4])
y=tf.constant([5,6,7,8])
z=tf.where(a)
z2=tf.where(a,x,y)

z==>[[0]
     [3]]
z2==>[ 1 6 7 4]
1.112 tf.unique(x, out_idx=None, name=None)

功能：罗列非重复元素及其编号。
输入：
例：
a = tf.constant([1,1,2,4,4,4,7,8,9,1])
y,idx=tf.unique(a)

y==>[1 2 4 7 8 9]
idx==>[0 0 1 2 2 2 3 4 5 0]
1.113 tf.edit_distance(hypothesis, truth, normalize=True, name='edit_distance')

功能：计算Levenshtein距离。
输入：hypothesis:'SparseTensor';
     truth:'SparseTensor'.
例：
hypothesis = tf.SparseTensor(
    [[0, 0, 0],
     [1, 0, 0]],
    ["a", "b"],
    (2, 1, 1))
truth = tf.SparseTensor(
    [[0, 1, 0],
     [1, 0, 0],
     [1, 0, 1],
     [1, 1, 0]],
    ["a", "b", "c", "a"],
    (2, 2, 2))
z=tf.edit_distance(hypothesis,truth)

z==>[[inf 1.]
     [0.5 1.]]
1.114 tf.invert_permutation(x, name=None)

功能：转换坐标与值。y(i)=x(i)的坐标   (for i in range (lens(x))。
输入：
例：
a=tf.constant([3,4,0,2,1])
z=tf.invert_permutation(a)

z==>[2 4 3 0 1]




