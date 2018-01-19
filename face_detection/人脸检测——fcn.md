在上一篇的基础上修改即可：[人脸检测——滑动窗口篇（训练和实现）](http://blog.csdn.net/u014365862/article/details/77816493)
！！！注意：这些是我的调试版本，最优版本不方便公开，但是自己可以查看论文，自行在此基础上修改，加深一些模型，加上回归框，要不fcn容易出现较大偏差。
fcn：
```
import tensorflow as tf  
import numpy as np  
import sys  
# from models import *  
from PIL import Image  
from PIL import ImageDraw  
from PIL import ImageFile  
from skimage.transform import pyramid_gaussian  
from skimage.transform import resize  
from matplotlib import pyplot  
ImageFile.LOAD_TRUNCATED_IMAGES = True  
import utils  
import cv2  
import pylab  
  
def fcn_12_detect(threshold, dropout=False, activation=tf.nn.relu):  
      
    imgs = tf.placeholder(tf.float32, [None, None, None, 3])  
    labels = tf.placeholder(tf.float32, [None, 2])  
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')  
    with tf.variable_scope('net_12'):  
        conv1,_ = utils.conv2d(x=imgs, n_output=16, k_w=3, k_h=3, d_w=1, d_h=1, name="conv1")  
        conv1 = activation(conv1)  
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME", name="pool1")  
        ip1,W1 = utils.conv2d(x=pool1, n_output=16, k_w=6, k_h=6, d_w=1, d_h=1, padding="VALID", name="ip1")  
        ip1 = activation(ip1)  
        if dropout:  
            ip1 = tf.nn.dropout(ip1, keep_prob)  
        ip2,W2 = utils.conv2d(x=ip1, n_output=2, k_w=1, k_h=1, d_w=1, d_h=1, name="ip2")  
  
        #pred = tf.nn.sigmoid(utils.flatten(ip2))  
        pred = tf.nn.sigmoid(ip2)  
          
        return {'imgs': imgs, 'keep_prob': keep_prob,'pred': pred, 'features': ip1}  
  
def fcn_24_detect(threshold, dropout=False, activation=tf.nn.relu):  
  
    imgs = tf.placeholder(tf.float32, [None, 24, 24, 3])  
    labels = tf.placeholder(tf.float32, [None, 2])  
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')  
      
    with tf.variable_scope('net_24'):  
        conv1, _ = utils.conv2d(x=imgs, n_output=64, k_w=5, k_h=5, d_w=1, d_h=1, name="conv1")  
        conv1 = activation(conv1)  
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME", name="pool1")  
        ip1, W1 = utils.conv2d(x=pool1, n_output=128, k_w=12, k_h=12, d_w=1, d_h=1, padding="VALID", name="ip1")  
        ip1 = activation(ip1)  
        concat = ip1  
        if dropout:  
            concat = tf.nn.dropout(concat, keep_prob)  
        ip2, W2 = utils.conv2d(x=concat, n_output=2, k_w=1, k_h=1, d_w=1, d_h=1, name="ip2")  
  
        pred = tf.nn.sigmoid(utils.flatten(ip2))  
        target = utils.flatten(labels)  
  
        regularizer = 8e-3 * (tf.nn.l2_loss(W1)+100*tf.nn.l2_loss(W2))  
  
        loss = tf.reduce_mean(tf.div(tf.add(-tf.reduce_sum(target * tf.log(pred + 1e-9),1), -tf.reduce_sum((1-target) * tf.log(1-pred + 1e-9),1)),2)) + regularizer  
        cost = tf.reduce_mean(loss)  
          
        predict = pred  
        max_idx_p = tf.argmax(predict, 1)    
        max_idx_l = tf.argmax(target, 1)    
        correct_pred = tf.equal(max_idx_p, max_idx_l)    
        acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))    
  
        thresholding_24 = tf.cast(tf.greater(pred, threshold), "float")  
        recall_24 = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(thresholding_24, tf.constant([1.0])), tf.equal(target, tf.constant([1.0]))), "float")) / tf.reduce_sum(target)  
  
  
        return { 'imgs': imgs, 'labels': labels,   
            'keep_prob': keep_prob, 'cost': cost, 'pred': pred, 'accuracy': acc, 'features': concat,  
            'recall': recall_24, 'thresholding': thresholding_24}  
  
def py_nms(dets, thresh, mode="Union"):  
    """ 
        greedily select boxes with high confidence 
        keep boxes overlap <= thresh 
        rule out overlap > thresh 
        :param dets: [[x1, y1, x2, y2 score]] 
        :param thresh: retain overlap <= thresh 
        :return: indexes to keep 
        """  
    if len(dets) == 0:  
        return []  
    x1 = dets[:, 0]  
    y1 = dets[:, 1]  
    x2 = dets[:, 2]  
    y2 = dets[:, 3]  
    scores = dets[:, 4]  
      
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)  
    order = scores.argsort()[::-1]  
      
    keep = []  
    while order.size > 0:  
        i = order[0]  
        keep.append(i)  
        xx1 = np.maximum(x1[i], x1[order[1:]])  
        yy1 = np.maximum(y1[i], y1[order[1:]])  
        xx2 = np.minimum(x2[i], x2[order[1:]])  
        yy2 = np.minimum(y2[i], y2[order[1:]])  
          
        w = np.maximum(0.0, xx2 - xx1 + 1)  
        h = np.maximum(0.0, yy2 - yy1 + 1)  
        inter = w * h  
        if mode == "Union":  
            ovr = inter / (areas[i] + areas[order[1:]] - inter)  
        elif mode == "Minimum":  
            ovr = inter / np.minimum(areas[i], areas[order[1:]])  
          
        inds = np.where(ovr <= thresh)[0]  
        order = order[inds + 1]  
      
    return dets[keep]  
  
def nms(boxes, threshold, method):  
    if boxes.size==0:  
        return np.empty((0,3))  
    x1 = boxes[:,0]  
    y1 = boxes[:,1]  
    x2 = boxes[:,2]  
    y2 = boxes[:,3]  
    s = boxes[:,4]  
    area = (x2-x1+1) * (y2-y1+1)  
    I = np.argsort(s)  
    pick = np.zeros_like(s, dtype=np.int16)  
    counter = 0  
    while I.size>0:  
        i = I[-1]  
        pick[counter] = i  
        counter += 1  
        idx = I[0:-1]  
        xx1 = np.maximum(x1[i], x1[idx])  
        yy1 = np.maximum(y1[i], y1[idx])  
        xx2 = np.minimum(x2[i], x2[idx])  
        yy2 = np.minimum(y2[i], y2[idx])  
        w = np.maximum(0.0, xx2-xx1+1)  
        h = np.maximum(0.0, yy2-yy1+1)  
        inter = w * h  
        if method is 'Min':  
            o = inter / np.minimum(area[i], area[idx])  
        else:  
            o = inter / (area[i] + area[idx] - inter)  
        I = I[np.where(o<=threshold)]  
    pick = pick[0:counter]  
    return pick  
  
  # 预处理变了要重新训练哦。
def image_preprocess(img):

    img = (img - 127.5)*0.0078125
    '''m = img.mean()
    s = img.std()
    min_s = 1.0/(np.sqrt(img.shape[0]*img.shape[1]*img.shape[2]))
    std = max(min_s, s)  
    img = (img-m)/std'''

    return img
  
def min_face(img, F, window_size, stride):  
    # img：输入图像，F：最小人脸大小， window_size：滑动窗，stride：滑动窗的步长。  
    h, w, _ = img.shape  
    w_re = int(float(w)*window_size/F)  
    h_re = int(float(h)*window_size/F)  
    if w_re<=window_size+stride or h_re<=window_size+stride:  
        print (None)  
    # 调整图片大小的时候注意参数，千万不要写反了  
    # 根据最小人脸缩放图片  
    img = cv2.resize(img, (w_re, h_re))  
    return img  
  
# 构建图像的金字塔，以便进行多尺度滑动窗口  
# image是输入图像，f为缩放的尺度， window_size最小尺度  
def pyramid(image, f, window_size):  
    w = image.shape[1]  
    h = image.shape[0]  
    img_ls = []  
    while( w > window_size and h > window_size):  
        img_ls.append(image)  
        w = int(w * f)  
        h = int(h * f)  
        image = cv2.resize(image, (w, h))  
    return img_ls  
  
# 选取map中大于人脸阀值的点，映射到原图片的窗口大小，默认map中的一个点对应输入图中的12*12的窗口，最后要根据缩放比例映射到原图。  
def generateBoundingBox(imap, scale, t):  
    # use heatmap to generate bounding boxes  
    stride=2  
    cellsize=12  
  
    imap = np.transpose(imap)  
    y, x = np.where(imap >= t)  
      
    score = imap[(y,x)]  
    bb = np.transpose(np.vstack([y,x]))  
    q1 = np.fix((stride*bb+1)/scale)  
    q2 = np.fix((stride*bb+cellsize-1+1)/scale)  
    boundingbox = np.hstack([q1, q2, np.expand_dims(score,1)])  
    return boundingbox  
  
def imresample(img, sz):  
    im_data = cv2.resize(img, (sz[1], sz[0]), interpolation=cv2.INTER_AREA) #pylint: disable=no-member  
    return im_data  
  
if __name__ == '__main__':  
  
    image = cv2.imread('images/11.jpg')  
    h,w,_ = image.shape  
    # 调参的参数  
    IMAGE_SIZE = 12  
    # 步长  
    stride = 2  
    # 最小人脸大小  
    F = 24  
    # 构建金字塔的比例  
    ff = 0.8  
    # 概率多大时判定为人脸？  
    p_12 = 0.8  
    p_24 = 0.8  
    # nms  
    overlapThresh_12 = 0.7
    # 是否启用net-24  
    net_24 = True  
    overlapThresh_24 = 0.3
    '''''--------------------------------------'''  
      
    net_12 = fcn_12_detect(0.0)  
    net_12_vars = [v for v in tf.trainable_variables() if v.name.startswith('net_12')]  
    saver_net_12 = tf.train.Saver(net_12_vars)  
      
    net_24 = fcn_24_detect(0.0)  
    net_24_vars = [v for v in tf.trainable_variables() if v.name.startswith('net_24')]  
    saver_net_24 = tf.train.Saver(net_24_vars)  
      
  
    sess = tf.Session()  
    sess.run(tf.initialize_all_variables())  
  
    saver_net_12.restore(sess, 'model/12-net/model_net_12-123246')  
    saver_net_24.restore(sess, 'model/24-net/model_net_24-161800')  
    # saver_cal_48.restore(sess, 'model/model_cal_48-10000')  
      
      
    # 需要检测的最小人脸  
    img =image  
    factor_count=0  
    total_boxes=np.empty((0,5))  
    points=[]  
    h=img.shape[0]  
    w=img.shape[1]  
    minl=np.amin([h, w])  
    m=12.0/F  
    minl=minl*m  
    # creat scale pyramid  
    scales=[]  
    factor=ff  
    while minl>=12:  
        scales += [m*np.power(factor, factor_count)]  
        minl = minl*factor  
        factor_count += 1  
  
    # first stage  
    for j in range(len(scales)):  
        scale=scales[j]  
        hs=int(np.ceil(h*scale))  
        ws=int(np.ceil(w*scale))  
        im_data = imresample(img, (hs, ws))  
        im_data = image_preprocess(im_data)  
        pred_cal_12 = sess.run(net_12['pred'], feed_dict={net_12['imgs']: [im_data]})  
        out = np.transpose(pred_cal_12, (0,2,1,3))  
        threshold_12 = p_12  
        boxes = generateBoundingBox(out[0,:,:,1].copy(), scale, threshold_12)  
        boxes = py_nms(boxes, overlapThresh_12, 'Union')  
        if boxes != []:  
            total_boxes = np.append(total_boxes, boxes, axis=0)  
          
      
    window_net = total_boxes  
    # 后面24-net，48-net  
  
    if window_net == []:  
        print "windows is None!"  
    if window_net != []:  
        print(window_net.shape)  
        for box in window_net:  
            #ImageDraw.Draw(image).rectangle((box[1], box[0], box[3], box[2]), outline = "red")  
            cv2.rectangle(image, (int(box[1]),int(box[0])), (int(box[3]),int(box[2])), (0, 255, 0), 2)  
    cv2.imwrite("images/face_img.jpg", image)  
    cv2.imshow("face detection", image)  
    cv2.waitKey(10000)  
    cv2.destroyAllWindows()  
      
  
    sess.close()  
```
检测结果：
![20170920084446842.jpg](http://upload-images.jianshu.io/upload_images/4618424-8f42f98cb2eaf477.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
