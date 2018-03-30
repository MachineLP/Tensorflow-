from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import sys
import numpy as np
import tensorflow as tf
import argparse
import os.path as osp
from PIL import Image
from functools import partial
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from eval import compute_map
#import models
plt.ioff()
tf.logging.set_verbosity(tf.logging.INFO)

CLASS_NAMES = [
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor',
]

weight_vgg_map = {
      'vgg_16/conv1/conv1_1/biases': 'conv1_1/bias',
            'vgg_16/conv1/conv1_1/weights': 'conv1_1/kernel',
            'vgg_16/conv1/conv1_2/biases': 'conv1_2/bias',
            'vgg_16/conv1/conv1_2/weights': 'conv1_2/kernel',
            'vgg_16/conv2/conv2_1/biases': 'conv2_1/bias',
            'vgg_16/conv2/conv2_1/weights': 'conv2_1/kernel',
            'vgg_16/conv2/conv2_2/biases': 'conv2_2/bias',
            'vgg_16/conv2/conv2_2/weights': 'conv2_2/kernel',
            'vgg_16/conv3/conv3_1/biases': 'conv3_1/bias',
            'vgg_16/conv3/conv3_1/weights': 'conv3_1/kernel',
            'vgg_16/conv3/conv3_2/biases': 'conv3_2/bias',
            'vgg_16/conv3/conv3_2/weights': 'conv3_2/kernel',
            'vgg_16/conv3/conv3_3/biases': 'conv3_3/bias',
            'vgg_16/conv3/conv3_3/weights': 'conv3_3/kernel',
            'vgg_16/conv4/conv4_1/biases': 'conv4_1/bias',
            'vgg_16/conv4/conv4_1/weights': 'conv4_1/kernel',
            'vgg_16/conv4/conv4_2/biases': 'conv4_2/bias',
            'vgg_16/conv4/conv4_2/weights': 'conv4_2/kernel',
            'vgg_16/conv4/conv4_3/biases': 'conv4_3/bias',
            'vgg_16/conv4/conv4_3/weights': 'conv4_3/kernel',
            'vgg_16/conv5/conv5_1/biases': 'conv5_1/bias',
            'vgg_16/conv5/conv5_1/weights': 'conv5_1/kernel',
            'vgg_16/conv5/conv5_2/biases': 'conv5_2/bias',
            'vgg_16/conv5/conv5_2/weights': 'conv5_2/kernel',
            'vgg_16/conv5/conv5_3/biases': 'conv5_3/bias',
            'vgg_16/conv5/conv5_3/weights': 'conv5_3/kernel',
            'vgg_16/fc6/biases': 'fc6/bias',
            'vgg_16/fc6/weights': 'fc6/kernel',
            'vgg_16/fc7/biases': 'fc7/bias',
            'vgg_16/fc7/weights': 'fc7/kernel'}

class RestoreHook(tf.train.SessionRunHook):
    def begin(self):
        tf.contrib.framework.init_from_checkpoint('./vgg_pretrained/vgg_16.ckpt', weight_vgg_map)

def cnn_model_fn(features, labels, mode, num_classes=20):

    # Input Layer
    input_layer = tf.reshape(features["x"], [-1, 256, 256, 3])
    valid = features["w"]
    layer = 'fc7'

    # Do data augmentation here !
    if mode == tf.estimator.ModeKeys.PREDICT:
        final_input = tf.map_fn(lambda im_tf: tf.image.central_crop(im_tf, float(0.875)), input_layer)

    # Block - 1  
    conv1 = tf.layers.conv2d(inputs=final_input, filters=64, kernel_size=[3,3], padding="same", activation=tf.nn.relu, use_bias=True, trainable=True, bias_initializer=tf.zeros_initializer(), kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv1_1')
    conv2 = tf.layers.conv2d(inputs=conv1, filters=64, kernel_size=[3,3], padding="same", activation=tf.nn.relu, use_bias=True, trainable=True, bias_initializer=tf.zeros_initializer(), kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv1_2')
    pool1 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Block - 2
    conv3 = tf.layers.conv2d(inputs=pool1, filters=128, kernel_size=[3,3], padding="same", activation=tf.nn.relu, use_bias=True, trainable=True, bias_initializer=tf.zeros_initializer(), kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv2_1')
    conv4 = tf.layers.conv2d(inputs=conv3, filters=128, kernel_size=[3,3], padding="same", activation=tf.nn.relu, use_bias=True, trainable=True, bias_initializer=tf.zeros_initializer(), kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv2_2')
    pool2 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)

    # Block - 3
    conv5 = tf.layers.conv2d(inputs=pool2, filters=256, kernel_size=[3,3], padding="same", activation=tf.nn.relu, use_bias=True, trainable=True, bias_initializer=tf.zeros_initializer(), kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv3_1')
    conv6 = tf.layers.conv2d(inputs=conv5, filters=256, kernel_size=[3,3], padding="same", activation=tf.nn.relu, use_bias=True, trainable=True, bias_initializer=tf.zeros_initializer(), kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv3_2')
    conv7 = tf.layers.conv2d(inputs=conv6, filters=256, kernel_size=[3,3], padding="same", activation=tf.nn.relu, use_bias=True, trainable=True, bias_initializer=tf.zeros_initializer(), kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv3_3')
    pool3 = tf.layers.max_pooling2d(inputs=conv7, pool_size=[2, 2], strides=2)

    # Block - 4
    conv8 = tf.layers.conv2d(inputs=pool3, filters=512, kernel_size=[3,3], padding="same", activation=tf.nn.relu, use_bias=True, trainable=True, bias_initializer=tf.zeros_initializer(), kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv4_1')
    conv9 = tf.layers.conv2d(inputs=conv8, filters=512, kernel_size=[3,3], padding="same", activation=tf.nn.relu, use_bias=True, trainable=True, bias_initializer=tf.zeros_initializer(), kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv4_2')
    conv10 = tf.layers.conv2d(inputs=conv9, filters=512, kernel_size=[3,3], padding="same", activation=tf.nn.relu, use_bias=True, trainable=True, bias_initializer=tf.zeros_initializer(), kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv4_3')
    pool4 = tf.layers.max_pooling2d(inputs=conv10, pool_size=[2, 2], strides=2)

    # Block - 5 
    conv11 = tf.layers.conv2d(inputs=pool4, filters=512, kernel_size=[3,3], padding="same", activation=tf.nn.relu, use_bias=True, trainable=True, bias_initializer=tf.zeros_initializer(), kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv5_1')
    conv12 = tf.layers.conv2d(inputs=conv11, filters=512, kernel_size=[3,3], padding="same", activation=tf.nn.relu, use_bias=True, trainable=True, bias_initializer=tf.zeros_initializer(), kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv5_2')
    conv13 = tf.layers.conv2d(inputs=conv12, filters=512, kernel_size=[3,3], padding="same", activation=tf.nn.relu, use_bias=True, trainable=True, bias_initializer=tf.zeros_initializer(), kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv5_3')
    pool5 = tf.layers.max_pooling2d(inputs=conv13, pool_size=[2, 2], strides=2)

    # FC Layers
    dense1 = tf.layers.conv2d(inputs=pool5, filters=4096, kernel_size=[7,7], strides=1, activation=tf.nn.relu, use_bias=True, trainable=True, bias_initializer=tf.zeros_initializer(), kernel_initializer=tf.contrib.layers.xavier_initializer(), name='fc6', padding='valid')
    dropout1 = tf.layers.dropout(inputs=dense1, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)
    dense2 = tf.layers.conv2d(inputs=dropout1, filters=4096, kernel_size=[1,1], strides=1, activation=tf.nn.relu, use_bias=True, trainable=True, bias_initializer=tf.zeros_initializer(), kernel_initializer=tf.contrib.layers.xavier_initializer(), name='fc7', padding='valid')
    dropout2 = tf.layers.dropout(inputs=dense2, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    dropout2_flat = tf.reshape(dropout2, [-1, 4096])
    logits = tf.layers.dense(inputs=dropout2_flat, units=num_classes, name='fc8_new')
    predictions = {"probabilities": tf.sigmoid(logits, name="sigmoid_tensor")}
    if layer=='pool5':
        predictions["im_rep"] =  pool5
    elif layer=='fc7':
        predictions["im_rep"] =  dense2

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

def load_pascal(data_dir, split='train'):

    # Get the exact dirs for Images and annorations
    images_dir = data_dir + '/VOCdevkit/VOC2007/JPEGImages/'
    annt_dir = data_dir + '/VOCdevkit/VOC2007/ImageSets/Main/'

    # Get the list of all classes, this is a Global variable
    global CLASS_NAMES

    # For the given split, get the dimensionality of output matrices
    # Do this by opening annotation file for any class, say aeroplane
    num_images = len(open(annt_dir+CLASS_NAMES[0]+'_'+split+'.txt','r').read().splitlines())
    num_classes = len(CLASS_NAMES)

    # Initialize images, labels, weights matrices
    images = np.zeros((num_images, 256, 256, 3), dtype='float32')
    labels = np.zeros((num_images, num_classes), dtype='int')
    weights = np.zeros((num_images, num_classes), dtype='int')

    # First load all the images for given split and then check for labels. This is faster.
    list_all_images = [i.split(' ')[0]+'.jpg' for i in open(annt_dir+CLASS_NAMES[0]+'_'+split+'.txt','r').read().splitlines()] # all files are in jpg format.
    for given_im in range(0, len(list_all_images)):
        print('Loading Image: ' + str(given_im))
        im_file_path = images_dir + list_all_images[given_im]
        im = np.asarray(Image.open(im_file_path).resize((256,256)).convert('RGB'), dtype='float32') # Just in case some image in grayscale
        images[given_im,:,:,:] = im

    # Now itwrate through annotation files and generate labels and weights
    for given_cls in range(0, len(CLASS_NAMES)):
        cls_name = CLASS_NAMES[given_cls]
        ann_file_name = annt_dir + cls_name + '_' + split + '.txt'
        cls_data = [i for i in open(ann_file_name, 'r').read().splitlines()]
        cls_ann = np.zeros((len(cls_data), 1))
        for  i in range(0, len(cls_data)):
            list_i = cls_data[i].split(' ')
            if len(list_i) == 2:
                cls_ann[i] = int(list_i[1])
            elif len(list_i) == 3:
                cls_ann[i] = int(list_i[2])

        for given_im in range(0, len(list_all_images)):
            if cls_ann[given_im] == 1:
                labels[given_im][given_cls] = 1
                weights[given_im][given_cls] = 1
            elif cls_ann[given_im] == 0:
                labels[given_im][given_cls] = 1
                weights[given_im][given_cls] = 0
            elif cls_ann[given_im] == -1:
                labels[given_im][given_cls] = 0
                weights[given_im][given_cls] = 1
            else:
                print('Something wrong in annotations: ' + str(given_im) + ' | ' + str(cls_name))
                sys.exit(1)
    return images, labels, weights, list_all_images

def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a classifier in tensorflow!')
    parser.add_argument(
        'data_dir', type=str, default='data/VOC2007',
        help='Path to PASCAL data storage')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args


def _get_el(arr, i):
    try:
        return arr[i]
    except IndexError:
        return arr


def main():

    args = parse_args()
    BATCH_SIZE = 10

    eval_data, eval_labels, eval_weights, im_names = load_pascal(args.data_dir, split='test')

    pascal_classifier = tf.estimator.Estimator(
        model_fn=partial(cnn_model_fn,
        num_classes=eval_labels.shape[1]),
        model_dir="./vgg_finetune/")

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data, "w": eval_weights},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)

    pred = list(pascal_classifier.predict(input_fn=eval_input_fn))
    im_reps = np.stack([p['im_rep'] for p in pred])

    rep_dict = {}
    for given_im in range(0, len(im_names)):
        im_name = im_names[given_im]
        im_rep = im_reps[given_im]
        rep_dict[im_name] = im_rep.flatten()

    random_picks = np.random.randint(low=0, high=len(im_names), size=10)
    print(random_picks)
    for im_num in range(0, len(random_picks)):
        im_name = im_names[im_num]
        im_rep = rep_dict[im_name]
        comp_dist = np.zeros((len(im_names), 1), dtype='float')
        for comp_im in range(0, len(im_names)):
            comp_im_name = im_names[comp_im]
            comp_im_rep  = rep_dict[comp_im_name]
            comp_dist[comp_im] = np.linalg.norm(im_rep-comp_im_rep, ord=2)
        closes_ims = [str(args.data_dir+'/VOCdevkit/VOC2007/JPEGImages/'+str(im_names[i])) for i in np.argpartition(comp_dist[:,0], 6)[:6]]
        f, axarr = plt.subplots(1,7)
        print(closes_ims)
        im_org = Image.open(args.data_dir+'/VOCdevkit/VOC2007/JPEGImages/'+str(im_name)).resize((100,100))
        axarr[0].imshow(im_org)
        axarr[1].imshow(Image.open(closes_ims[0]).resize((100,100)))
        axarr[2].imshow(Image.open(closes_ims[1]).resize((100,100)))
        axarr[3].imshow(Image.open(closes_ims[2]).resize((100,100)))
        axarr[4].imshow(Image.open(closes_ims[3]).resize((100,100)))
        axarr[5].imshow(Image.open(closes_ims[4]).resize((100,100)))
        axarr[6].imshow(Image.open(closes_ims[5]).resize((100,100)))
        f.savefig('vgg_fc7'+str(im_num)+'.png')

    
if __name__ == "__main__":
    main()
