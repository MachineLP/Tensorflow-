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
from tensorflow.python.training import training
from tensorflow.contrib.framework import list_variables
from tensorflow.contrib.framework import load_variable
import json
#import models

tf.logging.set_verbosity(tf.logging.INFO)
plt.ioff()
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


def cnn_model_fn(features, labels, mode, num_classes=20):
    # Input Layer
    input_layer = tf.reshape(features["x"], [-1, 256, 256, 3])
    valid = features["w"]

    # Do data augmentation here !
    if mode == tf.estimator.ModeKeys.PREDICT:
        final_input = tf.map_fn(lambda im_tf: tf.image.central_crop(im_tf, float(0.875)), input_layer)

    if mode == tf.estimator.ModeKeys.TRAIN:
        final_input = tf.map_fn(lambda im_tf: tf.image.random_flip_left_right(im_tf), input_layer)
        final_input = tf.map_fn(lambda im_tf: tf.random_crop(im_tf, size=[224,224,3]), final_input)

    # AlexNet archirecture
    conv1 = tf.layers.conv2d(inputs=final_input, filters=96, kernel_size=[11, 11], strides=4, padding="valid", activation=tf.nn.relu, use_bias=True, trainable=True, bias_initializer=tf.zeros_initializer(), kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[3, 3], strides=2)
    conv2 = tf.layers.conv2d(inputs=pool1, filters=256, kernel_size=[5, 5], strides=1, padding="same", activation=tf.nn.relu, use_bias=True, trainable=True, bias_initializer=tf.zeros_initializer(), kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[3, 3], strides=2)
    conv3 = tf.layers.conv2d(inputs=pool2, filters=384, kernel_size=[3, 3], strides=1, padding="same", activation=tf.nn.relu, use_bias=True, trainable=True, bias_initializer=tf.zeros_initializer(), kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
    conv4 = tf.layers.conv2d(inputs=conv3, filters=384, kernel_size=[3, 3], strides=1, padding="same", activation=tf.nn.relu, use_bias=True, trainable=True, bias_initializer=tf.zeros_initializer(), kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
    conv5 = tf.layers.conv2d(inputs=conv4, filters=256, kernel_size=[3, 3], strides=1, padding="same", activation=tf.nn.relu, use_bias=True, trainable=True, bias_initializer=tf.zeros_initializer(), kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
    pool3 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[3, 3], strides=2)
    pool3_flat = tf.reshape(pool3, [-1, 5 * 5 * 256]) # Check this once !
    dense1 = tf.layers.dense(inputs=pool3_flat, units=4096, activation=tf.nn.relu, use_bias=True, trainable=True, bias_initializer=tf.zeros_initializer(), kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
    dropout1 = tf.layers.dropout(inputs=dense1, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)
    dense2 = tf.layers.dense(inputs=dropout1, units=4096, activation=tf.nn.relu, use_bias=True, trainable=True, bias_initializer=tf.zeros_initializer(), kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
    dropout2 = tf.layers.dropout(inputs=dense2, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)
    logits = tf.layers.dense(inputs=dropout2, units=num_classes)
    predictions = {"probabilities": tf.sigmoid(logits, name="sigmoid_tensor")}
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.identity(tf.losses.sigmoid_cross_entropy(labels, logits=logits), name='loss')

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        tf.summary.scalar('Train Loss', loss)
        lr = tf.train.exponential_decay(learning_rate=0.001,global_step=tf.train.get_global_step(),decay_steps=10000,decay_rate=0.5, staircase=True)
        tf.summary.scalar('Learning rate', lr)
        optimizer = tf.train.MomentumOptimizer(
                    learning_rate=lr,
                    momentum=0.9)
        train_op = optimizer.minimize(loss=loss,global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

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
    im_names = []

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
    return images, labels, weights

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
    
    train_data, train_labels, train_weights = load_pascal(args.data_dir, split='trainval')
    eval_data, eval_labels, eval_weights = load_pascal(args.data_dir, split='test')

    pascal_classifier = tf.estimator.Estimator(
        model_fn=partial(cnn_model_fn,
        num_classes=eval_labels.shape[1]),
        model_dir="./alexnet_models/")
    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data, "w": eval_weights},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
  
    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data, "w": train_weights},
        y=train_labels,
        batch_size=BATCH_SIZE,
        num_epochs=None,
        shuffle=True)
    for given_iter in range(0,5):
        if given_iter == 0:
            pascal_classifier.train(input_fn=train_input_fn, steps=1)
            print([name for name, _ in list_variables(pascal_classifier.model_dir)])
            weights = load_variable(pascal_classifier.model_dir, 'conv2d/kernel')
            f, axrr = plt.subplots(16,6)
            for i in range(0,96):
                im = Image.fromarray((weights[:,:,:,i]*255).astype('uint8')).resize((50,50)).convert('LA')
                axrr[int(i/6)][int(i%6)].axis('off')
                axrr[int(i/6)][int(i%6)].imshow(im)
            plt.axis('off')
            f.savefig('conv2d_'+str(given_iter)+'.png')
        elif given_iter == 2 or given_iter == 1:
            pascal_classifier.train(input_fn=train_input_fn, steps=10000)
            weights = load_variable(pascal_classifier.model_dir, 'conv2d/kernel')
            f, axarr = plt.subplots(16,6)
            for i in range(0,96):
                im = Image.fromarray((weights[:,:,:,i]*255).astype('uint8')).resize((50,50)).convert('LA')
                axrr[int(i/6)][int(i%6)].axis('off')
                axrr[int(i/6)][int(i%6)].imshow(im)
            plt.axis('off')
            f.savefig('conv2d_'+str(given_iter)+'.png')

        else:
            pascal_classifier.train(input_fn=train_input_fn, steps=10000)
    weights = load_variable(pascal_classifier.model_dir, 'conv2d/kernel')
    f, axarr = plt.subplots(16,6)
    for i in range(0,96):
        im = Image.fromarray((weights[:,:,:,i]*255).astype('uint8')).resize((50,50)).convert('LA')
        axrr[int(i/6)][int(i%6)].axis('off')
        axrr[int(i/6)][int(i%6)].imshow(im)
    plt.axis('off')
    f.savefig('conv2d_final.png')
    
if __name__ == "__main__":
    main()
