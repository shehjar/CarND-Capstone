# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 07:56:56 2017

@author: kauls
"""

import os, time
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.contrib.layers import flatten
import preprocessing

def LeNet5(x, num_classes, keep_prob, mu=0, sigma=0.01):
    # Layer 1: Convolution: filter size 32
    conv1 = tf.layers.conv2d(x, 32, 7, padding='valid', activation=tf.nn.relu, 
                             kernel_initializer=tf.truncated_normal_initializer(stddev=sigma),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    # Pooling (2,2)
    pool1 = tf.layers.max_pooling2d(conv1, pool_size=[2, 2], strides=2)
    
    # Layer 2: Convolutional. Output = 10x10x64.
    conv2 = tf.layers.conv2d(pool1, 64, 3, padding='valid', activation=tf.nn.relu, 
                             kernel_initializer=tf.truncated_normal_initializer(stddev=sigma),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    # Maxpooling: (2,2)
    pool2 = tf.layers.max_pooling2d(conv2, pool_size=[2,2], strides=2)
    
    # Layer 3: Convolutional. filter size:128
    conv3 = tf.layers.conv2d(pool2, 128, 3, padding='valid', activation=tf.nn.relu,
                             kernel_initializer=tf.truncated_normal_initializer(stddev=sigma),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    # Pooling (2,2)
    pool3 = tf.layers.max_pooling2d(conv3, pool_size=[2,2], strides=2)
    
    # Flatten. Input = 5x5x64. Output = 1600
    flat = flatten(pool3)
    
    # Layer 3: Fully Connected. Input = 1600. Output = 400.
#    fc1_W = tf.Variable(tf.truncated_normal([10816,400], mean= mu, stddev= sigma))
#    fc1_b = tf.Variable(tf.zeros(400))
#    fc1 = tf.add(tf.matmul(flat,fc1_W),fc1_b)
    fc1 = tf.layers.dense(flat, 256, activation=tf.nn.relu,kernel_initializer=tf.truncated_normal_initializer(stddev=sigma))
    fc1 = tf.nn.dropout(fc1,keep_prob)
    
    # Layer 4: Fully Connected. Input = 400. Output = 100.
    #fc2_W = tf.Variable(tf.truncated_normal([400,100], mean= mu, stddev= sigma))
    #fc2_b = tf.Variable(tf.zeros(100))
    #fc2 = tf.add(tf.matmul(fc1,fc2_W),fc2_b)
    fc2 = tf.layers.dense(fc1, 128, activation=tf.nn.relu,kernel_initializer=tf.truncated_normal_initializer(stddev=sigma))
    fc2 = tf.nn.dropout(fc2, keep_prob)
    
    # Layer 5: Fully Connected. Input = 100. Output = 43.
    #fc3_W = tf.Variable(tf.truncated_normal([100,num_classes], mean= mu, stddev= sigma))
    #fc3_b = tf.Variable(tf.zeros(num_classes))
    #fc3 = tf.add(tf.matmul(fc2,fc3_W),fc3_b)
    fc3 = tf.layers.dense(fc2, num_classes,kernel_initializer=tf.truncated_normal_initializer(stddev=sigma))
    # Activation
    #logits = tf.nn.softmax(fc3)
    logits = fc3
    return logits

def optimize(logits, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    #logits = tf.reshape(nn_last_layer, (-1, num_classes))
    #labels = tf.reshape(correct_label, (-1, num_classes))
    #pred = tf.argmax(tf.nn.softmax(logits),axis=1)
    one_hot_y = tf.one_hot(correct_label, num_classes)
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y))
    opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = opt.minimize(cross_entropy_loss)
    
    return train_op, cross_entropy_loss

def run():
    num_classes = 2
    image_shape=(64,64)
    curdir = os.getcwd()
    pardir = os.path.abspath(os.path.join(curdir, os.pardir))
    datadir = os.path.join(pardir,'pictures')
    runsdir = os.path.join(curdir, 'runs')
    
    epochs = 20
    batch_size = 128
    
    # Get data
    data = preprocessing.read_data(datadir)
    train_data, test_data = preprocessing.test_train_split(data, 0.2)

    input_image = tf.placeholder(tf.float32,(None, image_shape[0], image_shape[1], 3))
    y_label = tf.placeholder(tf.int64, (None))
    prob = tf.placeholder(tf.float32)
    learning_rate_ph = tf.placeholder("float")
    logits = LeNet5(input_image, num_classes, prob)
    train_op, cross_entropy_loss = optimize(logits, y_label, learning_rate_ph, num_classes)
    pred_class = tf.argmax(tf.nn.softmax(logits), axis=1)
    correct_prediction = tf.equal(pred_class, y_label)
    float_cast_pred = tf.cast(correct_prediction,tf.float32)
    accuracy = tf.reduce_mean(float_cast_pred)
    #saver = tf.train.Saver()
    
    with tf.Session() as sess:
        # Get train data generator
        get_batches_fn = preprocessing.gen_batch_function(data, image_shape)
        sess.run(tf.global_variables_initializer())
        print("Training...")
        print()
        for i in range(epochs):
            train_loss = 0
            train_acc = 0
            samples = 0
            time_start = time.time()
            for images, labels in get_batches_fn(batch_size):
                _, loss, acc, pred_y, act_y, float_pred = sess.run([train_op, cross_entropy_loss, accuracy, pred_class, y_label, float_cast_pred], 
                                        feed_dict={input_image: images, y_label: labels, prob: 0.5, learning_rate_ph:1e-3})
                #print('Images shape:',images.shape)
                #print('pred_y', pred_y)
                #print('act_y', act_y)
                #print('float_cast_pred:', float_pred)
                train_loss += loss
                train_acc += acc
                samples += 1
                #print('loss:', loss, 'train_loss:',train_loss)
                #print('acc:', acc, 'train_acc:', train_acc)
                
            total_time = time.time() - time_start
            print("EPOCH {} ...".format(i+1))
            print("Loss = {}".format(train_loss/samples))
            print("Training accuracy = {}".format(train_acc/samples))
            print("Time = {} mins".format(total_time/60))
            print()
        # Test accuracy
        test_images, test_labels = preprocessing.gen_test_data(test_data, image_shape)
        loss, acc = sess.run([cross_entropy_loss, accuracy],
                             feed_dict={input_image:test_images, y_label:test_labels, prob:1})
        print("Test loss = {}".format(loss))
        print("Test accuracy = {}".format(acc))
        saver = tf.train.Saver()
        saver.save(sess, './model/model.ckpt')
        print('model saved!')
        
if __name__ == '__main__':
    run()