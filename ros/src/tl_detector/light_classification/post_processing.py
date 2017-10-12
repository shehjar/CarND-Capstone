# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 15:49:21 2017

@author: kauls
"""

import os,cv2
import numpy as np
import matplotlib.pyplot as plt
import yaml
import glob
import tensorflow as tf
from preprocessing import normalizing


#saver = tf.train.Saver()
curdir = os.getcwd()
pardir = os.path.abspath(os.path.join(curdir, os.pardir))
datadir = os.path.join(pardir,'pictures')
file_list = glob.glob(os.path.join(datadir,'png_files//*png'))
index = np.random.randint(len(file_list)-1)
test_image_file = file_list[index]
# read image in the format of RGB
test_img = cv2.cvtColor(cv2.imread(test_image_file), cv2.COLOR_BGR2RGB)
# Resize the image to 64x64 pixels
resize_test_img = cv2.resize(test_img, dsize=(64,64), interpolation=cv2.INTER_AREA)
#Normalize the image
test_image = normalizing(resize_test_img)

restore_file = './model/model.ckpt.meta'
saver = tf.train.import_meta_graph(restore_file)

with tf.Session() as sess:
    #saver.restore(sess, tf.train.latest_checkpoint('.'))
    saver.restore(sess, tf.train.latest_checkpoint('./model'))
    pred = sess.graph.get_tensor_by_name('pred:0')
    input_img = sess.graph.get_tensor_by_name('input_image:0')
    prob = sess.graph.get_tensor_by_name('prob:0')
    
    prediction = sess.run(pred, feed_dict={input_img:[test_image],prob:1})
    
    if prediction[0] == 0:
        print('The classifier has classified this as red!')
    else:
        print('The classifier has classified this as green!')
#    test_accuracy, test_cost = evaluate(X_test, y_test)
#    y_pred = prediction(X_test,y_test)
#    #print(y_pred)
#    print("Test Accuracy = {:.3f}".format(test_accuracy))
#    print("Test Cost = {:.3f}".format(test_cost))
    
# plot the image
plt.imshow(test_img)