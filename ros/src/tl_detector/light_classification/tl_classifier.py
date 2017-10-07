import os
from styx_msgs.msg import TrafficLight
import rospy
import numpy as np
import cv2
import tensorflow as tf
from train_model import LeNet5

#set the color thresh in HSV space
#for red, it can be in two ranges 1 and 2
lower_red_1 = np.array([0,200,200])
upper_red_1 = np.array([10,255,255])

lower_red_2 = np.array([170,200,200])
upper_red_2 = np.array([180,200,200])

RED_LIGHT_THRESHOLD = 20
#for yellow, only one range
lower_yellow = np.array([25,200,200])
upper_yellow = np.array([35,255,255])

YELLOW_LIGHT_THRESHOLD = 20

#for green, only one range
lower_green = np.array([60,200,200])
upper_green = np.array([70,255,255])
GREEN_LIGHT_THRESHOLD = 20

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        self.sess = tf.Session()
        classifier_dir = os.path.dirname(__file__)
        model_dir = os.path.join(classifier_dir, 'model')
        meta_path = os.path.join(model_dir, 'model.ckpt.meta')
        rospy.loginfo('Loading TL classifier meta graph from %s', meta_path)
        saver = tf.train.import_meta_graph(meta_path)
        rospy.loginfo('Loading TL classifier checkpoint from %s', model_dir)
        saver.restore(self.sess, tf.train.latest_checkpoint(model_dir))

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction

        if 0:
            # Use the TF classifier
            input_image = tf.placeholder(tf.float32, (None, 64, 64, 3))
            prob = tf.placeholder(tf.float32)
            logits = LeNet5(input_image, 2, prob)
            self.sess.run(logits, feed_dict={input_image: image, prob: 1})

        #change color to hsv space
        hsv= cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        #get mask
        red_mask_1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
        red_mask_2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        if ((np.sum(red_mask_1)+np.sum(red_mask_2)) > RED_LIGHT_THRESHOLD):
	    #detect red 
            rospy.loginfo("red light ahead")
	    return TrafficLight.RED
        elif (np.sum(yellow_mask)>YELLOW_LIGHT_THRESHOLD):
            #detect yellow
            rospy.loginfo("yellow light ahead")
	    return TrafficLight.YELLOW
        elif (np.sum(green_mask)>GREEN_LIGHT_THRESHOLD):
            #detect green
            rospy.loginfo("green light ahead")
	    return TrafficLight.GREEN
        return TrafficLight.UNKNOWN
