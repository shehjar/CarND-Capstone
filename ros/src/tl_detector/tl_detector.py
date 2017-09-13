#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import math
import numpy as np
import yaml


STATE_COUNT_THRESHOLD = 3

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

class TrafficLightInfo:
    def __init__(self, x, y, state):
        self.x = x
        self.y = y
        self.state = state


class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')
	self.ImageID = 1
        self.pose = None
        self.waypoints_msg = None
        self.camera_image = None
        self.traffic_light_labels = []

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb, queue_size=1)
        #sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights helps you acquire an accurate ground truth data source for the traffic light
        classifier, providing the location and current color state of all traffic lights in the
        simulator. This state can be used to generate classified images or subbed into your solution to
        help you work on another single component of the node. This topic won't be available when
        testing your solution in real life so don't rely on it in the final submission.
        '''
        self.traffic_lights_sub = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb, queue_size=1)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb, queue_size=1)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', TrafficLight, queue_size=1)

        self.bgr8_pub = rospy.Publisher('/image_color_bgr8', Image, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.saved_tl_info = TrafficLightInfo(0, 0, TrafficLight.UNKNOWN)
        self.state_count = 0
        self.traffic_light_labels = None

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, msg):
        self.waypoints_msg = msg

    def traffic_cb(self, msg):
        self.traffic_light_labels = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
        #
        # process_traffic_lights() - use detection
        # process_traffic_lights_test() - use info from /vehicle/traffic_lights topic
        #
        self.process_traffic_lights()
        tl_info = self.process_traffic_lights_test()
        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.saved_tl_info.state != tl_info.state:
            self.state_count = 0
            self.saved_tl_info.state = tl_info.state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.saved_tl_info = tl_info
        else:
            tl_info = self.saved_tl_info
        self.state_count += 1

	# publish TL info
	tl = TrafficLight()
	tl.state = tl_info.state
        tl.pose.pose.position.x = tl_info.x
        tl.pose.pose.position.y = tl_info.y
        tl.pose.pose.position.z = 0
        self.upcoming_red_light_pub.publish(tl)


    def get_closest_traffic_light(self, pose):
        """Identifies the closest traffic light to the given position
        Args:
            pose (Pose): position of car

        Returns:
            int: index of the closest traffic light

        """
        #TODO implement
        q = pose.orientation
        _, _, yaw = tf.transformations.euler_from_quaternion((q.x, q.y, q.z, q.w))
        limit_angle = math.pi / 2
        max_distance = 100
        min_distance = float("inf")
        traffic_light_idx = -1
        for idx, (x, y) in enumerate(self.config['light_positions']):
            heading = math.atan2((y - pose.position.y), (x - pose.position.x))
            if abs(yaw - heading) < limit_angle:
                distance = math.sqrt((x - pose.position.x) ** 2 + (y - pose.position.y) ** 2)
                # ignore traffic light if it is too far
                if distance > max_distance:
                    break

                if distance <= min_distance:
                    traffic_light_idx = idx
                    min_distance = distance
                else:
                    break

        #rospy.loginfo("traffic_light_idx = %d", traffic_light_idx)
        return traffic_light_idx

    def project_to_image_plane(self, point_in_world):
        """Project point from 3D world coordinates to 2D camera image location

        Args:
            point_in_world (Point): 3D location of a point in the world

        Returns:
            x (int): x coordinate of target point in image
            y (int): y coordinate of target point in image

        """

        fx = self.config['camera_info']['focal_length_x']
        fy = self.config['camera_info']['focal_length_y']
        image_width = self.config['camera_info']['image_width']
        image_height = self.config['camera_info']['image_height']

        # get transform between pose of camera and world frame
        trans = None
        try:
            now = rospy.Time.now()
            # Compensate for delay
            past = now - rospy.Duration(.5)
            self.listener.waitForTransform("/base_link", "/world", now, rospy.Duration(1.0))
            (trans, rot) = self.listener.lookupTransform("/base_link", "/world", past)
            t = tf.transformations.translation_matrix(trans)
            r = tf.transformations.quaternion_matrix(rot)
            # Car camera points slightly up
            camera_angle = 15.0
            camera_angle = camera_angle * math.pi / 180.0
            r_camera = tf.transformations.euler_matrix(0, camera_angle, 0)
            # Combine all matrices
            m = tf.transformations.concatenate_matrices(t, r)
            p = np.append(point_in_world, 1.0)
            tp = m.dot(p)
            rospy.loginfo("x = %f, y = %f, z = %f", tp[0], tp[1], tp[2])
            # Project
            x = fx * tp[1] / tp[0]
            y = fy * tp[2] / tp[0]
            # Map 2D point to image coordinates
            #x = int((0.5 - x) * image_width)
            #y = int((0.5 - y) * image_height)
            x = int(image_width / 2.0 + x * 800.0)
            y = int(image_height - y * 1000.0)
            distance = tp[0]
            rospy.loginfo("x = %d, y = %d", x, y)

        except (tf.Exception, tf.LookupException, tf.ConnectivityException):
            x = 0
            y = 0
            distance = 0
            rospy.logerr("Failed to find camera to map transform")

        #TODO Use tranform and rotation to calculate 2D position of light in image

        return (x, y, distance)

    def get_light_state(self, light_location):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if(not self.has_image):
            self.prev_light_loc = None
            return False

        encoding = 'rgb8'
        self.camera_image.encoding = encoding
        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, encoding)
        x, y, distance = self.project_to_image_plane(light_location)
        box_size = int(30.0 / distance * 100.0) if distance != 0 else 0
        cv2.rectangle(cv_image, (x - box_size, y - box_size), (x + box_size, y + box_size), (255, 0, 0), 2)
        self.bgr8_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, encoding))

        #TODO use light location to zoom in on traffic light in image

        #Get classification
        return self.light_classifier.get_classification(cv_image)

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        light_positions = self.config['light_positions']
        if self.pose is None:
            return

        traffic_light_idx = self.get_closest_traffic_light(self.pose.pose)

        # Find the closest visible traffic light (if one exists)
        light_location = None
        if traffic_light_idx >= 0:
            #light_location = np.array(light_positions[traffic_light_idx] + [5.0])
            #rospy.loginfo('Light location = %s', light_location)
            item = self.traffic_light_labels[traffic_light_idx]
            light_location = np.array([item.pose.pose.position.x,
                                       item.pose.pose.position.y,
                                       item.pose.pose.position.z])
            x, y, distance = self.project_to_image_plane(light_location)
            rospy.loginfo('Light x, y = %f, %f', x, y)
            state = self.get_light_state(light_location)
            return TrafficLightInfo(light_location[0], light_location[1], state)

        return TrafficLightInfo(0, 0, TrafficLight.UNKNOWN)

    def process_traffic_lights_test(self):

        tl_info = TrafficLightInfo(0, 0, TrafficLight.UNKNOWN)
        if self.pose is None or self.traffic_light_labels is None:
            return tl_info

        pose = self.pose.pose
        q = pose.orientation
        _, _, yaw = tf.transformations.euler_from_quaternion((q.x, q.y, q.z, q.w))
        limit_angle = math.pi / 2
        max_distance = 100
        min_distance = float("inf")
        traffic_light_idx = -1
        for idx, label in enumerate(self.traffic_light_labels):
            position = label.pose.pose.position
            heading = math.atan2((position.y - pose.position.y), (position.x - pose.position.x))
            if abs(yaw - heading) < limit_angle:
                distance = math.sqrt((position.x - pose.position.x) ** 2 + (position.y - pose.position.y) ** 2)
                # ignore traffic light if it is too far
                if distance > max_distance:
                    break

                if distance <= min_distance:
                    traffic_light_idx = idx
                    min_distance = distance
                else:
                    break

        if traffic_light_idx >= 0:
            traffic_light_item = self.traffic_light_labels[traffic_light_idx]
            tl_info.x = traffic_light_item.pose.pose.position.x
            tl_info.y = traffic_light_item.pose.pose.position.y
            tl_info.state = traffic_light_item.state

        return tl_info


if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
