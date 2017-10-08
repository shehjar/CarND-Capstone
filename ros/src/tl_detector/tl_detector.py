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
        self.traffic_light_labels = None
        self.saved_tl_info = TrafficLightInfo(0, 0, TrafficLight.UNKNOWN)
        self.state_count = 0
        self.light_classifier = TLClassifier()

        self.is_simulator = rospy.get_param('/simulator', 0)

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb, queue_size=1)
        #sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        self.traffic_lights_sub = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb, queue_size=1)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb, queue_size=1)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', TrafficLight, queue_size=1)

        self.bgr8_pub = rospy.Publisher('/image_color_bgr8', Image, queue_size=1)

        self.bridge = CvBridge()
        self.listener = tf.TransformListener()


        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, msg):
        self.waypoints_msg = msg

    def traffic_cb(self, msg):
        self.traffic_light_labels = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
        #
        # process_traffic_lights() - use detection
        # process_traffic_lights_test() - use info from /vehicle/traffic_lights topic
        #
        tl_info = self.process_traffic_lights()
        #tl_info = self.process_traffic_lights_test()
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
        q = pose.orientation
        _, _, yaw = tf.transformations.euler_from_quaternion((q.x, q.y, q.z, q.w))
        limit_angle = math.pi / 2
        max_distance = 100
        min_distance = float("inf")
        traffic_light_idx = -1
        for idx, (x, y) in enumerate(self.config['stop_line_positions']):
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

        pose = self.pose.pose
        p = pose.position
        t = tf.transformations.translation_matrix((-p.x, -p.y, -p.z))
        q = pose.orientation
        r = tf.transformations.quaternion_matrix((q.x, q.y, q.z, -q.w))

        if self.is_simulator:
            # Car camera points slightly up
            camera_pitch_angle = math.radians(10)
            r_camera = tf.transformations.euler_matrix(0, camera_pitch_angle, 0)
            # Camera seems to be a bit off to the side
            t_camera = tf.transformations.translation_matrix((0, 0.5, 0))
            # Combine all matrices
            m = tf.transformations.concatenate_matrices(r_camera, t_camera, r, t)
            # Make coordinate homogenous
            p = np.append(point_in_world, 1.0)
            # Transform the world point to camera coordinates
            tp = m.dot(p)
            # Project point to image plane
            # Note: the "correction" multipliers are tweaked by hand
            x = 3.5 * tp[1] / tp[0]
            y = 3.5 * tp[2] / tp[0]
            # Map camera image point to pixel coordinates
            x = int((0.5 - x) * image_width)
            y = int((0.5 - y) * image_height)
            # X-coordinate is the distance to the TL
            distance = tp[0]
        else:
            # Carla camera points slight to right and up
            camera_yaw_angle = math.radians(1)
            camera_pitch_angle = math.radians(7)
            # Camera seems to be a bit off to the side also
            t_camera = tf.transformations.translation_matrix((0, -0.3, 0))
            r_camera = tf.transformations.euler_matrix(0, camera_pitch_angle,
                                                       camera_yaw_angle)
            # Combine all matrices
            m = tf.transformations.concatenate_matrices(r_camera, t_camera, r, t)
            # Make coordinate homogenous
            p = np.append(point_in_world, 1.0)
            # Transform the world point to camera coordinates
            tp = m.dot(p)
            # Project point to image plane
            # Note: the "correction" multipliers are tweaked by hand
            x = 1.75 * tp[1] / tp[0]
            y = 1.75 * tp[2] / tp[0]
            # Map camera image point to pixel coordinates
            x = int((1 - x) * image_width)
            y = int((1 - y) * image_height)
            # X-coordinate is the distance to the TL
            distance = tp[0]

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
        # Tweak light height to get the projection closer
        if self.is_simulator:
            light_location[2] += 1.0
            box_dim = 30.0
        else:
            light_location[2] -= 0
            box_dim = 15.0
        
        x, y, distance = self.project_to_image_plane(light_location)
        if x < 0 or x > cv_image.shape[1] or y < 0 or y > cv_image.shape[0]:
            return False
        box_size = int(box_dim / distance * 120.0) if distance != 0 else 0

        # publish bounding box for debugging
	# please do not use traffic light as the bounding box color
        # cv2.rectangle(cv_image,
        #               (x - box_size, y - box_size),
        #               (x + box_size, y + box_size),
        #               (255, 255, 255), 1)
        # self.bgr8_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, encoding))

        #TODO use light location to zoom in on traffic light in image
        cv_image_with_borders = cv2.copyMakeBorder(cv_image,
                                                   box_size, box_size, box_size, box_size,
                                                   cv2.BORDER_CONSTANT,
                                                   value=[0, 0, 0])
        x1, y1, x2, y2 = x, y, x + 2 * box_size, y + 2 * box_size
        tl_image = cv_image_with_borders[y1:y2, x1:x2]
        print(x, y, distance, tl_image.shape)
        self.bgr8_pub.publish(self.bridge.cv2_to_imgmsg(tl_image, encoding))
        #Get classification
        return self.light_classifier.get_classification(tl_image)

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        stop_line_positions = self.config['stop_line_positions']
        if self.pose is None:
            return TrafficLightInfo(0, 0, TrafficLight.UNKNOWN)

        traffic_light_idx = self.get_closest_traffic_light(self.pose.pose)

        # Find the closest visible traffic light (if one exists)
        light_location = None
        if traffic_light_idx >= 0:
            item = self.traffic_light_labels[traffic_light_idx]
            light_location = np.array([item.pose.pose.position.x,
                                       item.pose.pose.position.y,
                                       item.pose.pose.position.z])
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
