#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose, Point
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
from traffic_light_config import config
import math
import numpy as np
import yaml

STATE_COUNT_THRESHOLD = 3

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints_msg = None
        self.camera_image = None
        self.traffic_light_labels = []

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

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

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Point, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, msg):
        self.waypoints_msg = msg

    def traffic_cb(self, msg):
        self.traffic_light_labels = msg.lights
        self.traffic_lights_sub.unregister()

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        return
        self.has_image = True
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

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
        """
        for idx, (x, y) in enumerate(config.light_positions):
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
        """
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

        rospy.loginfo("traffic_light_idx = %d", traffic_light_idx)
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
            self.listener.waitForTransform("/base_link", "/world", now, rospy.Duration(1.0))
            (trans, rot) = self.listener.lookupTransform("/base_link", "/world", now)

        except (tf.Exception, tf.LookupException, tf.ConnectivityException):
            rospy.logerr("Failed to find camera to map transform")

        #TODO Use tranform and rotation to calculate 2D position of light in image
        x = 0
        y = 0
        """
        if trans is not None:
            relative = point_in_world + trans
            rospy.loginfo("px = %f, py = %f, pz = %f", point_in_world[0], point_in_world[1], point_in_world[2])
            rospy.loginfo("tx = %f, ty = %f, tz = %f", trans[0], trans[1], trans[2])
            rospy.loginfo("rx = %f, ry = %f, rz = %f", relative[0], relative[1], relative[2])
            q = np.zeros((4,), dtype=np.float64)
            q[1:] = relative
            point_in_camera = tf.transformations.quaternion_multiply(
                tf.transformations.quaternion_multiply(rot, q),
                tf.transformations.quaternion_conjugate(rot)
            )[:3]

            # TODO: convert with focal length
            image_x = point_in_camera[1]
            image_y = point_in_camera[2]
            image_z = point_in_camera[0]
            rospy.loginfo("x = %f, y = %f, z = %f", image_x, image_y, image_z)

            x = fx * image_x / image_z
            y = fy * image_y / image_z

            #rospy.loginfo("x = %f, y = %f", x, y)

        """
        return (x, y)

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

        self.camera_image.encoding = "rgb8"
        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        x, y = self.project_to_image_plane(light_location)

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
        rospy.loginfo("lx = %f, ly = %f, lz = %f", self.pose.pose.position.x, self.pose.pose.position.y, self.pose.pose.position.z)

        #TODO find the closest visible traffic light (if one exists)
        light_location = None
        if traffic_light_idx >= 0:
            #light_location = np.array(config.light_positions[traffic_light_idx] + (3.0,))
            position = self.traffic_light_labels[traffic_light_idx].pose.pose.position
            #light_location = np.array([position.x, position.y, position.z])
            self.upcoming_red_light_pub.publish(Point(position.x, position.y, position.z))

        if light_location is not None:
            state = self.get_light_state(light_location)
            light_wp = -1
            return light_wp, state
        self.waypoints = None
        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
