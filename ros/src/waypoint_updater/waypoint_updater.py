#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint, TrafficLight
import math
import tf
from multiprocessing import Lock
import copy

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 200  # Number of waypoints we will publish. You can change this number

class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb, queue_size=1)
        self.base_waypoints_sub = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb, queue_size=1)

        # Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        rospy.Subscriber('/traffic_waypoint', TrafficLight, self.traffic_cb, queue_size=1)
        #self.base_waypoints_sub = rospy.Subscriber('/obstacle_waypoint', Point, self.obstacle_cb)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        self.waypoints_msg = None
        self.prev_closest_idx = 0
        self.closest_tl_idx = -1
        #
        # lock is required because in rospy subscriber's callbacks are executed in separate threads
        # https://answers.ros.org/question/110336/python-spin-once-equivalent/
        self.lock = Lock()

        rospy.spin()

    def pose_cb(self, msg):
        with self.lock:
            waypoints_msg = self.waypoints_msg
            closest_tl_idx = self.closest_tl_idx

        if waypoints_msg is not None:
            waypoints = waypoints_msg.waypoints
            #
            # find a closest waypoint
            #
            closest_idx = self.get_closest_waypoint(waypoints, msg.pose.position)
            self.prev_closest_idx = closest_idx
            point = waypoints[closest_idx]

            #
            # compose final waypoints
            #
            final_waypoints = Lane()
            final_waypoints.header.frame_id = self.waypoints_msg.header.frame_id
            final_waypoints.header.stamp = rospy.Time.now()
            max_idx = min(len(waypoints), closest_idx+LOOKAHEAD_WPS)
            final_waypoints.waypoints = waypoints[closest_idx:max_idx]

            #
            # Adjust speed for TL with read light if present
            #
            tl_distance_idx = closest_tl_idx - closest_idx
            if tl_distance_idx > 0 and tl_distance_idx < len(final_waypoints.waypoints):
                #
                # create a copy of waypoints as they will be modified
                #
                final_waypoints.waypoints = copy.deepcopy(final_waypoints.waypoints)

                for idx, point in enumerate(final_waypoints.waypoints):
                    velocity = point.twist.twist.linear.x
                    if idx <= tl_distance_idx:
                        distance = self.distance(point.pose.pose.position,
                                  final_waypoints.waypoints[tl_distance_idx].pose.pose.position)
                        stop_distance = 5.0
                        if distance <= stop_distance:
                            velocity = 0.0
                        else:
                            velocity = min(velocity, (distance-stop_distance)*0.1)
                    else:
                        velocity = 0.0

                    point.twist.twist.linear.x = velocity
            #
            # publish it
            #
            self.final_waypoints_pub.publish(final_waypoints)

    def waypoints_cb(self, msg):
        with self.lock:
            if self.waypoints_msg is None or self.waypoints_msg.header.stamp != msg.header.stamp:
                self.waypoints_msg = msg
                self.closest_tl_idx = -1

        self.base_waypoints_sub.unregister()

    def traffic_cb(self, msg):
        # Callback for /traffic_waypoint message.
        closest_tl_idx = -1
        if msg.state == TrafficLight.RED:
            with self.lock:
                waypoints_msg = self.waypoints_msg

            if waypoints_msg is not None:
                waypoints = waypoints_msg.waypoints

                #
                # find a closest waypoint
                #
                closest_tl_position = msg.pose.pose.position
                closest_tl_idx = self.get_closest_waypoint(waypoints, closest_tl_position)

        with self.lock:
            self.closest_tl_idx = closest_tl_idx

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        rospy.loginfo('obstacle_cb called')
        pass

    def get_closest_waypoint(self, waypoints, position):
        #
        # find a closest waypoint
        #
        closest_idx = self.prev_closest_idx
        if closest_idx > 50:
            closest_idx -= 50
        closest_distance = float("inf")
        for idx in range(closest_idx, len(waypoints)):
            point = waypoints[idx]
            dist = self.distance(point.pose.pose.position, position)
            if dist <= closest_distance:
                closest_distance = dist
                closest_idx = idx
            else:
                break

        return closest_idx

    def distance(self, a, b):
        return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
