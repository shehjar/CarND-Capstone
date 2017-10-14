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
MIN_VELOCITY = 1 # Minimum waypoint velocity

class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb, queue_size=1)
        self.base_waypoints_sub = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb, queue_size=1)

        # Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        rospy.Subscriber('/traffic_waypoint', TrafficLight, self.traffic_cb, queue_size=1)
        #self.base_waypoints_sub = rospy.Subscriber('/obstacle_waypoint', Point, self.obstacle_cb)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below
        self.waypoints_msg = None
        self.prev_closest_idx = 0
        self.closest_tl_idx = -1
        self.prev_final_waypoints = None
        self.prev_final_waypoints_start_idx = -1
        #
        # lock is required because in rospy subscriber's callbacks are executed in separate threads
        # https://answers.ros.org/question/110336/python-spin-once-equivalent/
        self.lock = Lock()

        rospy.spin()

    def pose_cb(self, msg):
        with self.lock:
            waypoints_msg = self.waypoints_msg
            closest_tl_idx = self.closest_tl_idx
            prev_final_waypoints = self.prev_final_waypoints
            prev_final_waypoints_start_idx = self.prev_final_waypoints_start_idx

        if waypoints_msg is not None:
            waypoints = waypoints_msg.waypoints
            #
            # find a closest waypoint
            #
            closest_idx = self.get_closest_waypoint(waypoints, msg.pose.position)
            self.prev_closest_idx = closest_idx
            point = waypoints[closest_idx]

            #
            # adjust it by heading
            #
            q = point.pose.pose.orientation
            position = point.pose.pose.position
            _, _, yaw = tf.transformations.euler_from_quaternion((q.x, q.y, q.z, q.w))
            heading = math.atan2((position.y - msg.pose.position.y), (position.x - msg.pose.position.x))

            if abs(yaw - heading) > (math.pi / 4):
                closest_idx += 1

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
            if tl_distance_idx > 0:
                #
                # create a copy of waypoints as they will be modified
                #
                final_waypoints.waypoints = copy.deepcopy(final_waypoints.waypoints)
                #
                # Get pose velocity using previous final_points if available
                #
                prev_final_waypoints_idx = closest_idx - prev_final_waypoints_start_idx
                if prev_final_waypoints is not None and \
                    prev_final_waypoints_start_idx >= 0 and prev_final_waypoints_idx < len(prev_final_waypoints):

                    velocity = prev_final_waypoints[prev_final_waypoints_idx].twist.twist.linear.x
                else:
                    velocity = final_waypoints.waypoints[0].twist.twist.linear.x
                #
                # compute velocity delta reduction per waypoint
                #
                delta = velocity / tl_distance_idx
                #
                # update waypoints' velocity, set 0 after TL.
                # Also set 0 under the MIN_VELOCITY to avoid zigzag-steering in slow
                # speeds.
                #
                for idx in range(len(final_waypoints.waypoints)):
                    if idx < tl_distance_idx and velocity - delta > MIN_VELOCITY:
                        velocity -= delta
                    else:
                        velocity = 0
                    final_waypoints.waypoints[idx].twist.twist.linear.x = velocity

            # save adjusted waypoints
            #
            with self.lock:
                self.prev_final_waypoints = final_waypoints.waypoints
                self.prev_final_waypoints_start_idx = closest_idx
            #
            # publish it
            #
            self.final_waypoints_pub.publish(final_waypoints)

    def waypoints_cb(self, msg):
        with self.lock:
            if self.waypoints_msg is None or self.waypoints_msg.header.stamp != msg.header.stamp:
                self.waypoints_msg = msg
                self.prev_closest_idx = 0
                self.closest_tl_idx = -1
                self.prev_final_waypoints = None

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
                position = msg.pose.pose.position
                closest_tl_idx = self.get_closest_waypoint(waypoints, position)

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
        if closest_idx > 5:
            closest_idx -= 5
        closest_distance = float("inf")
        distance = lambda a, b: math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)
        for idx in range(closest_idx, len(waypoints)):
            point = waypoints[idx]
            dist = distance(point.pose.pose.position, position)
            if dist < closest_distance:
                closest_distance = dist
                closest_idx = idx
            else:
                break

        return closest_idx

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist

if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
