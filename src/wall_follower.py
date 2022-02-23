#!/usr/bin/env python2
import random

import numpy as np
import rospy
import tf.transformations as tr
import tf2_ros

from std_msgs.msg import Float32
from rospy.numpy_msg import numpy_msg
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped

import geom
import line_finders
from visualization_tools import *


class ScanParameters:
    def __init__(self, scan):
        self.angle_min = scan.angle_min
        self.angle_max = scan.angle_max
        self.angle_increment = scan.angle_increment
        self.range_min = scan.range_min
        self.range_max = scan.range_max


class WallFollower:
    # Import ROS parameters from the "params.yaml" file.
    # Access these variables in class functions with self:
    # i.e. self.CONSTANT
    SCAN_TOPIC = rospy.get_param("wall_follower/scan_topic")
    DRIVE_TOPIC = rospy.get_param("wall_follower/drive_topic")
    SIDE = rospy.get_param("wall_follower/side")
    VELOCITY = rospy.get_param("wall_follower/velocity")
    DESIRED_DISTANCE = rospy.get_param("wall_follower/desired_distance")
    DEBUG = rospy.get_param("wall_follower/debug")
    WALL_TOPIC = "/wall"
    DISTANCE_THRESHOLD = .1  # meters
    SCAN_PARAMETERS = None
    LOOKAHEAD_DISTANCE = 1  # meters

    features = None

    def __init__(self, Kp=5, Ki=0, Kd=9):
        """
        Initializes the node, name it, and subscribe to the scan topic.

        Args:

        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.e_prev = 0
        self.e_sum = 0
        self.drive_pub = rospy.Publisher(
            self.DRIVE_TOPIC, AckermannDriveStamped, queue_size=1)
        self.scan_sub = rospy.Subscriber(
            self.SCAN_TOPIC, LaserScan, self.scan_callback)
        self.line_pub = rospy.Publisher(
            self.WALL_TOPIC + "/line", Marker, queue_size=10)
        self.text_pub = rospy.Publisher(
            self.WALL_TOPIC + "/dist", Marker, queue_size=10)
        self.debug_pub = rospy.Publisher(
            self.WALL_TOPIC + "/debug", Marker, queue_size=10)
        tfBuffer = tf2_ros.Buffer()
        listener = tf2_ros.TransformListener(tfBuffer)
        try:
            robot_trans = tfBuffer.lookup_transform(
                'base_link', 'laser_model', rospy.Time(0), rospy.Duration(1.0)).transform
            p = np.array([robot_trans.translation.x,
                          robot_trans.translation.y, robot_trans.translation.z])
            q = np.array([robot_trans.rotation.x, robot_trans.rotation.y,
                          robot_trans.rotation.z, robot_trans.rotation.w])
            self.robot_arr = tr.quaternion_matrix(q)
            self.robot_arr[0:3, -1] = p
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.loginfo("%s", e)

    def scan_callback(self, msg):
        """ 
        Callback function for the laser scanner.

        Args:
            msg (LaserScan): ROS message from the laser scanner.
        """
        if self.SCAN_PARAMETERS is None:
            self.SCAN_PARAMETERS = ScanParameters(msg)
            rospy.loginfo("%s", str(msg))
        self.points = geom.convert_scan_to_points(msg, side=self.SIDE)
        # self.features = [(self.points[0], self.points[-1])]
        self.features = self.process_points(self.points)
        curr_dist, curr_deriv, angle = self.cal_dist_from_wall()
        self.controller(curr_dist, curr_deriv, angle)
        self.visualize_features(curr_dist, angle)

    def process_points(self, points):
        # features = line_finders.split_and_merge(points)
        # features = line_finders.ransac(points)
        features = line_finders.plain_ols(points, self.SIDE)
        return features

    def cal_dist_from_wall(self):
        """
        Gets the distance from the wall.
        """
        dist = np.inf
        deriv = 0.0
        angle = 0.0
        for feature in self.features:
            new_dist = abs(geom.get_distance(
                feature[2], feature[3], np.zeros(2)))
            if new_dist < dist:
                dist = new_dist
                deriv = self.VELOCITY * feature[2] / feature[3] * dist
                # Closest point on line to another point is along perpendicular line
                angle = np.arctan(feature[2])
        return dist, deriv, angle

    def controller(self, dist, deriv, angle):
        """
        Controls the robot based on the distance from the wall.

        Args:
            dist (float): Distance from the wall.
        """
        vel = self.VELOCITY
        e = self.SIDE*(dist - self.DESIRED_DISTANCE)
        dedt = angle
        heading = self.Kd*dedt + self.Kp*e
        self.heading = min(.34, max(-.34, heading))
        self.drive(vel, self.heading)
        if self.DEBUG:
            rospy.loginfo("Distance from wall: %f", dist)
            rospy.loginfo("Current velocity: %f", vel)

    def drive(self, velocity, steering):
        """
        Publishes the given velocity and steering to the drive topic.

        Args:
            velocity (float): Velocity to publish.
            steering (float): Steering to publish.
        """
        msg = AckermannDriveStamped()
        msg.drive.speed = velocity
        msg.drive.steering_angle = steering
        self.drive_pub.publish(msg)

    def visualize_features(self, dist, angle):
        """
        Visualizes the features.
        """
        # rospy.loginfo("Number of Line Segments:%s", str(self.features))
        VisualizationTools.plot_line([f[0][0] for f in self.features] + [f[1][0] for f in self.features], [
                                     f[0][1] for f in self.features] + [f[1][1] for f in self.features], self.line_pub, frame="/base_link")

        VisualizationTools.plot_line([self.points[0][0], self.points[-1][0]], [
                                     self.points[0][1], self.points[-1][1]], self.debug_pub, color=(0, 1, 0), frame="/laser_model")
        VisualizationTools.plot_text(
            "Dist:" + str(dist) + "\n" + "Heading:" + str(self.heading) + "\n" + "Angle:" + str(angle), self.text_pub, 1)


def make_transform(arr, frame_name, parent='world'):
    t = geometry_msgs.msg.TransformStamped()
    t.header.stamp = rospy.Time.now()
    t.header.frame_id = parent
    t.child_frame_id = frame_name
    t.transform.translation.x = arr[0, -1]
    t.transform.translation.y = arr[1, -1]
    t.transform.translation.z = arr[2, -1]
    q = tr.quaternion_from_matrix(arr)
    t.transform.rotation.x = q[0]
    t.transform.rotation.y = q[1]
    t.transform.rotation.z = q[2]
    t.transform.rotation.w = q[3]
    return t


if __name__ == "__main__":
    rospy.init_node('wall_follower')
    wall_follower = WallFollower()
    rospy.loginfo('Initialized wall follower.')
    rospy.spin()
