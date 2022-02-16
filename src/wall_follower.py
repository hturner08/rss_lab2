#!/usr/bin/env python2
import random

import numpy as np

import rospy
from std_msgs.msg import Float32
from rospy.numpy_msg import numpy_msg
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped

import utils
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
    DEBUG = False
    WALL_TOPIC = "/wall"
    DISTANCE_THRESHOLD = .1  # meters
    SCAN_PARAMETERS = None

    def __init__(self):
        """
        Initializes the node, name it, and subscribe to the scan topic.

        Args:

        """

        self.drive_pub = rospy.Publisher(
            self.DRIVE_TOPIC, AckermannDriveStamped, queue_size=1)
        self.scan_sub = rospy.Subscriber(
            self.SCAN_TOPIC, LaserScan, self.scan_callback)
        self.line_pub = rospy.Publisher(self.WALL_TOPIC, Marker, queue_size=10)

    def scan_callback(self, msg):
        """ 
        Callback function for the laser scanner.

        Args:
            msg (LaserScan): ROS message from the laser scanner.
        """
        if self.SCAN_PARAMETERS is None:
            self.SCAN_PARAMETERS = ScanParameters(msg)
        self.points = utils.convert_scan_to_points(msg)
        self.features = self.process_points(self.points)
        self.controller(self.cal_dist_from_wall())
        self.visualize_features()

    def process_points_deprecated(self, points):
        """
        Iterate through points to produce up to wall features.

        Args:
            points (np.ndarray): Array containing the laser scan points in cartesian coordinates.

        Returns:
            features (np.ndarray): Array of features.
        """
        # TODO: Deal with entrances and exits.
        # Account for entrances and exits by seeing where points were removed in the ranges
        split_points = []
        i_last = 0
        for i in range(1, points.shape[0]):
            # if utils.get_angle(points[i, :], points[i-1, :]) > self.SCAN_PARAMETERS.angle_increment:
            split_points.append(points[i_last:i, :])
            i_last = i
        split_points.append(points[i_last:, :])
        # Calculate Lines
        features = []
        prev_feature_index = 0
        for group in points:
            for i in range(2, group.shape[0]):
                m, b = np.polyfit(group[i-2:i+1, 0], group[i-2:i+1, 1], 1)
                middle_dist = utils.get_distance(m, b, group[i, :])
                if middle_dist > self.DISTANCE_THRESHOLD:
                    # TODO: Get line using total least squares method
                    features.append((prev_feature_index, i))
                    prev_feature_index = i
                    i += 1
        return features

    def process_points(self, points):
        features = utils.split_and_merge(points)
        return features

    def cal_dist_from_wall(self):
        """
        Gets the distance from the wall.
        """
        dist = np.inf
        for feature in self.features:
            m, b = utils.get_line(feature[0], feature[1])
            dist = min(dist, utils.get_distance(m, b, np.zeros(2)))
        return dist

    def controller(self, dist):
        """
        Controls the robot based on the distance from the wall.

        Args:
            dist (float): Distance from the wall.
        """
        if self.DEBUG:
            rospy.loginfo("Distance from wall: %f", dist)
            rospy.loginfo("Current velocity: %f", self.VELOCITY)
        if dist < self.DESIRED_DISTANCE:
            self.drive(0, 0)
        else:
            self.drive(self.VELOCITY, 0)  # TODO: Add PID Controller

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

    def visualize_features(self):
        """
        Visualizes the features.
        """
        for feature in self.features:
            VisualizationTools.plot_line(
                feature[0], feature[1], self.line_pub, frame="/laser")


if __name__ == "__main__":
    rospy.init_node('wall_follower')
    wall_follower = WallFollower()
    rospy.loginfo('Initialized wall follower.')
    rospy.spin()
