#!/usr/bin/env python2
import numpy as np


def convert_scan_to_points(msg, keep_all=False, side=None):
    """
    Converts the given ROS LaserScan polar coordinate ranges to a list of points

    Args:
        msg (LaserScan): ROS message from the laser scanner.

    Returns:
        points (list): List of points.
    """
    if side is None:
        i = 0
        angle_min = msg.angle_min
        angle_max = msg.angle_max
    elif side == 1:
        i = len(msg.ranges)//2
        angle_min = 0
        angle_max = msg.angle_max
    elif side == -1:
        i = 0
        angle_min = msg.angle_min
        angle_max = 0
    else:
        raise ValueError("Side must be 1 or -1")
    points = np.empty((len(msg.ranges), 2))*np.nan
    angle = angle_min
    while angle < angle_max and i < len(msg.ranges)-1:
        if msg.ranges[i] > msg.range_min and msg.ranges[i] < msg.range_max:
            points[i, 0] = msg.ranges[i] * np.cos(angle)
            points[i, 1] = msg.ranges[i] * np.sin(angle)
        angle += msg.angle_increment
        i += 1
    if i == len(msg.ranges)-1:
        if msg.ranges[-1] > msg.range_min and msg.ranges[-1] < msg.range_max:
            points[-1, :] = (msg.ranges[-1] * np.cos(msg.angle_max),
                             msg.ranges[-1] * np.sin(msg.angle_max))
    if keep_all:
        return points
    return np.delete(points, np.where(np.isnan(points)), axis=0)


def get_angle(points):
    """
    Calculates the angle between two points.

    Args:
        points (np.ndarray): Array containing the two points.
    Returns:
        float: Angle between the two points.
    """
    return np.arctan2(points[1, 1]-points[0, 1], points[1, 0]-points[0, 0])


def get_line(p1, p2):
    """
    Calculates the line between two points.

    Args:
        p1 (np.ndarray): Array containing the first point.
        p2 (np.ndarray): Array containing the second point.
    Returns:
        m (float): Slope of the line.
        b (float): Y-intercept of the line.
    """
    if p1[0] == p2[0]:
        m = np.inf
        b = p1[0]
    else:
        m = (p2[1] - p1[1]) / (p2[0] - p1[0])
        b = p1[1] - m * p1[0]
    return m, b


def get_distance(m, b, point):
    """
    Calculates the distance of the given point from the line

    Args:
        m (float): Slope of the line.
        b (float): Y-intercept of the line.
        point (np.ndarray): Array containing the point.
    Returns:
        float: Distance of the given points.
    """
    return abs(m * point[0] + b - point[1]) / np.sqrt(m**2 + 1)


def get_farthest_point(m, b, points):
    """
    Calculates the farthest point from the line.

    Args:
        m (float): Slope of the line.
        b (float): Y-intercept of the line.
        points (np.ndarray): Array containing the points.
    Returns:
        farthest_index (int): Index of the farthest point.
        dist (float): Distance of the farthest point.
    """
    dist = 0
    farthest_index = None
    for i in range(points.shape[0]):
        new_dist = max(dist, get_distance(m, b, points[i, :]))
        # print("New Dist:", new_dist)
        if new_dist > dist:
            farthest_index = i
            dist = new_dist
    return farthest_index, dist


def get_point_over_threshold(m, b, points, thresh):
    """
    Calculates the point that is over the threshold distance from the line.

    Args:
        m (float): Slope of the line.
        b (float): Y-intercept of the line.
        points (np.ndarray): Array containing the points.
        thresh (float): Threshold distance.
    Returns:
        index (int): Index of the point.
        dist (float): Distance of the point.
    """
    dist = 0
    index = None
    for i in range(points.shape[0]):
        dist = get_distance(m, b, points[i, :])
        if dist > thresh:
            index = i
    return index, dist
