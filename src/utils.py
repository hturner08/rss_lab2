#!/usr/bin/env python2
import numpy as np


def convert_scan_to_points(msg, keep_all=False):
    """
    Converts the given ROS LaserScan polar coordinate ranges to a list of points

    Args:
        msg (LaserScan): ROS message from the laser scanner.

    Returns:
        points (list): List of points.
    """
    points = np.zeros((len(msg.ranges), 2))
    for i in range(len(msg.ranges)-1):
        if msg.ranges[i] > msg.range_min and msg.ranges[i] < msg.range_max:
            points[i, :] = (msg.ranges[i] * np.cos(i * msg.angle_increment),
                            msg.ranges[i] * np.sin(i * msg.angle_increment))
        else:
            points[i, :] = (np.nan, np.nan)
    if msg.ranges[-1] > msg.range_min and msg.ranges[-1] < msg.range_max:
        points[-1, :] = (msg.ranges[-1] * np.cos(msg.angle_max),
                         msg.ranges[-1] * np.sin(msg.angle_max))
    else:
        points[i, :] = (np.nan, np.nan)
    if keep_all:
        return points
    return np.delete(points, np.where(np.isnan(points[:, 0])), axis=0)


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
    dist = 0
    index = None
    for i in range(points.shape[0]):
        dist = get_distance(m, b, points[i, :])
        if dist > thresh:
            index = i
    return index, dist


def split_and_merge(points, thresh=2):
    """
    Uses the Split-And-Merge Algorithm to extract line features from the given
    scan.

    Args:
        points (np.ndarray): Array containing the laser scan points in cartesian coordinates.

    Returns:
        features (np.ndarray): Array of features.
    """
    point_sets = [points]
    lines = []
    index = 0
    while index < len(point_sets):
        curr_set = point_sets[index]
        point_sets.remove(curr_set)
        if curr_set.shape[0] < 2:
            continue
        # TODO: Allow option to replace OLS with other methods
        # print("Current Set:", curr_set)
        try:
            m, b = np.polyfit(curr_set[:, 0], curr_set[:, 1], 1)
        except (np.linalg.LinAlgError, ValueError) as e:
            # except np.linalg.LinAlgError as e:
            print("LinAlgError:", e)
            print(curr_set)
            lines.append((curr_set[0, :], curr_set[-1, :]))
            continue
        point_index, dist = get_farthest_point(m, b, curr_set)
        # print("Split index:", point_index)
        # print("Farthest Dist:", dist)
        # print("Line Parameters:", m, b)
        # print(curr_set.shape)
        if point_index is not None:
            if point_index == 0 or point_index == curr_set.shape[0] - 1:
                point_sets.append(curr_set[:curr_set.shape[0]//2, :])
                point_sets.append(curr_set[curr_set.shape[0]//2:, :])
            else:
                lines.append((curr_set[0, :], curr_set[-1, :]))
        else:
            lines.append((curr_set[0, :], curr_set[-1, :]))
    return lines
