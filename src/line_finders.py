import geom
import numpy as np
import warnings
# TODO: Implement RANSACC


def ransac(points, iterations=100, threshold=0.05, min_points=100):
    """
    Uses the RANSAC algorithm to extract line from the given points. Defaults to OLS 

    Args:
        points (np.ndarray): Array containing the laser scan points in cartesian coordinates.

    Returns:
        features (list): List of line features containing a single line segment and its parameters.
    """
    for i in range(iterations):
        indices = np.random.choice(points.shape[0], min_points, replace=False)
        np.sort(indices)
        subset = points[indices, :]
        m, b = np.polyfit(subset[:, 0], subset[:, 1], 1)
        inliers = []
        for j in range(points.shape[0]):
            if geom.get_distance(m, b, points[j]) < threshold:
                inliers.append(j)
        if len(inliers)/len(points) > 0.75:
            return [(points[indices[0], :], points[indices[-1], :], m, b)]
    warnings.warn("RANSAC failed to find a line. Defaulting to Plain OLS.")
    return plain_ols(points)


def plain_ols(points, side=None):
    """
    Uses ordinary least squares to extract line from the given points.

    Args:
        points (np.ndarray): Array containing the laser scan points in cartesian coordinates.

    Returns:
        features (list): List of line features containing a single line segment and its parameters.
    """
    size = points.shape[0]
    # if side == 1:
    #     start = size//6
    #     end = 7*size//12
    # elif side == -1:
    #     start = size//2
    #     end = size//2 + size//3
    # else:
    start = 0
    end = size
    m, b = np.polyfit(points[start:end, 0], points[start:end, 1], 1)
    return [(points[start, :], points[min(end, size-1), :], m, b)]


def split_and_merge(points, m_thresh=0.1, b_thresh=0.1):
    """
    Uses the Split-And-Merge Algorithm to extract line features from the given
    points.

    Args:
        points (np.ndarray): Array containing the laser scan points in cartesian coordinates.

    Returns:
        features (list): LIst of line features containing line segments and their parameters.
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
        try:
            m, b = np.polyfit(curr_set[:, 0], curr_set[:, 1], 1)
        except (np.linalg.LinAlgError, ValueError) as e:
            m = np.inf
            b = np.average(curr_set[:, 0])
            lines.append((curr_set[0, :], curr_set[-1, :], m, b))
            continue
        point_index, dist = geom.get_farthest_point(m, b, curr_set)
        if point_index is not None:
            if point_index == 0 or point_index == curr_set.shape[0] - 1:
                point_sets.append(curr_set[:curr_set.shape[0]//2, :])
                point_sets.append(curr_set[curr_set.shape[0]//2:, :])
            else:
                lines.append((curr_set[0, :], curr_set[-1, :], m, b))
        else:
            lines.append((curr_set[0, :], curr_set[-1, :], m, b))
    # print(lines)
    merged = False
    while not merged:
        merged = True
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                if abs(lines[i][2] - lines[j][2]) < m_thresh and abs(lines[i][3] - lines[j][3]) < b_thresh:
                    lines[i] = (lines[i][0], lines[j][1],
                                lines[i][2], lines[i][3])
                    lines.remove(lines[j])
                    merged = False
                    break
    return lines
