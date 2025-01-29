from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Tuple, cast

import numpy as np

if TYPE_CHECKING:
    import rosbag

from navigation.projection_guidance import ProjectionGuidance


def translate_points(points: np.ndarray, translation: np.ndarray) -> np.ndarray:
    # points shape: (..., 2)
    # translation shape: (..., 2)

    return points + translation


def rotate_points(points: np.ndarray, angle: np.ndarray) -> np.ndarray:
    # points shape: (..., 2)
    # angle shape: (...)

    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    rotation_matrix = np.stack((cos_theta, -sin_theta, sin_theta, cos_theta), axis=-1)
    rotation_matrix = np.reshape(rotation_matrix, (*angle.shape, 2, 2))

    return np.einsum("...ij,...j->...i", rotation_matrix, points)


def quaternion_to_heading(
    x: float,
    y: float,
    z: float,
    w: float,
) -> float:
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * 2 + z * 2))


@dataclass
class TopicInfo:
    msg_type: str
    message_count: int
    connections: int
    frequency: float


def print_topic_info(bag: "rosbag.Bag"):
    topic_info = cast(Dict[str, TopicInfo], bag.get_type_and_topic_info().topics)

    for topic, info in topic_info.items():
        print(topic)
        print("\tMessage Type:", info.msg_type)
        print("\tMessage Count:", info.message_count)
        print("\tConnections:", info.connections)
        print("\tFrequency:", info.frequency)


def calculate_trajectories(coefficients: np.ndarray, projection_guidance: "ProjectionGuidance"):
    if coefficients.ndim == 3:  # best_priest_coefficients
        x, y = projection_guidance.coefficients_to_trajectory(
            coefficients[:, :, 0], coefficients[:, :, 1], position_only=True
        )
        return np.stack([x, y], axis=-1)
    elif coefficients.ndim == 4:  # elite_priest_coefficients
        trajectories = []
        for timestep_coeffs in coefficients:
            x, y = projection_guidance.coefficients_to_trajectory(
                timestep_coeffs[:, :, 0], timestep_coeffs[:, :, 1], position_only=True
            )
            trajectories.append(np.stack([x, y], axis=-1))
        return np.array(trajectories)
    else:
        raise ValueError("Unexpected coefficient shape")


def bresenham_ray_tracing(start: Tuple[float, float], end: Tuple[float, float]):
    """
    Implementation of Bresenham's line drawing algorithm
    See en.wikipedia.org/wiki/Bresenham's_line_algorithm
    Bresenham's Line Algorithm
    Produces a np.array from start and end (original from roguebasin.com)
    >>> points1 = bresenham((4, 4), (6, 10))
    >>> print(points1)
    np.array([[4,4], [4,5], [5,6], [5,7], [5,8], [6,9], [6,10]])
    """
    # setup initial conditions
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1
    is_steep = abs(dy) > abs(dx)  # determine how steep the line is
    if is_steep:  # rotate line
        x1, y1 = y1, x1
        x2, y2 = y2, x2
    # swap start and end points if necessary and store swap state
    swapped = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True
    dx = x2 - x1  # recalculate differentials
    dy = y2 - y1  # recalculate differentials
    error = int(dx / 2.0)  # calculate error
    y_step = 1 if y1 < y2 else -1
    # iterate over bounding box generating points between start and end
    y = y1
    points = []
    for x in range(x1, x2 + 1):
        coord = [y, x] if is_steep else (x, y)
        points.append(coord)
        error -= abs(dy)
        if error < 0:
            y += y_step
            error += dx
    if swapped:  # reverse the list if the coordinates were swapped
        points.reverse()
    points = np.array(points)
    return points
