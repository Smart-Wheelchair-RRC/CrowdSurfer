from typing import Dict

import numpy as np
from genpy import Message

from navigation.constants import DataKeys

from .constants import ExtraDataKeys
from .utils import quaternion_to_heading


def process_odometry_message(
    odometry_message: Message,
) -> Dict[str, np.ndarray]:
    return {
        DataKeys.ODOMETRY: np.array(
            (
                odometry_message.pose.pose.position.x,
                odometry_message.pose.pose.position.y,
                quaternion_to_heading(
                    odometry_message.pose.pose.orientation.x,
                    odometry_message.pose.pose.orientation.y,
                    odometry_message.pose.pose.orientation.z,
                    odometry_message.pose.pose.orientation.w,
                ),
            )
        )
    }


# Expecting nav_msgs/OccupancyGrid message format
def process_occupancy_map_message(
    occupancy_map_message: Message,
) -> Dict[str, np.ndarray]:
    # Return flat to numpy 2D array
    return {
        DataKeys.OCCUPANCY_MAP: np.array(occupancy_map_message.data).reshape(
            (occupancy_map_message.info.height, occupancy_map_message.info.width)
        )
    }


# Expecting geometry_msgs/PoseArray message format
def process_obstacle_metadata_message(
    dynamic_obstacle_message: Message,
) -> Dict[str, np.ndarray]:
    body_pose = []
    dynamic_obstacle_position = []
    for pose in dynamic_obstacle_message.poses:
        position = [pose.position.x, pose.position.y, pose.position.z]
        orientation = quaternion_to_heading(
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        )
        dynamic_obstacle_position.append(position)
        body_pose.append(orientation)

    return {
        ExtraDataKeys.DYNAMIC_OBSTACLE_POSITION: np.array(dynamic_obstacle_position),
        ExtraDataKeys.BODY_POSE: np.array(body_pose),
    }


# This function takes in pedestrian_pose messages and converts them to a numpy array
def process_pedestrian_pose_message(pedestrian_pose_message: Message, max_obstacles: int = 10) -> Dict[str, np.ndarray]:
    # Padded numpy array for consistent array length
    position_array = np.full((max_obstacles, 3), np.nan)  # (timestamp, id, x, y)

    for index, marker in enumerate(pedestrian_pose_message.markers):
        if index >= max_obstacles:
            break  # Restrict extra obstacles

        if marker is not None:
            position_array[index] = [
                int(marker.id),
                marker.pose.position.x,
                marker.pose.position.y,
            ]

    return {ExtraDataKeys.DYNAMIC_OBSTACLE_POSITION: position_array}


# This function takes in laserscan messages and converts them to a numpy array
def process_laser_scan_message(
    laser_scan_message: Message,
    maximum_points: int = 1080,
    max_useful_range: float = 50,
) -> Dict[str, np.ndarray]:
    ranges = np.array(laser_scan_message.ranges)

    # Check for zero or very small angle_increment
    if abs(laser_scan_message.angle_increment) < 1e-10:
        num_points = len(ranges)
        angles = np.linspace(laser_scan_message.angle_min, laser_scan_message.angle_max, num_points)
    else:
        angles = np.arange(
            laser_scan_message.angle_min,
            laser_scan_message.angle_max + laser_scan_message.angle_increment,
            laser_scan_message.angle_increment,
        )

    # Ensure angles and ranges have the same length. If not, set as same.
    if len(angles) != len(ranges):
        min_len = min(len(angles), len(ranges))
        angles = angles[:min_len]
        ranges = ranges[:min_len]

    valid_ranges = (
        (ranges >= laser_scan_message.range_min)
        & ((ranges <= laser_scan_message.range_max) | (np.isinf(laser_scan_message.range_max)))
        & (ranges <= max_useful_range)
    )
    valid_ranges_array = ranges[valid_ranges]
    valid_angles_array = angles[valid_ranges]

    # Pad remaining positions for consistent array length
    padded_ranges = np.pad(
        valid_ranges_array,
        (0, maximum_points - len(valid_ranges_array)),
        mode="constant",
        constant_values=np.nan,
    )[:maximum_points]
    padded_angles = np.pad(
        valid_angles_array,
        (0, maximum_points - len(valid_angles_array)),
        mode="constant",
        constant_values=np.nan,
    )[:maximum_points]

    return {
        ExtraDataKeys.LASER_SCAN: np.stack((padded_ranges, padded_angles), axis=-1),  # shape: (maximum_points, 2)
    }


def process_pointcloud_message(
    pointcloud_message: Message,
    max_downsampled_points: int = 100,
    downsampling_voxel_size: float = 0.16,
) -> Dict[str, np.ndarray]:
    """
    Assumes sensor_msgs/PointCloud message format
    """
    import open3d

    points = [np.array((point.x, point.y, point.z)) for point in pointcloud_message.points]
    point_cloud = np.stack(points, axis=0) if len(points) > 0 else np.zeros((0, 3))  # shape: (num_points, 3)

    if downsampling_voxel_size > 0:
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(point_cloud)
        downsampled_pcd = pcd.voxel_down_sample(voxel_size=downsampling_voxel_size)
        current_pcd = np.asarray(downsampled_pcd.points)  # shape: (num_points, 3)

    current_pcd = current_pcd[..., :2][np.argsort(np.linalg.norm(current_pcd, axis=-1), axis=-1)][
        :max_downsampled_points
    ]  # shape: (<=max_downsampled_points, 2)

    out_pcd = np.full((max_downsampled_points, 2), np.nan)
    out_pcd[: current_pcd.shape[0], :] = current_pcd[:max_downsampled_points]

    return {DataKeys.POINT_CLOUD: out_pcd}
