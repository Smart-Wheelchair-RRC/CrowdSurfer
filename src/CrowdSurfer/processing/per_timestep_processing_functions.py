"""
Processing Functions for a single timestep
When creating processing functions, separate into per timestep functions where possible before importing into the main processing function.
"""

from typing import Optional, Tuple, Union

import numpy as np
import open3d

from .utils import bresenham_ray_tracing


def generate_ray_traced_occupancy_map_from_point_cloud(
    point_cloud: np.ndarray,
    max_distance: float = 5,
    resolution: float = 0.05,
) -> np.ndarray:
    grid_size = int(2 * max_distance / resolution)

    center_x, center_y = grid_size // 2, grid_size // 2

    occupancy_map = np.ones((grid_size, grid_size)) / 2

    if point_cloud.shape[0] == 0 or point_cloud.shape[1] == 0:
        return occupancy_map

    point_cloud = point_cloud[:, :2]
    point_cloud_valid = point_cloud[~np.isnan(point_cloud).any(axis=1)]

    def is_valid(cell: Union[np.ndarray, Tuple[float, float]], grid_size: int):
        if cell[0] < grid_size and cell[0] >= 0 and cell[1] < grid_size and cell[1] >= 0:
            return True
        return False

    for x, y in point_cloud_valid:
        ix = int(center_x + x / resolution)
        iy = int(center_y + y / resolution)

        laser_beams = bresenham_ray_tracing((center_x, center_y), (ix, iy))  # line form the lidar to the occupied point
        for laser_beam in laser_beams:
            if is_valid(laser_beam, grid_size=grid_size):
                occupancy_map[laser_beam[0]][laser_beam[1]] = 1.0  # free area 1.0
        if is_valid((ix, iy), grid_size=grid_size):
            occupancy_map[ix][iy] = 0.0  # occupied area 0.0
            # occupancy_map[ix + 1][iy] = 1.0  # extend the occupied area
            # occupancy_map[ix][iy + 1] = 1.0  # extend the occupied area
            # occupancy_map[ix + 1][iy + 1] = 1.0  # extend the occupied area

    return occupancy_map


def laser_scan_to_point_cloud(laser_scan: np.ndarray) -> np.ndarray:
    """
    Convert a laser scan to a point cloud.

    Args:
        laser_scan: Laser scan of shape: (..., 2)

    Returns:
        The point cloud with shape (..., 2).
    """

    x = laser_scan[..., 0] * np.cos(laser_scan[..., 1])  # shape: (...)
    y = laser_scan[..., 0] * np.sin(laser_scan[..., 1])  # shape: (...)

    return np.stack((x, y), axis=-1)  # shape: (..., 2)


def downsample_point_cloud(
    point_cloud: np.ndarray,
    max_downsampled_points: int = 100,
    downsampling_voxel_size: float = 0.16,
    padding_value: Optional[float] = np.nan,
) -> np.ndarray:
    """
    Downsample a point cloud by voxelizing it and taking the first `max_downsampled_points` points or padding it.

    Args:
        point_cloud: Point cloud of shape: (num_points, 2 or 3)
        max_downsampled_points: The maximum number of points in the downsampled point cloud.
        downsampling_voxel_size: The voxel size for downsampling the point cloud.

    Returns:
        The downsampled point cloud with shape (max_downsampled_points, 2).
    """
    out_pcd = np.full((max_downsampled_points, 2), padding_value)

    if point_cloud.shape[0] == 0 or point_cloud.shape[1] == 0:
        return out_pcd if padding_value is not None else np.full((0, 2), 1000)

    if point_cloud.shape[1] == 2:
        point_cloud = np.concatenate([point_cloud, np.zeros((point_cloud.shape[0], 1))], axis=1)

    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(point_cloud)
    downsampled_pcd = pcd.voxel_down_sample(voxel_size=downsampling_voxel_size)
    current_pcd = np.asarray(downsampled_pcd.points)  # shape: (num_points, 3)

    current_pcd = current_pcd[..., :2][np.argsort(np.linalg.norm(current_pcd, axis=-1), axis=-1)][
        :max_downsampled_points
    ]  # shape: (<=max_downsampled_points, 2)

    if padding_value is not None:
        out_pcd[: current_pcd.shape[0], :] = current_pcd[:max_downsampled_points]
        return out_pcd
    else:
        return current_pcd


def generate_occupancy_map_from_point_cloud(
    point_cloud: np.ndarray,
    map_size: Tuple[int, int] = (60, 60),
    resolution: float = 0.1,
) -> np.ndarray:
    """
    Generate an occupancy map from a laser scan.

    Args:
        point_cloud: Point cloud of shape: (num_points, 2 or 3)
        map_size: The size of the occupancy map in pixels.
        resolution: The resolution of the occupancy map in meters per pixel.

    Returns:
        The occupancy map with shape (map_size[0], map_size[1]).
    """

    map_height, map_width = map_size

    origin = (map_width * resolution / 2, map_height * resolution / 2)

    occupancy_map = np.zeros(map_size, dtype=np.int8)

    if point_cloud.shape[0] == 0 or point_cloud.shape[1] == 0:
        return occupancy_map

    # Polar to Cartesian coordinates transformation
    x = point_cloud[:, 0]
    y = point_cloud[:, 1]

    # Transformation to map frame
    map_x = np.round((x + origin[0]) / resolution).astype(int)
    map_y = np.round((y + origin[1]) / resolution).astype(int)

    valid_points = (map_x >= 0) & (map_x < map_width) & (map_y >= 0) & (map_y < map_height)
    map_x = map_x[valid_points]
    map_y = map_y[valid_points]

    if len(map_x) > 0:
        occupancy_map[map_y, map_x] = 100  # Occupied

    # PLACEHOLDER ego-agent position on map
    ego_x = int(map_width / 2)
    ego_y = int(map_height / 2)
    occupancy_map[ego_y, ego_x] = 50  # Ego-agent

    return occupancy_map
