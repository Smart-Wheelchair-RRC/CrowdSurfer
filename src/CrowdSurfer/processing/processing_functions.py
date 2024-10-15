from typing import TYPE_CHECKING, Dict

import numpy as np

from navigation.constants import DataKeys

from . import per_timestep_processing_functions
from .utils import rotate_points, translate_points

if TYPE_CHECKING:
    from navigation.priest_guidance import PriestPlanner

from .constants import ExtraDataKeys


def get_or_assert(dictionary: Dict[str, np.ndarray], key: str) -> np.ndarray:
    output = dictionary.get(key)
    assert output is not None, f"Input dictionary must contain {key} key"
    return output


###############################################
########## Post Processing Functions ##########
###############################################
## NOTE: All functions must return a dictionary with the required keys and numpy arrays as values, all numpy arrays must have the same length (shape[0] across all keys)


def replace_odometry_heading_with_calculated_heading(
    data: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    odometry = get_or_assert(data, DataKeys.ODOMETRY)

    headings = np.arctan2(
        np.diff(odometry[:, 1]),
        np.diff(odometry[:, 0]),
    )  # shape (n-1,)
    odometry[:, 2] = np.concatenate((headings, [headings[-1]]))  # shape (n,)

    data[DataKeys.ODOMETRY] = odometry
    return data


def create_ego_velocities_from_odometry(
    data: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    odometry = get_or_assert(data, DataKeys.ODOMETRY)
    timestamps = get_or_assert(data, ExtraDataKeys.TIMESTAMP)

    odometry_diff = np.diff(odometry[:, :2], axis=0)  # shape: (n-1, 2)

    delta_t = np.diff(timestamps)  # shape: (n-1,)

    # Ensure delta_t is not zero
    delta_t[delta_t == 0] = 1e-6

    ego_velocities = odometry_diff / delta_t[:, np.newaxis]  # shape: (n-1, 2)
    ego_velocities = np.concatenate((np.zeros((1, 2)), ego_velocities), axis=0)  # shape: (n, 2)

    # Rotate velocities to be in the ego frame
    ego_velocities = rotate_points(ego_velocities, -odometry[:, 2])

    data[DataKeys.EGO_VELOCITY] = ego_velocities
    return data


def create_obstacle_velocities_from_positions(
    data: Dict[str, np.ndarray],
    transform_to_ego_frame: bool = False,  # For custom bags
    max_obstacles: int = 10,
    max_camera_angle: float = np.pi / 2,
) -> Dict[str, np.ndarray]:
    positions_per_timestep = get_or_assert(data, ExtraDataKeys.DYNAMIC_OBSTACLE_POSITION)

    # if transform_to_ego_frame:
    #     odometry = get_or_assert(data, DataKeys.ODOMETRY)

    timestamps = get_or_assert(data, ExtraDataKeys.TIMESTAMP)

    previous_data = {}
    obstacle_metadata = []

    for timestep, positions in enumerate(positions_per_timestep):
        positions: np.ndarray
        current_timestamp = timestamps[timestep]

        # Filter obstacles in front of the ego-agent
        front_obstacles = positions[positions[:, 1] > 0]

        distances = np.linalg.norm(front_obstacles[:, 1:3], axis=1)

        closest_indices = np.argsort(distances)[:max_obstacles]

        obstacle_position_velocity_array = np.full((max_obstacles, 5), np.nan)

        valid_obstacle_count = 0
        for obstacle_index in closest_indices:
            obstacle = front_obstacles[obstacle_index]
            obstacle_id, obstacle_position_x, obstacle_position_y = obstacle

            if np.isnan(obstacle_id):  # Skip rest if the obstacle data is not available (padded row)
                continue

            velocity_x, velocity_y = (
                0,
                0,
            )  # Default to zeros if velocity can't be calculated

            if obstacle_id in previous_data:
                (
                    previous_timestamp,
                    previous_obstacle_position_x,
                    previous_obstacle_position_y,
                ) = previous_data[obstacle_id]
                dt = current_timestamp - previous_timestamp

                if dt > 0:
                    velocity_x = (obstacle_position_x - previous_obstacle_position_x) / dt
                    velocity_y = (obstacle_position_y - previous_obstacle_position_y) / dt

            # Filter obstacles by fov angle
            obstacle_angle = np.arctan2(obstacle_position_y, obstacle_position_x)

            if abs(obstacle_angle) <= max_camera_angle / 2:
                obstacle_position_velocity_array[valid_obstacle_count] = [
                    obstacle_id,
                    obstacle_position_x,
                    obstacle_position_y,
                    velocity_x,
                    velocity_y,
                ]

                previous_data[obstacle_id] = (
                    current_timestamp,
                    obstacle_position_x,
                    obstacle_position_y,
                )

                valid_obstacle_count += 1

        # Removing the camera_link to lidar_link bias
        if transform_to_ego_frame:
            obstacle_position_velocity_array[:, 1:2] -= 0.8

        obstacle_metadata.append(obstacle_position_velocity_array)

    data[DataKeys.DYNAMIC_OBSTACLES] = np.array(obstacle_metadata)
    del data[ExtraDataKeys.DYNAMIC_OBSTACLE_POSITION]

    return data


# This function converts raw laserscan numpy array to point cloud format
def convert_laser_scan_to_point_cloud(
    data: Dict[str, np.ndarray],
    maximum_points: int = 100,
) -> Dict[str, np.ndarray]:
    laser_scan = get_or_assert(data, ExtraDataKeys.LASER_SCAN)

    x = laser_scan[..., 0] * np.cos(laser_scan[..., 1])  # shape: (n, max_points)
    y = laser_scan[..., 0] * np.sin(laser_scan[..., 1])  # shape: (n, max_points)
    point_cloud = np.stack((x, y), axis=-1)  # shape: (n, max_points, 2)

    all_pcd = []

    for timestep in range(point_cloud.shape[0]):
        current_pcd = point_cloud[timestep]
        current_pcd[..., :2][np.argsort(np.linalg.norm(current_pcd, axis=-1), axis=-1)][
            :maximum_points
        ]  # shape: (<=max_downsampled_points, 2)
        all_pcd.append(current_pcd)

    data[DataKeys.POINT_CLOUD] = np.array(all_pcd)

    return data


# This function converts the laser_scan data to Occupancy map data and updates the data-dictionary
def create_occupancy_map_from_point_cloud(
    data: Dict[str, np.ndarray], map_size=(60, 60), resolution=0.1
) -> Dict[str, np.ndarray]:
    point_clouds = get_or_assert(data, DataKeys.POINT_CLOUD)

    num_timesteps = point_clouds.shape[0]

    occupancy_maps = np.zeros((num_timesteps, *map_size), dtype=np.int8)

    for t in range(point_clouds.shape[0]):
        valid_indices = ~np.isnan(point_clouds[t, :, 0])

        # Checking for valid ranges, if not found, skip timestep.
        if len(valid_indices) == 0:
            continue

        occupancy_maps[t] = per_timestep_processing_functions.generate_occupancy_map_from_point_cloud(
            point_cloud=point_clouds[t, valid_indices, :],
            map_size=map_size,
            resolution=resolution,
        )

    data[DataKeys.OCCUPANCY_MAP] = occupancy_maps

    return data


def create_goal_positions_and_expert_trajectory_from_odometry(
    data: Dict[str, np.ndarray],
    goal_increment: int = 20,
) -> Dict[str, np.ndarray]:
    odometry = get_or_assert(data, DataKeys.ODOMETRY)

    timestamps = get_or_assert(data, ExtraDataKeys.TIMESTAMP)

    CUSTOM_DT = 0.1
    goal_increment = int(
        goal_increment * (CUSTOM_DT / np.mean(np.diff(timestamps)))
    )  # Adjust goal increment based on average dt

    goal_positions = np.zeros((odometry.shape[0], 2))
    expert_trajectories = np.zeros((odometry.shape[0], goal_increment + 1, 2))
    for timestep in range(odometry.shape[0]):
        goal_timestep = min(timestep + goal_increment, odometry.shape[0] - 1)

        goal_positions[timestep] = odometry[goal_timestep, :2]

        expert_trajectory_from_timestep_to_goal = odometry[timestep : goal_timestep + 1, :2]

        # Transform expert trajectory to be in the ego frame
        expert_trajectory_from_timestep_to_goal = rotate_points(
            translate_points(expert_trajectory_from_timestep_to_goal, -odometry[timestep, :2]),
            -odometry[timestep, 2],
        )

        # Ensure trajectory has the same length by padding with goal position
        if expert_trajectory_from_timestep_to_goal.shape[0] < goal_increment + 1:
            expert_trajectory_from_timestep_to_goal = np.pad(
                expert_trajectory_from_timestep_to_goal,
                (
                    (
                        0,
                        goal_increment + 1 - expert_trajectory_from_timestep_to_goal.shape[0],
                    ),
                    (0, 0),
                ),
                mode="edge",
            )

        expert_trajectories[timestep] = expert_trajectory_from_timestep_to_goal

    # Rotate goal positions to be in the ego frame
    goal_positions = rotate_points(translate_points(goal_positions, -odometry[:, :2]), -odometry[:, 2])

    data[DataKeys.GOAL_POSITION] = goal_positions
    data[DataKeys.EXPERT_TRAJECTORY] = expert_trajectories

    return data


# This function computes the PRIEST trajectories
def create_priest_trajectories_from_observations(
    data: Dict[str, np.ndarray],
    planner: "PriestPlanner",
) -> Dict[str, np.ndarray]:
    """
    This function computes the PRIEST trajectories, expects all inputs to be in the ego frame.
    """
    import os

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    point_cloud = data.get(DataKeys.POINT_CLOUD)
    assert (
        point_cloud is not None or planner.num_obstacles == 0
    ), "Input dictionary must contain 'point_cloud' key or max_static_obstacles must be 0"

    dynamic_obstacle_metadata = data.get(DataKeys.DYNAMIC_OBSTACLES)
    assert (
        dynamic_obstacle_metadata is not None or planner.num_dynamic_obstacles == 0
    ), "Input dictionary must contain 'dynamic_obstacle_metadata' key or max_dynamic_obstacles must be 0"

    ego_velocity = get_or_assert(data, DataKeys.EGO_VELOCITY)

    goal_position = get_or_assert(data, DataKeys.GOAL_POSITION)

    expert_trajectory = get_or_assert(data, DataKeys.EXPERT_TRAJECTORY)

    best_coefficients = []
    elite_coefficients = []
    best_trajectories = []
    elite_trajectories = []

    for timestep in range(goal_position.shape[0]):
        current_velocity: np.ndarray = ego_velocity[timestep]  # shape (2,)
        current_goal_position: np.ndarray = goal_position[timestep]  # shape: (2,)
        current_point_cloud: np.ndarray = (
            point_cloud[timestep]  # shape: (max_downsampled_points, 2)
            if planner.num_obstacles != 0
            else np.zeros((0, 2))
        )
        current_dynamic_obstacle_metadata: np.ndarray = (
            dynamic_obstacle_metadata[timestep]  # shape: (max_obstacles, 5)
            if planner.num_dynamic_obstacles != 0
            else np.zeros((0, 5))
        )

        planner.x_init = 0
        planner.y_init = 0
        planner.theta_init = 0

        planner.vx_init = current_velocity[0]
        planner.vy_init = current_velocity[1]

        # Setting PRIEST goal as the final odometry positions of the current rosbag
        planner.x_fin = current_goal_position[0]
        planner.y_fin = current_goal_position[1]

        # PointCloud Data Processing
        x_obs_init_1: np.ndarray = current_point_cloud[:, 0]
        y_obs_init_1: np.ndarray = current_point_cloud[:, 1]

        mask = (
            # (x_obs_init_1 < 30)
            # & (y_obs_init_1 < 30)
            np.logical_not(np.isnan(x_obs_init_1)) & np.logical_not(np.isnan(y_obs_init_1))
        )
        x_obs_init_1 = x_obs_init_1[mask]
        y_obs_init_1 = y_obs_init_1[mask]

        planner.update_obstacle_pointcloud(np.column_stack((x_obs_init_1, y_obs_init_1, np.zeros_like(x_obs_init_1))))

        # Dynamic Obstacle Data Processing
        num_obstacles = np.sum(~np.isnan(current_dynamic_obstacle_metadata[:, 0]))

        # Padding for redundant obstacles
        planner.x_obs_init_dy.fill(1000.0)
        planner.y_obs_init_dy.fill(1000.0)
        planner.vx_obs_dy.fill(0.0)
        planner.vy_obs_dy.fill(0.0)

        id_to_index = {}
        current_index = 0

        for obstacle_index in range(num_obstacles):
            obstacle_id = current_dynamic_obstacle_metadata[obstacle_index, 0]

            if obstacle_id not in id_to_index:
                if current_index < planner.num_dynamic_obstacles:
                    id_to_index[obstacle_id] = current_index
                    current_index += 1
                else:
                    raise ValueError("Warning: More obstacles detected than can be handled.")

            array_idx = id_to_index[obstacle_id]
            planner.x_obs_init_dy[array_idx] = current_dynamic_obstacle_metadata[obstacle_index, 1]
            planner.y_obs_init_dy[array_idx] = current_dynamic_obstacle_metadata[obstacle_index, 2]
            planner.vx_obs_dy[array_idx] = current_dynamic_obstacle_metadata[obstacle_index, 3]
            planner.vy_obs_dy[array_idx] = current_dynamic_obstacle_metadata[obstacle_index, 4]

        c_x_best, c_y_best, x_best, y_best, c_x_elite, c_y_elite, x_elite, y_elite, _ = planner.run_optimization(
            custom_x_waypoint=expert_trajectory[timestep, :, 0],
            custom_y_waypoint=expert_trajectory[timestep, :, 1],
        )
        best_coefficients.append(np.stack([c_x_best, c_y_best], axis=-1))
        elite_coefficients.append(np.stack([c_x_elite, c_y_elite], axis=-1))
        best_trajectories.append(np.stack([x_best, y_best], axis=-1))
        elite_trajectories.append(np.stack([x_elite, y_elite], axis=-1))

    data[DataKeys.BEST_PRIEST_COEFFICIENTS] = np.array(best_coefficients)
    data[DataKeys.ELITE_PRIEST_COEFFICIENTS] = np.array(elite_coefficients)
    data[ExtraDataKeys.BEST_PRIEST_TRAJECTORY] = np.array(best_trajectories)
    data[ExtraDataKeys.ELITE_PRIEST_TRAJECTORY] = np.array(elite_trajectories)

    return data


def delete_from_data(data: Dict[str, np.ndarray], key: str) -> Dict[str, np.ndarray]:
    assert key in data, f"Key '{key}' not found in data"
    del data[key]
    return data


###############################################
############## Editing Functions ##############
###############################################


def downsample_point_cloud(
    data: Dict[str, np.ndarray],
    max_downsampled_points: int = 1950,
    downsampling_voxel_size: float = 0.16,
) -> Dict[str, np.ndarray]:
    assert downsampling_voxel_size > 0, "Downsampling voxel size must be greater than 0"

    point_cloud = get_or_assert(data, DataKeys.POINT_CLOUD)

    # Accounting for the missing 'z' column
    point_cloud_3d = np.concatenate((point_cloud, np.zeros_like(point_cloud[:, :, 0][..., np.newaxis])), axis=2)

    all_pcd = []

    for timestep in range(point_cloud_3d.shape[0]):
        all_pcd.append(
            per_timestep_processing_functions.downsample_point_cloud(
                point_cloud_3d[timestep],
                max_downsampled_points=max_downsampled_points,
                downsampling_voxel_size=downsampling_voxel_size,
            )
        )

    data[DataKeys.DOWNSAMPLED_POINT_CLOUD] = np.array(all_pcd)

    return data


def prune_timesteps_close_to_goal(
    data: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    # Prune timesteps where expert trajectory contains goal at position other than the last timestep
    expert_trajectory = get_or_assert(data, DataKeys.EXPERT_TRAJECTORY)  # shape: (n, goal_increment + 1, 2)

    goal_position = get_or_assert(data, DataKeys.GOAL_POSITION)  # shape: (n, 2)

    # Check if expert trajectory contains goal position at before the last timestep
    invalid_timesteps = np.all(np.isclose(expert_trajectory[:, -2], goal_position, atol=1e-3), axis=-1)

    for key in data.keys():
        data[key] = data[key][~invalid_timesteps]

    return data


# def expert_trajectory_to_bernstein(data: Dict[str, np.ndarray], degree=10) -> Dict[str, np.ndarray]:
#     from scipy.special import comb

#     trajectory = get_or_assert(data, DataKeys.EXPERT_TRAJECTORY)

#     num_timesteps, traj_length, coords = trajectory.shape
#     assert coords == 2, "Input trajectory should have shape [num_timesteps, 30, 2]"

#     # Normalize time to [0, 1]
#     t = np.linspace(0, 1, traj_length)

#     # Compute Bernstein basis polynomials
#     n = degree
#     k = np.arange(n + 1)
#     t_matrix = t[:, np.newaxis]
#     k_matrix = k[np.newaxis, :]
#     basis = comb(n, k_matrix) * (t_matrix**k_matrix) * ((1 - t_matrix) ** (n - k_matrix))

#     # Compute least squares solution for Bernstein coefficients
#     bernstein_coeffs = np.zeros((num_timesteps, n + 1, 2))
#     for i in range(num_timesteps):
#         for j in range(2):
#             bernstein_coeffs[i, :, j] = np.linalg.lstsq(basis, trajectory[i, :, j], rcond=None)[0]

#     data["expert_coefficients"] = bernstein_coeffs

#     return data


def expert_trajectory_to_bernstein_priest(data: Dict[str, np.ndarray], planner: "PriestPlanner"):
    import os

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    point_cloud = data.get(DataKeys.POINT_CLOUD)
    assert (
        point_cloud is not None or planner.num_obstacles == 0
    ), "Input dictionary must contain 'point_cloud' key or max_static_obstacles must be 0"

    ego_velocity = get_or_assert(data, DataKeys.EGO_VELOCITY)
    goal_position = get_or_assert(data, DataKeys.GOAL_POSITION)
    expert_trajectory = get_or_assert(data, DataKeys.EXPERT_TRAJECTORY)

    best_coefficients = []
    elite_coefficients = []

    for timestep in range(goal_position.shape[0]):
        current_velocity: np.ndarray = ego_velocity[timestep]  # shape (2,)
        current_goal_position: np.ndarray = goal_position[timestep]  # shape: (2,)
        current_point_cloud: np.ndarray = (
            point_cloud[timestep]  # shape: (max_downsampled_points, 2)
            if planner.num_obstacles != 0
            else np.zeros((0, 2))
        )

        planner.x_init = 0
        planner.y_init = 0
        planner.theta_init = 0

        planner.vx_init = current_velocity[0]
        planner.vy_init = current_velocity[1]

        # Setting PRIEST goal as the final odometry positions of the current rosbag
        planner.x_fin = current_goal_position[0]
        planner.y_fin = current_goal_position[1]

        # PointCloud Data Processing
        redundant_static_obs = np.full(
            (current_point_cloud.shape[0], 2), 1000.0
        )  # Making Static Obstacles out of range
        x_obs_init_1: np.ndarray = redundant_static_obs[:, 0]
        y_obs_init_1: np.ndarray = redundant_static_obs[:, 1]

        planner.update_obstacle_pointcloud(np.column_stack((x_obs_init_1, y_obs_init_1, np.zeros_like(x_obs_init_1))))

        # Setting all dynamic obstacle positions out of range
        planner.x_obs_init_dy.fill(1000.0)
        planner.y_obs_init_dy.fill(1000.0)
        planner.vx_obs_dy.fill(0.0)
        planner.vy_obs_dy.fill(0.0)

        c_x_best, c_y_best, _, _, c_x_elite, c_y_elite, _, _, _ = planner.run_optimization(
            custom_x_waypoint=expert_trajectory[timestep, :, 0],
            custom_y_waypoint=expert_trajectory[timestep, :, 1],
        )
        best_coefficients.append(np.stack([c_x_best, c_y_best], axis=-1))
        elite_coefficients.append(np.stack([c_x_elite, c_y_elite], axis=-1))

    data[DataKeys.BEST_EXPERT_COEFFICIENTS] = np.array(best_coefficients)
    data[DataKeys.ELITE_EXPERT_COEFFICIENTS] = np.array(elite_coefficients)

    return data


def replace_close_to_goal_with_linear_coefficients(
    data: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    # Prune timesteps where expert trajectory contains goal at position other than the last timestep
    expert_trajectory = get_or_assert(data, DataKeys.EXPERT_TRAJECTORY)  # shape: (n, goal_increment + 1, 2)
    goal_position = get_or_assert(data, DataKeys.GOAL_POSITION)  # shape: (n, 2)
    best_priest_coefficients = get_or_assert(data, DataKeys.BEST_PRIEST_COEFFICIENTS)
    elite_priest_coefficients = get_or_assert(data, DataKeys.ELITE_PRIEST_COEFFICIENTS)
    best_expert_coefficients = get_or_assert(data, DataKeys.BEST_EXPERT_COEFFICIENTS)
    elite_expert_coefficients = get_or_assert(data, DataKeys.ELITE_EXPERT_COEFFICIENTS)
    odometry = get_or_assert(data, DataKeys.ODOMETRY)

    # Check if expert trajectory contains goal position at before the last timestep
    invalid_timesteps = np.all(np.isclose(expert_trajectory[:, -2], goal_position, atol=1e-3), axis=-1)

    # Check if odometry is the same as previous timestep
    invalid_timesteps |= np.all(
        np.isclose(odometry[:, :2], np.roll(odometry[:, :2], 1, axis=0), atol=1e-3),
        axis=-1,
    )

    # Check if goal position is the same as previous timestep
    invalid_timesteps |= np.all(np.isclose(goal_position, np.roll(goal_position, 1, axis=0), atol=1e-3), axis=-1)

    linear_coefficients = np.transpose(
        np.linspace(
            best_priest_coefficients[:, 0],
            goal_position,
            num=best_priest_coefficients.shape[1],
        ),
        (1, 0, 2),
    )

    # Replace the coefficients with linear interpolation between the first and last coefficient
    data[DataKeys.BEST_PRIEST_COEFFICIENTS] = np.where(
        invalid_timesteps[:, np.newaxis, np.newaxis],
        linear_coefficients,
        best_priest_coefficients,
    )

    data[DataKeys.BEST_EXPERT_COEFFICIENTS] = np.where(
        invalid_timesteps[:, np.newaxis, np.newaxis],
        linear_coefficients,
        best_expert_coefficients,
    )

    elite_linear_coefficients = np.transpose(
        np.linspace(
            elite_priest_coefficients[:, :, 0],
            np.tile(
                goal_position[:, np.newaxis, :],
                (1, elite_priest_coefficients.shape[1], 1),
            ),
            num=elite_priest_coefficients.shape[2],
        ),
        (1, 2, 0, 3),
    )

    data[DataKeys.ELITE_PRIEST_COEFFICIENTS] = np.where(
        invalid_timesteps[:, np.newaxis, np.newaxis, np.newaxis],
        elite_linear_coefficients,
        elite_priest_coefficients,
    )

    data[DataKeys.BEST_EXPERT_COEFFICIENTS] = np.where(
        invalid_timesteps[:, np.newaxis, np.newaxis, np.newaxis],
        elite_linear_coefficients,
        elite_expert_coefficients,
    )

    return data


def create_ray_traced_occupancy_map_from_point_cloud(
    data: Dict[str, np.ndarray],
    max_distance: float = 5,
    resolution: float = 0.05,
):
    """
    Computes Occupancy map using Breshen ray casting
    """
    points = get_or_assert(data, DataKeys.POINT_CLOUD)

    maps = []
    for timestep in range(points.shape[0]):
        maps.append(
            per_timestep_processing_functions.generate_ray_traced_occupancy_map_from_point_cloud(
                point_cloud=points[timestep], max_distance=max_distance, resolution=resolution
            )
        )

    data[DataKeys.RAY_TRACED_OCCUPANCY_MAP] = np.array(maps)

    return data
