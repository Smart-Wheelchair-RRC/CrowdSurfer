from dataclasses import dataclass


@dataclass
class DataDirectories:
    TRAIN: str = "train"
    VALIDATION: str = "validation"
    INFERENCE: str = "inference"


@dataclass
class DataKeys:
    """
    Contains the set of keys that are used to access data in the dataset.
    """

    ODOMETRY: str = "odometry"
    EGO_VELOCITY: str = "ego_velocity"
    DYNAMIC_OBSTACLES: str = "dynamic_obstacle_metadata"
    POINT_CLOUD: str = "point_cloud"
    DOWNSAMPLED_POINT_CLOUD: str = "downsampled_point_cloud"
    OCCUPANCY_MAP: str = "occupancy_map"
    RAY_TRACED_OCCUPANCY_MAP: str = "ray_traced_occupancy_map"
    GOAL_POSITION: str = "goal_position"
    EXPERT_TRAJECTORY: str = "expert_trajectory"
    BEST_PRIEST_COEFFICIENTS: str = "best_priest_coefficients"
    ELITE_PRIEST_COEFFICIENTS: str = "elite_priest_coefficients"
    BEST_EXPERT_COEFFICIENTS: str = "best_expert_coefficients"
    ELITE_EXPERT_COEFFICIENTS: str = "elite_expert_coefficients"
