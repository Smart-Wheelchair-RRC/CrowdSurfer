from dataclasses import dataclass


@dataclass
class ExtraDataKeys:
    """
    Contains extra keys that are used to access data in the dataset, but are only for processing intermediates.
    """

    BODY_POSE: str = "body_pose"
    DYNAMIC_OBSTACLE_POSITION: str = "dynamic_obstacle_position"
    LASER_SCAN: str = "laser_scan"
    BEST_PRIEST_TRAJECTORY: str = "best_trajectory"
    ELITE_PRIEST_TRAJECTORY: str = "elite_trajectory"
    TIMESTAMP: str = "timestamp"
