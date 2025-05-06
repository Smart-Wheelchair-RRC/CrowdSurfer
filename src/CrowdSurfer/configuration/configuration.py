from dataclasses import MISSING, dataclass
from enum import Enum, auto
from typing import List, Optional


class CoefficientConfiguration(Enum):
    BEST_PRIEST = auto()
    ELITE_PRIEST = auto()
    BEST_EXPERT = auto()
    ELITE_EXPERT = auto()


class StaticObstacleType(Enum):
    OCCUPANCY_MAP = auto()
    POINT_CLOUD = auto()


class GuidanceType(Enum):
    PRIEST = auto()
    PROJECTION = auto()


class Mode(Enum):
    TRAIN_VQVAE = auto()
    TRAIN_PIXELCNN = auto()
    TRAIN_SCORING_NETWORK = auto()
    INFERENCE_VQVAE = auto()
    INFERENCE_PIXELCNN = auto()
    INFERENCE_COMPLETE = auto()
    VISUALIZE = auto()
    LIVE = auto()


class DynamicObstaclesMessageType(Enum):
    MARKER_ARRAY = auto()
    AGENT_STATES = auto()
    ODOMETRY_ARRAY = auto()
    TRACKED_PERSONS = auto()


@dataclass
class DatasetConfiguration:
    directory: str = MISSING  # type: ignore
    trajectory_length: int = MISSING  # type: ignore
    trajectory_time: int = MISSING  # type: ignore
    coefficient_configuration: List[CoefficientConfiguration] = MISSING  # type: ignore
    static_obstacle_type: StaticObstacleType = StaticObstacleType.OCCUPANCY_MAP
    num_elite_coefficients: int = 80
    padding: int = 0


@dataclass
class TrainerConfiguration:
    results_directory: str = MISSING  # type: ignore
    batch_size: int = MISSING  # type: ignore
    learning_rate: float = MISSING  # type: ignore
    num_epochs: int = MISSING  # type: ignore
    epochs_per_save: int = 10
    use_safetensors_for_saving: bool = False
    dataloader_num_workers: int = 12
    dataloader_pin_memory: bool = True


@dataclass
class VQVAEConfiguration:
    num_embeddings: int = MISSING  # type: ignore
    embedding_dim: int = MISSING  # type: ignore
    hidden_channels: int = 96
    checkpoint_path: Optional[str] = None


@dataclass
class PixelCNNConfiguration:
    observation_embedding_dim: int = 32
    checkpoint_path: Optional[str] = None


@dataclass
class ScoringNetworkConfiguration:
    num_samples: int = 50
    checkpoint_path: Optional[str] = None


@dataclass
class ProjectionConfiguration:
    guidance_type: Optional[GuidanceType] = GuidanceType.PRIEST
    num_priest_iterations: int = 1
    use_obstacle_constraints: bool = True
    max_dynamic_obstacles: int = 10
    max_static_obstacles: int = 50
    padding: int = 1000


@dataclass
class LiveConfiguration:
    world_frame: str = "rplidar"
    robot_base_frame: str = "base_link"
    odometry_topic: str = "/odometry"
    dynamic_obstacle_topic: str = "/pedestrians_pose"
    point_cloud_topic: str = "/pointcloud"
    laser_scan_topic: str = "/laserscan"
    velocity_command_topic: str = "/cmd_vel"
    goal_topic: str = "move_base_simple/goal"
    sub_goal_topic: str = "/subgoal"
    path_topic: str = "/trajectory"
    threshold_distance: float = 1.0
    padding: float = 1000
    time_horizon: float = 3
    previous_time_steps_for_dynamic: int = 5
    dynamic_obstacle_message_type: DynamicObstaclesMessageType = (
        DynamicObstaclesMessageType.MARKER_ARRAY
    )
    use_global_path: bool = True


@dataclass
class Configuration:
    vqvae: VQVAEConfiguration = MISSING  # type: ignore
    pixelcnn: PixelCNNConfiguration = MISSING  # type: ignore
    scoring_network: ScoringNetworkConfiguration = MISSING  # type: ignore
    trainer: TrainerConfiguration = MISSING  # type: ignore
    dataset: DatasetConfiguration = MISSING  # type: ignore
    projection: ProjectionConfiguration = MISSING  # type: ignore
    live: LiveConfiguration = MISSING  # type: ignore
    mode: Mode = MISSING  # type: ignore
