from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional

from omegaconf import MISSING


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
    LIVE_3DLIDAR = auto()


class DynamicObstaclesMessageType(Enum):
    MARKER_ARRAY = auto()
    AGENT_STATES = auto()
    ODOMETRY_ARRAY = auto()
    TRACKED_PERSONS = auto()


@dataclass
class DatasetConfiguration:
    directory: str = "data"
    trajectory_length: int = 50
    trajectory_time: int = 5
    coefficient_configuration: List[CoefficientConfiguration] = [
        CoefficientConfiguration.BEST_EXPERT
    ]
    static_obstacle_type: StaticObstacleType = StaticObstacleType.OCCUPANCY_MAP
    num_elite_coefficients: int = 80
    padding: int = 0


@dataclass
class TrainerConfiguration:
    results_directory: str = "results"  # type: ignore
    batch_size: int = MISSING  # type: ignore
    learning_rate: float = MISSING  # type: ignore
    num_epochs: int = MISSING  # type: ignore
    epochs_per_save: int = 10
    use_safetensors_for_saving: bool = False
    dataloader_num_workers: int = 12
    dataloader_pin_memory: bool = True


@dataclass
class VQVAEConfiguration:
    num_embeddings: int = 64
    embedding_dim: int = 4
    hidden_channels: int = 96
    checkpoint_path: Optional[str] = MISSING


@dataclass
class PixelCNNConfiguration:
    observation_embedding_dim: int = 32
    checkpoint_path: Optional[str] = MISSING


@dataclass
class ScoringNetworkConfiguration:
    num_samples: int = 50
    checkpoint_path: Optional[str] = None


@dataclass
class ProjectionConfiguration:
    guidance_type: Optional[GuidanceType] = GuidanceType.PRIEST
    use_obstacle_constraints: bool = True
    max_dynamic_obstacles: int = 10
    max_static_obstacles: int = 100
    padding: int = 1000
    max_velocity: float = 1.0
    max_outer_iterations: int = 2
    max_inner_iterations: int = 13
    tracking_weight: float = 1.0
    smoothness_weight: float = 0.2
    robot_radius: float = 0.5
    obstacle_radius: float = 0.3
    desired_velocity: float = 1.0


@dataclass
class LiveConfiguration:
    world_frame: str = "rplidar"
    robot_base_frame: str = "base_link"
    dynamic_obstacle_topic: str = "/pedestrians_pose"
    laser_scan_topic: str = "/laserscan"
    point_cloud_topic: str = "/pointcloud"
    velocity_command_topic: str = "/cmd_vel"
    goal_topic: str = "move_base_simple/goal"
    sub_goal_topic: str = "/subgoal"
    path_topic: str = "/trajectory"
    threshold_distance: float = 1.0
    padding: float = 1000
    dynamic_obstacle_message_type: DynamicObstaclesMessageType = (
        DynamicObstaclesMessageType.MARKER_ARRAY
    )
    use_global_path: bool = True
    max_angular_velocity: float = float("inf")  # Default to no limit


@dataclass
class Configuration:
    vqvae: VQVAEConfiguration = field(default_factory=VQVAEConfiguration)
    pixelcnn: PixelCNNConfiguration = field(default_factory=PixelCNNConfiguration)
    scoring_network: ScoringNetworkConfiguration = field(
        default_factory=ScoringNetworkConfiguration
    )
    trainer: TrainerConfiguration = field(default_factory=TrainerConfiguration)
    dataset: DatasetConfiguration = field(default_factory=DatasetConfiguration)
    projection: ProjectionConfiguration = field(default_factory=ProjectionConfiguration)
    live: LiveConfiguration = field(default_factory=LiveConfiguration)
    mode: Mode = Mode.INFERENCE_COMPLETE
