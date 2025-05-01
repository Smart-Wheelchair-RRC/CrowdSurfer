import torch
import torch.nn as nn
from torch import Tensor

from configuration import StaticObstacleType


class DynamicObstacleEmbedding(nn.Module):
    def __init__(self, embedding_dim: int, hidden_channels: int = 256):
        super().__init__()
        self.convolutions = nn.Sequential(
            nn.Conv1d(4, 32, kernel_size=1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
        )

        self.output = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_channels, 128),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(128, embedding_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        # x shape: (batch_size, 4, max_obstacles)
        x = self.convolutions(x)  # (batch_size, hidden_channels, max_obstacles)
        return self.output(x)  # (batch_size, embedding_dim)


class PointCloudEmbedding(nn.Module):
    def __init__(self, embedding_dim: int, hidden_channels: int = 256):
        super().__init__()
        self.convolutions = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
        )

        self.output = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_channels, 128),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(128, embedding_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        # x shape: (batch_size, 2, num_points)
        x = self.convolutions(x)  # (batch_size, hidden_channels, num_points)
        return self.output(x)  # (batch_size, embedding_dim)


class OccupancyMapEmbedding(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()

        self.convolutions = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.output = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(128, embedding_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        # x shape: (batch_size, 1, height, width)
        x = self.convolutions(x)
        return self.output(x)


class ObservationEmbedding(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        static_obstacle_type: StaticObstacleType,
        hidden_dim: int = 256,
    ):
        assert embedding_dim >= 8, "embedding_dim must be at least 8"

        super().__init__()
        if static_obstacle_type is StaticObstacleType.OCCUPANCY_MAP:
            self.static_obstacle_embedding = OccupancyMapEmbedding(embedding_dim)
        elif static_obstacle_type is StaticObstacleType.POINT_CLOUD:
            self.static_obstacle_embedding = PointCloudEmbedding(embedding_dim)

        # self.occupancy_map_embedding = OccupancyMapEmbedding(embedding_dim)
        # self.point_cloud_embedding = PointCloudEmbedding(embedding_dim)
        self.dynamic_obstacle_embedding = DynamicObstacleEmbedding(embedding_dim)

        self.timestep_combination_layer = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.timestep_output = nn.Linear(hidden_dim, embedding_dim)

        self.heading_to_goal_embedding = nn.Sequential(
            nn.Linear(1, embedding_dim // 8),
            nn.ReLU(),
        )

        self.combination_layer = nn.Sequential(
            nn.Linear((embedding_dim * 2) + (embedding_dim // 8), embedding_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        static_obstacle_vector: Tensor,
        # point_cloud_vector: Tensor,
        dynamic_obstacle_vector: Tensor,
        heading_to_goal_vector: Tensor,
        # body_pose_vector: Tensor,
    ) -> Tensor:
        # occupancy_map_vector: (batch_size, 1, height, width)
        # point_cloud_vector: (batch_size, 2, num_points)
        # dynamic_obstacle_vector: (batch_size, num_timesteps, 4, max_obstacles)
        # heading_to_goal_vector: (batch_size, 1) (angle)

        static_obstacle_embedding = self.static_obstacle_embedding.forward(
            static_obstacle_vector
        )  # (batch_size, embedding_dim)

        # point_cloud_embedding = self.point_cloud_embedding.forward(
        #     point_cloud_vector
        # )  # (batch_size, embedding_dim)

        dynamic_obstacle_embedding = self.dynamic_obstacle_embedding.forward(
            dynamic_obstacle_vector.flatten(start_dim=0, end_dim=1)  # (batch_size * num_timesteps, 4, max_obstacles)
        )  # (batch_size * num_timesteps, embedding_dim)

        dynamic_obstacle_embedding = self.timestep_combination_layer.forward(
            dynamic_obstacle_embedding.unflatten(
                0, (dynamic_obstacle_vector.size(0), dynamic_obstacle_vector.size(1))
            )  # (batch_size, num_timesteps, embedding_dim)
        )[0][:, -1, :]  # (batch_size, hidden_dim)

        dynamic_obstacle_embedding = self.timestep_output.forward(dynamic_obstacle_embedding)

        heading_to_goal_embedding = self.heading_to_goal_embedding.forward(heading_to_goal_vector)

        combined_embedding = self.combination_layer.forward(
            torch.concatenate(
                [
                    static_obstacle_embedding,
                    dynamic_obstacle_embedding,
                    heading_to_goal_embedding,
                ],
                dim=1,
            )  # (batch_size, embedding_dim * 2 + embedding_dim // 4)
        )  # (batch_size, embedding_dim)

        return combined_embedding
