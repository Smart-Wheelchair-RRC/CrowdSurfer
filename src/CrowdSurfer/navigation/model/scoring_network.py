from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from configuration import StaticObstacleType

from .observation_embedding import ObservationEmbedding


class ScoringNetwork(nn.Module):
    """
    Scoring network for the pipeline, is used to score the input batch of trajectories.
    Trajectories most similar to the expert trajectory are assigned a higher score (this is how loss is calculated)
    Note that this also works for comparing coefficients instead of trajectories (change num_features).
    """

    def __init__(
        self,
        obstacle_embedding_dim: int,
        hidden_channels: int = 64,
        dense_hidden_dim: int = 128,
        num_dense_layers: int = 3,
    ):
        super().__init__()
        self.convolutions = nn.Sequential(
            nn.Conv1d(
                2,
                hidden_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Conv1d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

        self.conditioning_layer = nn.Sequential(
            nn.Linear(obstacle_embedding_dim, hidden_channels),
            nn.ReLU(),
        )

        self.dense_layers = nn.Sequential(
            nn.Sequential(
                nn.Linear(hidden_channels * 2, dense_hidden_dim),
                nn.ReLU(),
            ),
            *(
                nn.Sequential(
                    nn.Linear(dense_hidden_dim, dense_hidden_dim),
                    nn.ReLU(),
                )
                for _ in range(num_dense_layers - 1)
            ),
            nn.Linear(dense_hidden_dim, 1),
        )

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        # shape x: (batch_size, num_trajectories, num_channels, num_features)
        # condition: (batch_size, condition_embedding_dim)

        # Calculate the score for each trajectory in the batch
        convoluted = self.convolutions.forward(
            x.flatten(start_dim=0, end_dim=1)
        )  # shape: (batch_size * num_trajectories, hidden_channels)
        convoluted = convoluted.reshape(
            x.size(0), x.size(1), -1
        )  # shape: (batch_size, num_trajectories, hidden_channels)

        condition = self.conditioning_layer.forward(condition)  # shape: (batch_size, hidden_channels)
        condition = condition.unsqueeze(1).expand_as(
            convoluted
        )  # shape: (batch_size, num_trajectories, hidden_channels)

        x = torch.cat([convoluted, condition], dim=-1)  # shape: (batch_size, num_trajectories, hidden_channels * 2)
        x = self.dense_layers.forward(x)  # shape: (batch_size, num_trajectories, 1)

        return x.squeeze(-1)  # shape: (batch_size, num_trajectories)

    def loss(self, prediction: Tensor, target: Tensor, trajectories: Tensor) -> Tensor:
        # shape prediction: (batch_size, num_trajectories)
        # shape target: (batch_size, num_channels, <=num_features)
        # shape trajectories: (batch_size, num_trajectories, num_channels, num_features)

        # Interpolate trajectories to have same num_features as target
        target = cast(
            Tensor,
            F.interpolate(
                target,
                size=trajectories.size(-1),
                mode="linear",
                align_corners=False,
            ),
        )  # shape: (batch_size, num_channels, num_features)

        # 1. Calculate which trajectory in the batch is the most similar to the target trajectory
        target = target.unsqueeze(1).expand_as(
            trajectories
        )  # shape: (batch_size, num_trajectories, num_channels, num_features)
        similarity = torch.norm(trajectories - target, dim=(2, 3))  # shape: (batch_size, num_trajectories)
        most_similar_trajectory = torch.argmin(similarity, dim=1)  # shape: (batch_size,)

        # 2. Calculate the loss as cross-entropy loss
        return F.cross_entropy(prediction, most_similar_trajectory)  # shape: ()


class CombinedScoringNetwork(nn.Module):
    def __init__(
        self,
        observation_embedding_dim: int,
        static_obstacle_type: StaticObstacleType,
        observation_embedding_hidden_dim: int = 32,
    ):
        super().__init__()
        self.scoring_network = ScoringNetwork(
            obstacle_embedding_dim=observation_embedding_dim,
        )
        self.observation_embedding = ObservationEmbedding(
            embedding_dim=observation_embedding_dim,
            hidden_dim=observation_embedding_hidden_dim,
            static_obstacle_type=static_obstacle_type,
        )

    def forward(
        self,
        trajectories: Tensor,
        static_obstacles: Tensor,
        dynamic_obstacles: Tensor,
        heading_to_goal: Tensor,
    ):
        # shape trajectories: (batch_size, num_trajectories, num_channels, num_features)
        observation_embedding = self.observation_embedding.forward(
            static_obstacle_vector=static_obstacles,
            dynamic_obstacle_vector=dynamic_obstacles,
            heading_to_goal_vector=heading_to_goal.unsqueeze(-1),
        )  # (batch_size, observation_embedding_dim)

        return self.scoring_network.forward(trajectories, observation_embedding)  # (batch_size, num_trajectories)
