import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from configuration import StaticObstacleType
from navigation.model.observation_embedding import ObservationEmbedding


class MaskedConv1d(nn.Module):
    MASK_TYPES = ["A", "B"]

    def __init__(self, mask_type: str, conv1d: nn.Conv1d):
        super().__init__()
        assert mask_type in self.MASK_TYPES

        self.conv1d = conv1d

        self.register_buffer("mask", self.conv1d.weight.data.clone())
        self.mask: Tensor
        _, _, size = self.conv1d.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, size // 2 + (mask_type == "B") :] = 0

    def forward(self, x):
        # x: (batch_size, input_channels, sequence_length)
        self.conv1d.weight.data *= self.mask
        output = self.conv1d.forward(x)
        return output  # (batch_size, output_channels, sequence_length)


class ConditionedGatedMaskedConv1d(nn.Module):
    def __init__(
        self,
        mask_type: str,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        padding: int,
        condition_embedding_dim: int,
        bias=False,
    ):
        super().__init__()
        self.masked_convolution_1 = MaskedConv1d(
            mask_type,
            conv1d=nn.Conv1d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=bias,
            ),
        )
        self.masked_convolution_2 = MaskedConv1d(
            mask_type,
            conv1d=nn.Conv1d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=bias,
            ),
        )
        self.condition_convolution_1 = nn.Conv1d(condition_embedding_dim, output_channels, kernel_size=1)
        self.condition_convolution_2 = nn.Conv1d(condition_embedding_dim, output_channels, kernel_size=1)

    def forward(self, x, condition_embedding):
        # x: (batch_size, input_channels, sequence_length)
        # condition_embedding: (batch_size, condition_embedding_dim, 1)
        input = F.tanh(
            self.masked_convolution_1(x)  # (batch_size, output_channels, sequence_length)
            + self.condition_convolution_1(condition_embedding)  # (batch_size, output_channels, 1)
        )  # (batch_size, output_channels, sequence_length)
        gate = F.sigmoid(
            self.masked_convolution_2(x)  # (batch_size, output_channels, sequence_length)
            + self.condition_convolution_2(condition_embedding)  # (batch_size, output_channels, 1)
        )  # (batch_size, output_channels, sequence_length)
        return input * gate  # (batch_size, output_channels, sequence_length)


class PixelCNN(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        condition_embedding_dim: int,
        input_channels: int = 1,
        kernel_size: int = 3,
        padding: int = 1,
        hidden_channels: int = 32,
        num_layers: int = 10,
        residual_connection_every: int = 2,
    ):
        super().__init__()
        self.residual_connection_every = residual_connection_every
        self.input_layer = nn.ModuleList(
            [
                ConditionedGatedMaskedConv1d(
                    "A",
                    input_channels=input_channels,
                    output_channels=hidden_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    condition_embedding_dim=condition_embedding_dim,
                    bias=False,
                ),
                nn.BatchNorm1d(hidden_channels),
            ]
        )

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.extend(
                [
                    ConditionedGatedMaskedConv1d(
                        "B",
                        input_channels=hidden_channels,
                        output_channels=hidden_channels,
                        kernel_size=kernel_size,
                        padding=padding,
                        condition_embedding_dim=condition_embedding_dim,
                        bias=False,
                    ),
                    nn.BatchNorm1d(hidden_channels),
                ]
            )

        self.output_layer = nn.Conv1d(hidden_channels, num_embeddings, kernel_size=1)

    def forward(self, x, condition_embedding) -> Tensor:
        # x: (batch_size, input_channels, sequence_length)
        # condition_embedding: (batch_size, condition_embedding_dim, 1)

        for layer in self.input_layer:
            if isinstance(layer, ConditionedGatedMaskedConv1d):
                x = layer(x, condition_embedding=condition_embedding)
            else:
                x = layer(x)

        residual = x

        for i, layer in enumerate([*self.layers]):
            if isinstance(layer, ConditionedGatedMaskedConv1d):
                x = layer(x, condition_embedding=condition_embedding)
            else:
                x = layer(x)
            if (i + 1) % self.residual_connection_every == 0:
                x += residual
                residual = x

        return self.output_layer(x)  # (batch_size, num_embeddings, sequence_length)


class CombinedPixelCNN(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        vqvae_hidden_channels: int,
        observation_embedding_dim: int,
        static_obstacle_type: StaticObstacleType,
        kernel_size: int = 3,
        padding: int = 1,
        hidden_channels: int = 32,
        num_layers: int = 10,
        observation_embedding_hidden_dim: int = 32,
    ):
        super().__init__()
        self.input_shape = (vqvae_hidden_channels, num_embeddings)
        self.pixel_cnn = PixelCNN(
            num_embeddings=num_embeddings,
            input_channels=num_embeddings,
            kernel_size=kernel_size,
            padding=padding,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            condition_embedding_dim=observation_embedding_dim,
        )
        self.observation_embedding = ObservationEmbedding(
            embedding_dim=observation_embedding_dim,
            hidden_dim=observation_embedding_hidden_dim,
            static_obstacle_type=static_obstacle_type,
        )

    def forward(
        self,
        static_obstacles: Tensor,
        # point_cloud: Tensor,
        dynamic_obstacles: Tensor,
        # body_pose: Tensor,
        heading_to_goal: Tensor,
    ):
        observation_embedding = self.observation_embedding.forward(
            static_obstacle_vector=static_obstacles,
            dynamic_obstacle_vector=dynamic_obstacles,
            heading_to_goal_vector=heading_to_goal.unsqueeze(-1),
        )  # (batch_size, observation_embedding_dim)

        codebook_probabilities = torch.zeros(observation_embedding.size(0), *self.input_shape).to(
            observation_embedding.device
        )  # (batch_size, sequence_length, num_embeddings)
        output = self.pixel_cnn.forward(
            codebook_probabilities.transpose(1, 2), observation_embedding.unsqueeze(-1)
        )  # (batch_size, num_embeddings, sequence_length)
        return output.transpose(1, 2), observation_embedding  # (batch_size, sequence_length, num_embeddings)
