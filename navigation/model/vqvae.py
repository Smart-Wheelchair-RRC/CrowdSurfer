from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class Encoder(nn.Module):
    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        output_channels: int,
        num_hidden_layers: int,
        kernel_size: int = 11,
        padding: int = 5,
    ):
        super(Encoder, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Conv1d(
                input_channels,
                hidden_channels,
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
        )
        self.hidden_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        hidden_channels,
                        hidden_channels,
                        kernel_size=kernel_size,
                        padding=padding,
                    ),
                    nn.BatchNorm1d(hidden_channels),
                    nn.ReLU(),
                )
                for _ in range(num_hidden_layers)
            ]
        )
        self.output_layer = nn.Sequential(
            nn.Conv1d(
                hidden_channels,
                output_channels,
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        # x shape: (batch_size, num_channels, num_timesteps)
        x = self.input_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output_layer(x)
        return x  # shape: (batch_size, output_channels, num_timesteps)


class Decoder(nn.Module):
    def __init__(
        self,
        input_features: int,
        hidden_features: int,
        output_features: int,
        num_hidden_layers: int,
    ):
        super(Decoder, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(
                input_features,
                hidden_features,
            ),
            nn.BatchNorm1d(hidden_features),
            nn.ReLU(),
        )
        self.hidden_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(
                        hidden_features,
                        hidden_features,
                    ),
                    nn.BatchNorm1d(hidden_features),
                    nn.ReLU(),
                )
                for _ in range(num_hidden_layers)
            ]
        )
        self.output_layer = nn.Linear(
            hidden_features,
            output_features,
        )

    def forward(self, x: Tensor) -> Tensor:
        # x shape: (batch_size, hidden_channels * embedding_dim)
        x = self.input_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output_layer(x)
        return x


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, beta=0.2):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.beta = beta
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

    def get_distances(self, x: Tensor) -> Tensor:
        # x shape: (batch_size, num_channels, embedding_dim)
        distances = torch.cdist(
            x.reshape(-1, self.embedding_dim), self.embedding.weight
        )
        return distances.reshape(
            x.shape[0], -1, self.num_embeddings
        )  # shape: (batch_size, num_channels, num_embeddings)

    def get_indices(self, x: Tensor) -> Tensor:
        # x shape: (batch_size, num_channels, embedding_dim)
        distances = self.get_distances(
            x
        )  # shape: (batch_size, num_channels, num_embeddings)
        return torch.argmin(distances, dim=-1)  # shape: (batch_size, num_channels)

    def loss(self, x: Tensor, quantized: Tensor) -> Tensor:
        commitment_loss = F.mse_loss(quantized.detach(), x)
        codebook_loss = F.mse_loss(quantized, x.detach())
        return codebook_loss + self.beta * commitment_loss

    def forward(self, x: Tensor) -> Tensor:
        # x shape: (batch_size, num_channels, embedding_dim)
        indices = self.get_indices(x)

        quantized = self.embedding.forward(indices)

        return quantized  # shape (batch_size, num_channels, embedding_dim)


class VQVAE(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        num_features: int = 11,
        hidden_channels: int = 96,
        input_channels: int = 2,
        num_hidden_layers: int = 10,
        hidden_feature_dim: int = 1024,
    ):
        super(VQVAE, self).__init__()
        self.num_features = num_features
        self.input_channels = input_channels
        self.encoder = Encoder(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            output_channels=hidden_channels,
            num_hidden_layers=num_hidden_layers,
        )
        self.reshape_to_embedding_dim = nn.Linear(num_features, embedding_dim)
        self.vector_quantizer = VectorQuantizer(
            num_embeddings=num_embeddings, embedding_dim=embedding_dim
        )
        self.decoder = Decoder(
            input_features=hidden_channels * embedding_dim,
            hidden_features=hidden_feature_dim,
            output_features=hidden_feature_dim,
            num_hidden_layers=3,
        )
        self.output_layer = nn.Linear(hidden_feature_dim, input_channels * num_features)

    def encode(self, x: Tensor) -> Tensor:
        return self.reshape_to_embedding_dim(
            self.encoder.forward(
                x
            )  # shape: (batch_size, hidden_channels, num_features)
        )  # shape: (batch_size, hidden_channels, embedding_dim)

    def decode(self, quantized_embedding: Tensor) -> Tensor:
        output: Tensor = self.output_layer(
            self.decoder.forward(
                quantized_embedding.flatten(
                    start_dim=-2
                )  # shape: (batch_size, hidden_channels * embedding_dim)
            )  # shape: (batch_size, hidden_feature_dim)
        )  # shape: (batch_size, input_channels * num_features)

        return output.reshape(
            quantized_embedding.shape[0], -1, self.num_features
        )  # shape: (batch_size, input_channels, num_features)

    def decode_from_indices(self, codebook_indices: Tensor) -> Tuple[Tensor, Tensor]:
        # codebook_indices shape: (batch_size, hidden_channels)

        quantized_embedding = self.vector_quantizer.embedding.forward(
            codebook_indices
        )  # shape: (batch_size, hidden_channels, embedding_dim)

        return (
            self.decode(
                quantized_embedding
            ),  # shape: (batch_size, input_channels, num_features)
            quantized_embedding,  # shape: (batch_size, hidden_channels, embedding_dim)
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        # x shape: (batch_size, input_channels, num_features)
        embedding = self.encode(
            x
        )  # shape: (batch_size, hidden_channels, embedding_dim)

        quantized_embedding = self.vector_quantizer.forward(
            embedding
        )  # shape: (batch_size, hidden_channels, embedding_dim)

        # The following line is equivalent to embedding in forward, and only
        # to allow gradients to flow through vector quantizer (not required during inference).
        quantized_embedding = embedding + (quantized_embedding - embedding).detach()

        return (
            self.decode(
                quantized_embedding
            ),  # shape: (batch_size, input_channels, num_features)
            embedding,  # shape: (batch_size, hidden_channels, embedding_dim)
            quantized_embedding,  # shape: (batch_size, hidden_channels, embedding_dim)
        )

    def loss(
        self,
        prediction: Tensor,
        embedding: Tensor,
        quantized_embedding: Tensor,
        target: Tensor,
        beta: float = 1e-5,
    ) -> Tensor:
        reconstruction_loss = F.mse_loss(prediction, target)
        commitment_loss = self.vector_quantizer.loss(embedding, quantized_embedding)
        return reconstruction_loss + commitment_loss * beta


if __name__ == "__main__":
    num_timesteps = 1000
    model = VQVAE(num_embeddings=16, embedding_dim=64, num_features=num_timesteps)
    x = torch.randn(2, 3, num_timesteps)
    x_hat, embedding, quantized = model.forward(x)
    loss = model.loss(x, embedding, quantized, x_hat)
    print(x_hat.shape, embedding.shape, quantized.shape, loss)
