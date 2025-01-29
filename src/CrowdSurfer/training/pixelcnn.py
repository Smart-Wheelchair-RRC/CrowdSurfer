import os
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import load_checkpoint_in_model
from matplotlib import pyplot as plt
from torch import Tensor

import configuration
from configuration import GuidanceType, Mode, StaticObstacleType
from navigation.constants import DataDirectories
from navigation.dataset import PixelCNNDataset
from navigation.model import VQVAE, CombinedPixelCNN
from navigation.projection_guidance import ProjectionGuidance
from navigation.utilities import project_coefficients

from .trainer import Trainer, TrainerMode


class PixelCNNTrainer(Trainer):
    def __init__(
        self,
        configuration: configuration.Configuration,
    ):
        super().__init__(
            model=CombinedPixelCNN(
                num_embeddings=configuration.vqvae.num_embeddings,
                vqvae_hidden_channels=configuration.vqvae.hidden_channels,
                observation_embedding_dim=configuration.pixelcnn.observation_embedding_dim,
                static_obstacle_type=configuration.dataset.static_obstacle_type,
            ),
            dataset=PixelCNNDataset(
                dataset_configuration=configuration.dataset,
                projection_configuration=configuration.projection,
                type=DataDirectories.TRAIN,
            ),
            results_directory=configuration.trainer.results_directory,
            learning_rate=configuration.trainer.learning_rate,
            batch_size=configuration.trainer.batch_size,
            mode=TrainerMode.TRAIN if configuration.mode is Mode.TRAIN_PIXELCNN else TrainerMode.INFERENCE,
            validation_dataset=PixelCNNDataset(
                dataset_configuration=configuration.dataset,
                projection_configuration=configuration.projection,
                type=DataDirectories.VALIDATION,
            )
            if os.path.exists(os.path.join(configuration.dataset.directory, DataDirectories.VALIDATION))
            else None,
            dataloader_num_workers=configuration.trainer.dataloader_num_workers,
            dataloader_pin_memory=configuration.trainer.dataloader_pin_memory,
        )

        self.model: CombinedPixelCNN
        self.dataset: PixelCNNDataset
        self.validation_dataset: PixelCNNDataset

        self.static_obstacle_type = configuration.dataset.static_obstacle_type
        self.use_projection_guidance = configuration.projection.guidance_type is GuidanceType.PROJECTION

        self.vqvae = VQVAE(
            num_embeddings=configuration.vqvae.num_embeddings,
            embedding_dim=configuration.vqvae.embedding_dim,
            hidden_channels=configuration.vqvae.hidden_channels,
            num_features=11,
        ).to(self.accelerator.device)

        self.projection_guidance = ProjectionGuidance(
            num_obstacles=0,
            num_timesteps=configuration.dataset.trajectory_length,
            total_time=configuration.dataset.trajectory_time,
            obstacle_ellipse_semi_major_axis=0,
            obstacle_ellipse_semi_minor_axis=0,
            max_projection_iterations=1,
            device=self.accelerator.device,
        )

        # Load VQ-VAE checkpoint
        assert configuration.vqvae.checkpoint_path is not None
        load_checkpoint_in_model(self.vqvae, configuration.vqvae.checkpoint_path)

        self.vqvae.eval()

    def forward(self, data: Dict[str, Tensor]) -> Tensor:
        return self.model.forward(
            static_obstacles=data["static_obstacles"],
            dynamic_obstacles=data["dynamic_obstacles"],
            heading_to_goal=data["heading_to_goal"],
        )[0]  # shape: (batch_size, vqvae_hidden_channels, vqvae_num_embeddings)

    def loss(self, output: Tensor, data: Dict[str, Tensor]) -> Tensor:
        coefficients = data["coefficients"]

        with torch.no_grad():
            embedding = self.vqvae.encode(coefficients)
            indices = self.vqvae.vector_quantizer.get_indices(embedding)  # shape: (batch_size, hidden_channels)

        return F.cross_entropy(output.transpose(-1, -2), indices)

    def inference(
        self,
        dataset_index: Optional[int] = None,
        num_samples: int = 1,
        save_plot_image: bool = False,
        show_plot: bool = True,
    ):
        assert self.mode is TrainerMode.INFERENCE, "Inference mode is not enabled."
        assert self.validation_dataset is not None, "Validation dataset is not provided."

        if dataset_index is None:
            dataset_index = np.random.randint(0, len(self.dataset))

        print("Data Index:", dataset_index)
        data = self.validation_dataset[dataset_index]

        coefficients = data["coefficients"].unsqueeze(0).to(self.accelerator.device)
        expert_trajectory = data["expert_trajectory"].unsqueeze(0).to(self.accelerator.device)
        static_obstacles = data["static_obstacles"].unsqueeze(0).to(self.accelerator.device)
        dynamic_obstacles = data["dynamic_obstacles"].unsqueeze(0).to(self.accelerator.device)
        heading_to_goal = data["heading_to_goal"].unsqueeze(0).to(self.accelerator.device)

        self.model.eval()

        with torch.no_grad():
            embedding = self.vqvae.encode(coefficients)  # shape: (batch_size, hidden_channels, embedding_dim)

            distances = self.vqvae.vector_quantizer.get_distances(embedding).flatten(
                0, 1
            )  # shape: (batch_size * hidden_channels, num_embeddings)

            loss_indices = self.vqvae.vector_quantizer.get_indices(embedding)

            # Use distances as logits for a multinomial distribution
            vqvae_indices = (
                torch.multinomial(
                    torch.softmax(-distances, dim=-1),
                    num_samples=num_samples,
                    replacement=True,
                )
                .unflatten(0, (embedding.shape[0], -1))
                .transpose(1, 2)
                .flatten(0, 1)
            )  # shape: (batch_size * num_samples, hidden_channels)

            vqvae_output, _ = self.vqvae.decode_from_indices(
                vqvae_indices
            )  # shape: (batch_size * num_samples, num_channels, num_features)

            if self.use_projection_guidance:
                vqvae_output = project_coefficients(
                    projection_guidance=self.projection_guidance,
                    coefficients=vqvae_output,
                    initial_ego_position_x=torch.zeros(num_samples).to(self.accelerator.device),
                    initial_ego_position_y=torch.zeros(num_samples).to(self.accelerator.device),
                    initial_ego_velocity_x=torch.zeros(num_samples).to(self.accelerator.device),
                    initial_ego_velocity_y=torch.zeros(num_samples).to(self.accelerator.device),
                    initial_ego_acceleration_x=torch.zeros(num_samples).to(self.accelerator.device),
                    initial_ego_acceleration_y=torch.zeros(num_samples).to(self.accelerator.device),
                    final_ego_position_x=expert_trajectory[:, 0, -1].tile(num_samples),
                    final_ego_position_y=expert_trajectory[:, 1, -1].tile(num_samples),
                    obstacle_positions_x=torch.tensor([]).to(self.accelerator.device).unsqueeze(0),
                    obstacle_positions_y=torch.tensor([]).to(self.accelerator.device).unsqueeze(0),
                    obstacle_velocities_x=torch.tensor([]).to(self.accelerator.device).unsqueeze(0),
                    obstacle_velocities_y=torch.tensor([]).to(self.accelerator.device).unsqueeze(0),
                )

            pixelcnn_output = self.model.forward(
                static_obstacles=static_obstacles,
                dynamic_obstacles=dynamic_obstacles,
                heading_to_goal=heading_to_goal,
            )[0]

            loss = F.cross_entropy(
                pixelcnn_output.transpose(-1, -2),
                loss_indices,
            )

            pixelcnn_output = pixelcnn_output.flatten(
                0, 1
            )  # shape: (batch_size * vqvae_hidden_channels, vqvae_num_embeddings)

            pixelcnn_indices = (
                torch.multinomial(
                    torch.exp(torch.log_softmax(pixelcnn_output, dim=-1)),
                    num_samples=num_samples,
                    replacement=True,
                )
                .unflatten(0, (embedding.shape[0], -1))
                .transpose(1, 2)
                .flatten(0, 1)
            )  # shape: (batch_size * num_samples, hidden_channels)

            pixelcnn_output, _ = self.vqvae.decode_from_indices(
                pixelcnn_indices
            )  # shape: (batch_size * num_samples, num_channels, num_features)

            if self.use_projection_guidance:
                pixelcnn_output = project_coefficients(
                    projection_guidance=self.projection_guidance,
                    coefficients=pixelcnn_output,
                    initial_ego_position_x=torch.zeros(num_samples).to(self.accelerator.device),
                    initial_ego_position_y=torch.zeros(num_samples).to(self.accelerator.device),
                    initial_ego_velocity_x=torch.zeros(num_samples).to(self.accelerator.device),
                    initial_ego_velocity_y=torch.zeros(num_samples).to(self.accelerator.device),
                    initial_ego_acceleration_x=torch.zeros(num_samples).to(self.accelerator.device),
                    initial_ego_acceleration_y=torch.zeros(num_samples).to(self.accelerator.device),
                    final_ego_position_x=expert_trajectory[:, 0, -1].tile(num_samples),
                    final_ego_position_y=expert_trajectory[:, 1, -1].tile(num_samples),
                    obstacle_positions_x=torch.tensor([]).to(self.accelerator.device).unsqueeze(0),
                    obstacle_positions_y=torch.tensor([]).to(self.accelerator.device).unsqueeze(0),
                    obstacle_velocities_x=torch.tensor([]).to(self.accelerator.device).unsqueeze(0),
                    obstacle_velocities_y=torch.tensor([]).to(self.accelerator.device).unsqueeze(0),
                )

        print(f"Loss: {loss.item()}")

        target_x, target_y = self.projection_guidance.coefficients_to_trajectory(
            coefficients_x=coefficients[:, 0, :],
            coefficients_y=coefficients[:, 1, :],
            position_only=True,
        )

        vqvae_projected_trajectory_x, vqvae_projected_trajectory_y = (
            self.projection_guidance.coefficients_to_trajectory(
                coefficients_x=vqvae_output[:, 0, :],
                coefficients_y=vqvae_output[:, 1, :],
                position_only=True,
            )
        )

        pixelcnn_projected_trajectory_x, pixelcnn_projected_trajectory_y = (
            self.projection_guidance.coefficients_to_trajectory(
                coefficients_x=pixelcnn_output[:, 0, :],
                coefficients_y=pixelcnn_output[:, 1, :],
                position_only=True,
            )
        )

        if not show_plot and not save_plot_image:
            return

        if self.static_obstacle_type is StaticObstacleType.OCCUPANCY_MAP:
            # Plot occupancy map (static_obstacles)
            extent = 5

            static_obstacles = static_obstacles.squeeze().cpu().numpy()

            plt.imshow(
                static_obstacles,
                cmap="binary",
                extent=(
                    -extent,
                    extent,
                    -extent,
                    extent,
                ),
                alpha=0.5,
            )

        # Plot dynamic obstacles

        for i in range(dynamic_obstacles.shape[1]):
            plt.scatter(
                dynamic_obstacles[0, i, 0].cpu().numpy(),
                dynamic_obstacles[0, i, 1].cpu().numpy(),
                color="yellow",
            )

        # Plot x and y coordinates
        for sample_index in range(num_samples):
            plt.plot(
                vqvae_projected_trajectory_x[sample_index].squeeze().cpu().numpy(),
                vqvae_projected_trajectory_y[sample_index].squeeze().cpu().numpy(),
                label="Output",
                color="black",
                alpha=0.7,
            )
            plt.plot(
                pixelcnn_projected_trajectory_x[sample_index].squeeze().cpu().numpy(),
                pixelcnn_projected_trajectory_y[sample_index].squeeze().cpu().numpy(),
                label="Output",
                color="green",
            )

        plt.plot(
            target_x.squeeze().cpu().numpy(),
            target_y.squeeze().cpu().numpy(),
            label="Input",
            color="red",
        )

        plt.plot(
            expert_trajectory[0, 0].cpu().numpy(),
            expert_trajectory[0, 1].cpu().numpy(),
            label="Expert",
            color="blue",
        )

        if save_plot_image:
            plt.savefig(self.results_directory.joinpath(f"inference_{dataset_index}.png"))

        if show_plot:
            plt.show()

        plt.close()
