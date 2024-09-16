from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

import configuration
from configuration import GuidanceType, Mode
from navigation.dataset import VQVAEDataset
from navigation.model import VQVAE
from navigation.projection_guidance import ProjectionGuidance
from navigation.utilities import project_coefficients

from .trainer import Trainer, TrainerMode


class VQVAETrainer(Trainer):
    def __init__(self, configuration: configuration.Configuration):
        super().__init__(
            model=VQVAE(
                num_embeddings=configuration.vqvae.num_embeddings,
                # embedding_dim=8,  # complex
                embedding_dim=configuration.vqvae.embedding_dim,  # complex2
                hidden_channels=configuration.vqvae.hidden_channels,
            ),
            dataset=VQVAEDataset(
                dataset_configuration=configuration.dataset,
                projection_configuration=configuration.projection,
            ),
            results_directory=configuration.trainer.results_directory,
            learning_rate=configuration.trainer.learning_rate,
            batch_size=configuration.trainer.batch_size,
            mode=TrainerMode.TRAIN if configuration.mode == Mode.TRAIN_VQVAE else TrainerMode.INFERENCE,
            dataloader_num_workers=configuration.trainer.dataloader_num_workers,
            dataloader_pin_memory=configuration.trainer.dataloader_pin_memory,
        )

        self.model: VQVAE
        self.dataset: VQVAEDataset

        self.use_projection_guidance = configuration.projection.guidance_type is GuidanceType.PROJECTION

        self.projection_guidance = ProjectionGuidance(
            num_obstacles=0,
            num_timesteps=configuration.dataset.trajectory_length,
            total_time=configuration.dataset.trajectory_time,
            obstacle_ellipse_semi_major_axis=0,
            obstacle_ellipse_semi_minor_axis=0,
            max_projection_iterations=1,
            device=self.accelerator.device,
        )

    def inference(
        self,
        dataset_index: Optional[int] = None,
        num_samples: int = 1,
        save_plot_image: bool = False,
        show_plot: bool = True,
    ):
        assert self.mode is TrainerMode.INFERENCE, "Inference mode is not enabled."
        if dataset_index is None:
            dataset_index = np.random.randint(0, len(self.dataset))

        print("Data Index:", dataset_index)
        data = self.dataset[dataset_index]

        coefficients = data["coefficients"].unsqueeze(0).to(self.accelerator.device)
        expert_trajectory = data["expert_trajectory"].unsqueeze(0).to(self.accelerator.device)
        ego_velocity = data["projection_ego_velocity"].unsqueeze(0).to(self.accelerator.device)
        goal_position = data["projection_goal_position"].unsqueeze(0).to(self.accelerator.device)

        self.model.eval()

        with torch.no_grad():
            embedding = self.model.encode(coefficients)  # shape: (batch_size, hidden_channels, embedding_dim)

            distances = self.model.vector_quantizer.get_distances(embedding).flatten(
                0, 1
            )  # shape: (batch_size * hidden_channels, num_embeddings)

            # Use distances as logits for a multinomial distribution
            indices = (
                torch.multinomial(
                    torch.softmax(-distances, dim=-1),
                    num_samples=num_samples,
                    replacement=True,
                )
                .unflatten(0, (embedding.shape[0], -1))
                .transpose(1, 2)
                .flatten(0, 1)
            )  # shape: (batch_size * num_samples, hidden_channels)

            output, quantized_embedding = self.model.decode_from_indices(
                indices
            )  # shape: (batch_size * num_samples, num_channels, num_features)
            if self.use_projection_guidance:
                output = project_coefficients(
                    projection_guidance=self.projection_guidance,
                    coefficients=output,
                    initial_ego_position_x=torch.zeros(num_samples).to(self.accelerator.device),
                    initial_ego_position_y=torch.zeros(num_samples).to(self.accelerator.device),
                    initial_ego_velocity_x=ego_velocity[:, 0].tile(num_samples),
                    initial_ego_velocity_y=ego_velocity[:, 1].tile(num_samples),
                    initial_ego_acceleration_x=torch.zeros(num_samples).to(self.accelerator.device),
                    initial_ego_acceleration_y=torch.zeros(num_samples).to(self.accelerator.device),
                    final_ego_position_x=goal_position[:, 0].tile(num_samples),
                    final_ego_position_y=goal_position[:, 1].tile(num_samples),
                    obstacle_positions_x=torch.tensor([]).to(self.accelerator.device).unsqueeze(0),
                    obstacle_positions_y=torch.tensor([]).to(self.accelerator.device).unsqueeze(0),
                    obstacle_velocities_x=torch.tensor([]).to(self.accelerator.device).unsqueeze(0),
                    obstacle_velocities_y=torch.tensor([]).to(self.accelerator.device).unsqueeze(0),
                )

            loss = self.model.loss(
                output,
                embedding=embedding.repeat(num_samples, 1, 1),
                quantized_embedding=quantized_embedding,
                target=coefficients.repeat(num_samples, 1, 1),
                beta=1e-5,
            )

        print(f"Loss: {loss.item() / num_samples}")

        target_x, target_y = self.projection_guidance.coefficients_to_trajectory(
            coefficients_x=coefficients[:, 0, :],
            coefficients_y=coefficients[:, 1, :],
            position_only=True,
        )

        projected_trajectory_x, projected_trajectory_y = self.projection_guidance.coefficients_to_trajectory(
            coefficients_x=output[:, 0, :],
            coefficients_y=output[:, 1, :],
            position_only=True,
        )

        if not show_plot and not save_plot_image:
            return

        # Plot x and y coordinates
        for sample_index in range(num_samples):
            plt.plot(
                projected_trajectory_x[sample_index].squeeze().cpu().numpy(),
                projected_trajectory_y[sample_index].squeeze().cpu().numpy(),
                label="Output",
                color="grey",
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

    def forward(self, data: Dict[str, Tensor]) -> Tuple[Tensor, ...]:
        coefficients = data["coefficients"]

        output, embedding, quantized_embedding = self.model.forward(coefficients)

        if self.use_projection_guidance:
            output = project_coefficients(
                projection_guidance=self.projection_guidance,
                coefficients=output,
                initial_ego_position_x=torch.zeros(output.shape[0]).to(self.accelerator.device),
                initial_ego_position_y=torch.zeros(output.shape[0]).to(self.accelerator.device),
                initial_ego_velocity_x=data["projection_ego_velocity"][:, 0],
                initial_ego_velocity_y=data["projection_ego_velocity"][:, 1],
                initial_ego_acceleration_x=torch.zeros(output.shape[0]).to(self.accelerator.device),
                initial_ego_acceleration_y=torch.zeros(output.shape[0]).to(self.accelerator.device),
                final_ego_position_x=data["projection_goal_position"][:, 0],
                final_ego_position_y=data["projection_goal_position"][:, 1],
                obstacle_positions_x=torch.tensor([])
                .to(self.accelerator.device)
                .unsqueeze(0)
                .tile(output.shape[0], -1),
                obstacle_positions_y=torch.tensor([])
                .to(self.accelerator.device)
                .unsqueeze(0)
                .tile(output.shape[0], -1),
                obstacle_velocities_x=torch.tensor([])
                .to(self.accelerator.device)
                .unsqueeze(0)
                .tile(output.shape[0], -1),
                obstacle_velocities_y=torch.tensor([])
                .to(self.accelerator.device)
                .unsqueeze(0)
                .tile(output.shape[0], -1),
            )

        output_trajectory = self.projection_guidance.coefficients_to_trajectory(
            coefficients_x=output[:, 0, :],
            coefficients_y=output[:, 1, :],
            position_only=True,
        )
        output_trajectory = torch.stack((output_trajectory[0], output_trajectory[1]), dim=1)

        return output_trajectory, embedding, quantized_embedding

    def loss(self, output: Tuple[Tensor, ...], data: Dict[str, Tensor]) -> Tensor:
        output_trajectory, embedding, quantized_embedding = output
        trajectory = self.projection_guidance.coefficients_to_trajectory(
            coefficients_x=data["coefficients"][:, 0, :],
            coefficients_y=data["coefficients"][:, 1, :],
            position_only=True,
        )
        trajectory = torch.stack((trajectory[0], trajectory[1]), dim=1)
        return self.model.loss(
            output_trajectory,
            embedding=embedding,
            quantized_embedding=quantized_embedding,
            target=trajectory,
            beta=1e-5,
        )
