import os
from typing import Dict, Tuple

import configuration
import numpy as np
import torch
from accelerate import load_checkpoint_in_model
from configuration import GuidanceType, Mode
from navigation.constants import DataDirectories
from navigation.dataset import ScoringNetworkDataset
from navigation.model import VQVAE, CombinedPixelCNN, ScoringNetwork
from navigation.projection_guidance import ProjectionGuidance
from navigation.utilities import project_coefficients
from torch import Tensor

from .trainer import Trainer, TrainerMode


class ScoringNetworkTrainer(Trainer):
    def __init__(
        self,
        configuration: configuration.Configuration,
    ):
        self.num_samples = configuration.scoring_network.num_samples
        self.guidance_type = configuration.projection.guidance_type
        self.max_projection_dynamic_obstacles = (
            configuration.projection.max_dynamic_obstacles
        )
        self.max_projection_static_obstacles = (
            configuration.projection.max_static_obstacles
        )
        self.use_obstacle_constraints_for_guidance = (
            configuration.projection.use_obstacle_constraints
        )
        super().__init__(
            model=ScoringNetwork(
                obstacle_embedding_dim=configuration.pixelcnn.observation_embedding_dim,
            ),
            dataset=ScoringNetworkDataset(
                dataset_configuration=configuration.dataset,
                projection_configuration=configuration.projection,
                type=DataDirectories.TRAIN,
            ),
            results_directory=configuration.trainer.results_directory,
            learning_rate=configuration.trainer.learning_rate,
            batch_size=configuration.trainer.batch_size,
            mode=TrainerMode.TRAIN
            if configuration.mode is Mode.TRAIN_SCORING_NETWORK
            else TrainerMode.INFERENCE,
            validation_dataset=ScoringNetworkDataset(
                dataset_configuration=configuration.dataset,
                projection_configuration=configuration.projection,
                type=DataDirectories.VALIDATION,
            )
            if os.path.exists(
                os.path.join(
                    configuration.dataset.directory, DataDirectories.VALIDATION
                )
            )
            else None,
            dataloader_num_workers=configuration.trainer.dataloader_num_workers,
            dataloader_pin_memory=configuration.trainer.dataloader_pin_memory,
        )

        self.model: ScoringNetwork
        self.dataset: ScoringNetworkDataset
        self.validation_dataset: ScoringNetworkDataset

        self.projection_guidance = ProjectionGuidance(
            num_obstacles=0,
            num_timesteps=configuration.dataset.trajectory_length,
            total_time=configuration.dataset.trajectory_time,
            obstacle_ellipse_semi_major_axis=0,
            obstacle_ellipse_semi_minor_axis=0,
            max_projection_iterations=1,
            device=self.accelerator.device,
        )

        self.vqvae = VQVAE(
            num_embeddings=configuration.vqvae.num_embeddings,
            embedding_dim=configuration.vqvae.embedding_dim,
            hidden_channels=configuration.vqvae.hidden_channels,
            num_features=11,
        ).to(self.accelerator.device)

        self.pixelcnn = CombinedPixelCNN(
            num_embeddings=configuration.vqvae.num_embeddings,
            vqvae_hidden_channels=configuration.vqvae.hidden_channels,
            observation_embedding_dim=configuration.pixelcnn.observation_embedding_dim,
            static_obstacle_type=configuration.dataset.static_obstacle_type,
        ).to(self.accelerator.device)

        assert configuration.vqvae.checkpoint_path is not None
        assert configuration.pixelcnn.checkpoint_path is not None
        load_checkpoint_in_model(self.vqvae, configuration.vqvae.checkpoint_path)
        load_checkpoint_in_model(self.pixelcnn, configuration.pixelcnn.checkpoint_path)

        self.vqvae.eval()
        self.pixelcnn.eval()

        self.projection_guidance = ProjectionGuidance(
            num_obstacles=(
                self.max_projection_dynamic_obstacles
                + self.max_projection_static_obstacles
                if self.use_obstacle_constraints_for_guidance
                else 0
            ),
            num_timesteps=configuration.dataset.trajectory_length,
            total_time=configuration.dataset.trajectory_time,
            obstacle_ellipse_semi_major_axis=0.5,
            obstacle_ellipse_semi_minor_axis=0.5,
            max_projection_iterations=2,
            device=self.accelerator.device,
        )

        if self.guidance_type is GuidanceType.PRIEST:
            from navigation.priest_guidance import PriestPlanner

            self.priest_planner = PriestPlanner(
                num_dynamic_obstacles=self.max_projection_dynamic_obstacles,
                num_static_obstacles=self.max_projection_static_obstacles,
                time_horizon=configuration.dataset.trajectory_time,
                trajectory_length=configuration.dataset.trajectory_length,
                tracking_weight=1.2,
                num_waypoints=configuration.dataset.trajectory_length,
                trajectory_batch_size=self.num_samples,
                max_outer_iterations=configuration.projection.max_outer_iterations,
            )

    def _get_probability_distribution_and_embedding_from_pixelcnn(
        self, data: Dict[str, Tensor]
    ):
        with torch.no_grad():
            pixelcnn_output, observation_embedding = self.pixelcnn.forward(
                static_obstacles=data["static_obstacles"],
                dynamic_obstacles=data["dynamic_obstacles"],
                heading_to_goal=data["heading_to_goal"],
            )
        return (
            torch.exp(
                torch.log_softmax(pixelcnn_output, dim=-1)
            ),  # shape: (batch_size, vqvae_hidden_channels, vqvae_num_embeddings)
            observation_embedding,  # shape: (batch_size, observation_embedding_dim)
        )

    def _sample_from_vqvae(self, probability_distribution: Tensor):
        pixelcnn_indices = (
            torch.multinomial(
                probability_distribution.flatten(0, 1),
                num_samples=self.num_samples,
                replacement=True,
            )
            .unflatten(0, (-1, probability_distribution.shape[1]))
            .transpose(1, 2)
            .flatten(0, 1)
        )  # shape: (batch_size * num_samples, hidden_channels)

        with torch.no_grad():
            output_coefficients, _ = self.vqvae.decode_from_indices(pixelcnn_indices)

        return output_coefficients  # shape: (batch_size * num_samples, 2, 11)

    def _run_projection_guidance(
        self, coefficients: Tensor, data: Dict[str, Tensor]
    ) -> Tensor:
        return project_coefficients(
            projection_guidance=self.projection_guidance,
            coefficients=coefficients,
            initial_ego_position_x=torch.zeros_like(
                data["projection_ego_velocity"][:, 0]
            )
            .to(self.accelerator.device)
            .tile(self.num_samples),
            initial_ego_position_y=torch.zeros_like(
                data["projection_ego_velocity"][:, 0]
            )
            .to(self.accelerator.device)
            .tile(self.num_samples),
            initial_ego_velocity_x=data["projection_ego_velocity"][:, 0].tile(
                self.num_samples
            ),
            initial_ego_velocity_y=data["projection_ego_velocity"][:, 0].tile(
                self.num_samples
            ),
            initial_ego_acceleration_x=torch.zeros_like(
                data["projection_ego_velocity"][:, 0]
            )
            .to(self.accelerator.device)
            .tile(self.num_samples),
            initial_ego_acceleration_y=torch.zeros_like(
                data["projection_ego_velocity"][:, 0]
            )
            .to(self.accelerator.device)
            .tile(self.num_samples),
            final_ego_position_x=data["projection_goal_position"][:, 0].tile(
                self.num_samples
            ),
            final_ego_position_y=data["projection_goal_position"][:, 1].tile(
                self.num_samples
            ),
            obstacle_positions_x=(
                data["projection_obstacle_positions"][:, 0].tile(self.num_samples, 1)
                if self.use_obstacle_constraints_for_guidance
                else None
            ),
            obstacle_positions_y=(
                data["projection_obstacle_positions"][:, 1].tile(self.num_samples, 1)
                if self.use_obstacle_constraints_for_guidance
                else None
            ),
            obstacle_velocities_x=(
                data["projection_obstacle_velocities"][:, 0].tile(self.num_samples, 1)
                if self.use_obstacle_constraints_for_guidance
                else None
            ),
            obstacle_velocities_y=(
                data["projection_obstacle_velocities"][:, 1].tile(self.num_samples, 1)
                if self.use_obstacle_constraints_for_guidance
                else None
            ),
        )  # shape: (batch_size * num_samples, 2, 11)

    def _run_priest(self, coefficients: Tensor, data: Dict[str, Tensor]) -> Tensor:
        elites_x = []
        elites_y = []
        coefficients = coefficients.unflatten(0, (-1, self.num_samples))
        for i in range(coefficients.shape[0]):
            _, _, _, _, c_x_elite, c_y_elite, _, _, _ = (
                self.priest_planner.run_optimization(
                    initial_x_position=0,
                    initial_y_position=0,
                    initial_x_velocity=data["projection_ego_velocity"][0, 0]
                    .cpu()
                    .numpy()
                    .item(),
                    initial_y_velocity=data["projection_ego_velocity"][0, 1]
                    .cpu()
                    .numpy()
                    .item(),
                    initial_x_acceleration=0,
                    initial_y_acceleration=0,
                    goal_x_position=data["projection_goal_position"][0, 0]
                    .cpu()
                    .numpy()
                    .item(),
                    goal_y_position=data["projection_goal_position"][0, 1]
                    .cpu()
                    .numpy()
                    .item(),
                    dynamic_obstacle_x_positions=data["projection_obstacle_positions"][
                        0, 0, : self.max_projection_dynamic_obstacles
                    ]
                    .cpu()
                    .numpy()
                    if self.use_obstacle_constraints_for_guidance
                    else None,
                    dynamic_obstacle_y_positions=data["projection_obstacle_positions"][
                        0, 1, : self.max_projection_dynamic_obstacles
                    ]
                    .cpu()
                    .numpy()
                    if self.use_obstacle_constraints_for_guidance
                    else None,
                    dynamic_obstacle_x_velocities=data[
                        "projection_obstacle_velocities"
                    ][0, 0, : self.max_projection_dynamic_obstacles]
                    .cpu()
                    .numpy()
                    if self.use_obstacle_constraints_for_guidance
                    else None,
                    dynamic_obstacle_y_velocities=data[
                        "projection_obstacle_velocities"
                    ][0, 1, : self.max_projection_dynamic_obstacles]
                    .cpu()
                    .numpy()
                    if self.use_obstacle_constraints_for_guidance
                    else None,
                    static_obstacle_x_positions=data["projection_obstacle_positions"][
                        0, 0, self.max_projection_dynamic_obstacles :
                    ]
                    .cpu()
                    .numpy()
                    if self.use_obstacle_constraints_for_guidance
                    else None,
                    static_obstacle_y_positions=data["projection_obstacle_positions"][
                        0, 1, self.max_projection_dynamic_obstacles :
                    ]
                    .cpu()
                    .numpy()
                    if self.use_obstacle_constraints_for_guidance
                    else None,
                    custom_x_coefficients=coefficients[i, :, 0, :].cpu().numpy(),
                    custom_y_coefficients=coefficients[i, :, 1, :].cpu().numpy(),
                )
            )
            elites_x.append(c_x_elite)
            elites_y.append(c_y_elite)

        elites_x = np.stack(elites_x, axis=0).reshape((-1, 11))
        elites_y = np.stack(elites_y, axis=1).reshape((-1, 11))
        return torch.stack(
            (torch.tensor(np.array(elites_x)), torch.tensor(np.array(elites_y))),
            dim=1,
        ).to(self.accelerator.device)  # shape: (batch_size * num_samples, 2, 11)

    def forward(self, data: Dict[str, Tensor]) -> Tuple[Tensor, Tensor]:
        probability_distribution, observation_embedding = (
            self._get_probability_distribution_and_embedding_from_pixelcnn(data)
        )
        coefficients = self._sample_from_vqvae(
            probability_distribution
        )  # shape: (batch_size * num_samples, 2, 11)

        if self.guidance_type == GuidanceType.PROJECTION:
            coefficients = self._run_projection_guidance(coefficients, data)
        elif self.guidance_type == GuidanceType.PRIEST:
            coefficients = self._run_priest(coefficients, data)

        trajectory_x, trajectory_y = (
            self.projection_guidance.coefficients_to_trajectory(
                coefficients[:, 0, :], coefficients[:, 1, :], position_only=True
            )
        )
        trajectories = torch.stack(
            (trajectory_x, trajectory_y), dim=1
        )  # shape: (batch_size * num_samples, 2, num_trajectory_timesteps)

        return (
            self.model.forward(
                coefficients.unflatten(0, (-1, self.num_samples)),
                condition=observation_embedding,
            ),  # shape: (batch_size, num_samples)
            trajectories.unflatten(
                0, (-1, self.num_samples)
            ),  # shape: (batch_size * num_samples, 2, num_trajectory_timesteps)
        )

    def loss(self, output: Tuple[Tensor, Tensor], data: Dict[str, Tensor]) -> Tensor:
        prediction, trajectories = output
        return self.model.loss(
            prediction, target=data["expert_trajectory"], trajectories=trajectories
        )
