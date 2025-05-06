from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from accelerate import load_checkpoint_in_model
from torch import Tensor

import configuration
from configuration import GuidanceType
from navigation.model import VQVAE, CombinedPixelCNN, ScoringNetwork
from navigation.projection_guidance import ProjectionGuidance
from navigation.utilities import project_coefficients


@dataclass
class InferenceData:
    """
    Dataclass to hold the input data for the pipeline. All data is in Ego frame.

    Attributes:
        static_obstacles: Tensor of shape (batch_size, 1, height, width) or (batch_size, 2, max_points)
            if static_obstacle_type is OCCUPANCY_MAP [DEFAULT] or DOWNSAMPLED_POINT_CLOUD
        dynamic_obstacles: Tensor of shape (batch_size, num_previous_timesteps, 4, max_obstacles)
            where num_previous_timesteps is 5 by default, and 4 channels corresponds to [x, y, vx, vy]
        heading_to_goal: Tensor of shape (batch_size, 1)
            where 1 corresponds to the heading angle in radians
        ego_velocity_for_projection: Tensor of shape (batch_size, 2)
            where 2 corresponds to [vx, vy]
        goal_position_for_projection: Tensor of shape (batch_size, 2)
            where 2 corresponds to [x, y]
        obstacle_positions_for_projection: Tensor of shape (batch_size, 2, max_projection_dynamic_obstacles + max_projection_static_obstacles)
        obstacle_velocities_for_projection: Tensor of shape (batch_size, 2, max_projection_dynamic_obstacles + max_projection_static_obstacles)
            where max_projection_dynamic_obstacles and max_projection_static_obstacles are 10 and 50 by default respectively.
    """

    static_obstacles: Tensor
    dynamic_obstacles: Tensor
    heading_to_goal: Tensor
    ego_velocity_for_projection: Tensor
    goal_position_for_projection: Tensor
    obstacle_positions_for_projection: Tensor
    obstacle_velocities_for_projection: Tensor
    ego_acceleration_for_projection: Tensor = None

    def __post_init__(self):
        if self.ego_acceleration_for_projection is None:
            self.ego_acceleration_for_projection = torch.zeros_like(
                self.ego_velocity_for_projection
            )


class Pipeline:
    def __init__(self, configuration: configuration.Configuration):
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
        self.num_samples = configuration.scoring_network.num_samples
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.vqvae = VQVAE(
            num_embeddings=configuration.vqvae.num_embeddings,
            embedding_dim=configuration.vqvae.embedding_dim,
            hidden_channels=configuration.vqvae.hidden_channels,
        ).to(self.device)
        self.pixelcnn = CombinedPixelCNN(
            num_embeddings=configuration.vqvae.num_embeddings,
            vqvae_hidden_channels=configuration.vqvae.hidden_channels,
            observation_embedding_dim=configuration.pixelcnn.observation_embedding_dim,
            static_obstacle_type=configuration.dataset.static_obstacle_type,
        ).to(self.device)

        assert configuration.vqvae.checkpoint_path is not None
        assert configuration.pixelcnn.checkpoint_path is not None

        load_checkpoint_in_model(self.vqvae, configuration.vqvae.checkpoint_path)
        load_checkpoint_in_model(self.pixelcnn, configuration.pixelcnn.checkpoint_path)

        self.vqvae.eval()
        self.pixelcnn.eval()

        if configuration.scoring_network.checkpoint_path is not None:
            self.scoring_network = ScoringNetwork(
                obstacle_embedding_dim=configuration.pixelcnn.observation_embedding_dim,
            ).to(self.device)
            load_checkpoint_in_model(
                self.scoring_network, configuration.scoring_network.checkpoint_path
            )
            self.scoring_network.eval()
        else:
            self.scoring_network = None

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
            device=self.device,
        )

        if self.guidance_type is GuidanceType.PRIEST:
            from navigation.priest_guidance import PriestPlanner

            self.priest_planner = PriestPlanner(
                num_dynamic_obstacles=self.max_projection_dynamic_obstacles,
                num_static_obstacles=self.max_projection_static_obstacles,
                time_horizon=configuration.dataset.trajectory_time,
                trajectory_length=configuration.dataset.trajectory_length,
                tracking_weight=1.0,
                # weight_smoothness=0.8,
                static_obstacle_semi_minor_axis=0.8,
                static_obstacle_semi_major_axis=0.8,
                dynamic_obstacle_semi_minor_axis=0.8,
                dynamic_obstacle_semi_major_axis=0.8,
                num_waypoints=configuration.dataset.trajectory_length,
                trajectory_batch_size=self.num_samples,
                max_outer_iterations=configuration.projection.num_priest_iterations,
            )

    def _get_probability_distribution_and_embedding_from_pixelcnn(
        self, data: InferenceData
    ) -> Tuple[Tensor, Tensor]:
        with torch.no_grad():
            pixelcnn_output, observation_embedding = self.pixelcnn.forward(
                static_obstacles=data.static_obstacles,
                dynamic_obstacles=data.dynamic_obstacles,
                heading_to_goal=data.heading_to_goal,
            )
        return (
            torch.exp(
                torch.log_softmax(pixelcnn_output, dim=-1)
            ),  # shape: (batch_size, vqvae_hidden_channels, vqvae_num_embeddings)
            observation_embedding,  # shape: (batch_size, observation_embedding_dim)
        )

    def _sample_from_vqvae(self, probability_distribution: Tensor) -> Tensor:
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
        self, coefficients: Tensor, data: InferenceData
    ) -> Tensor:
        return project_coefficients(
            projection_guidance=self.projection_guidance,
            coefficients=coefficients,
            initial_ego_position_x=torch.zeros_like(
                data.ego_velocity_for_projection[:, 0]
            )
            .to(self.device)
            .tile(self.num_samples),
            initial_ego_position_y=torch.zeros_like(
                data.ego_velocity_for_projection[:, 1]
            )
            .to(self.device)
            .tile(self.num_samples),
            initial_ego_velocity_x=data.ego_velocity_for_projection[:, 0].tile(
                self.num_samples
            ),
            initial_ego_velocity_y=data.ego_velocity_for_projection[:, 1].tile(
                self.num_samples
            ),
            initial_ego_acceleration_x=data.ego_acceleration_for_projection[:, 0].tile(
                self.num_samples
            ),
            initial_ego_acceleration_y=data.ego_acceleration_for_projection[:, 1].tile(
                self.num_samples
            ),
            final_ego_position_x=data.goal_position_for_projection[:, 0].tile(
                self.num_samples
            ),
            final_ego_position_y=data.goal_position_for_projection[:, 1].tile(
                self.num_samples
            ),
            obstacle_positions_x=(
                data.obstacle_positions_for_projection[:, 0].tile(self.num_samples, 1)
                if self.use_obstacle_constraints_for_guidance
                else None
            ),
            obstacle_positions_y=(
                data.obstacle_positions_for_projection[:, 1].tile(self.num_samples, 1)
                if self.use_obstacle_constraints_for_guidance
                else None
            ),
            obstacle_velocities_x=(
                data.obstacle_velocities_for_projection[:, 0].tile(self.num_samples, 1)
                if self.use_obstacle_constraints_for_guidance
                else None
            ),
            obstacle_velocities_y=(
                data.obstacle_velocities_for_projection[:, 1].tile(self.num_samples, 1)
                if self.use_obstacle_constraints_for_guidance
                else None
            ),
        )  # shape: (batch_size * num_samples, 2, 11)

    def _run_priest(self, coefficients: Tensor, data: InferenceData) -> Tensor:
        assert coefficients.shape[0] == self.num_samples

        obstacle_positions = data.obstacle_positions_for_projection[0].cpu().numpy()

        obstacle_velocities = (
            data.obstacle_velocities_for_projection[
                0, :, : self.max_projection_dynamic_obstacles
            ]
            .cpu()
            .numpy()
        )

        _, _, _, _, c_x_elite, c_y_elite, _, _, idx_min = (
            self.priest_planner.run_optimization(
                initial_x_position=0.0,
                initial_y_position=0.0,
                initial_x_velocity=data.ego_velocity_for_projection[0, 0]
                .cpu()
                .numpy()
                .item(),
                initial_y_velocity=data.ego_velocity_for_projection[0, 1]
                .cpu()
                .numpy()
                .item(),
                initial_x_acceleration=data.ego_acceleration_for_projection[0, 0]
                .cpu()
                .numpy()
                .item(),
                initial_y_acceleration=data.ego_acceleration_for_projection[0, 1]
                .cpu()
                .numpy()
                .item(),
                goal_x_position=data.goal_position_for_projection[0, 0]
                .cpu()
                .numpy()
                .item(),
                goal_y_position=data.goal_position_for_projection[0, 1]
                .cpu()
                .numpy()
                .item(),
                dynamic_obstacle_x_positions=obstacle_positions[
                    0, : self.max_projection_dynamic_obstacles
                ]
                if self.use_obstacle_constraints_for_guidance
                else None,
                dynamic_obstacle_y_positions=obstacle_positions[
                    1, : self.max_projection_dynamic_obstacles
                ]
                if self.use_obstacle_constraints_for_guidance
                else None,
                dynamic_obstacle_x_velocities=obstacle_velocities[0]
                if self.use_obstacle_constraints_for_guidance
                else None,
                dynamic_obstacle_y_velocities=obstacle_velocities[1]
                if self.use_obstacle_constraints_for_guidance
                else None,
                static_obstacle_x_positions=obstacle_positions[
                    0, self.max_projection_dynamic_obstacles :
                ]
                if self.use_obstacle_constraints_for_guidance
                else None,
                static_obstacle_y_positions=obstacle_positions[
                    1, self.max_projection_dynamic_obstacles :
                ]
                if self.use_obstacle_constraints_for_guidance
                else None,
                custom_x_coefficients=coefficients[:, 0, :].cpu().numpy(),
                custom_y_coefficients=coefficients[:, 1, :].cpu().numpy(),
            )
        )

        return (
            torch.stack(
                (torch.tensor(np.array(c_x_elite)), torch.tensor(np.array(c_y_elite))),
                dim=1,
            ).to(self.device),  # shape: (batch_size * num_samples, 2, 11)
            int(idx_min),
        )

    def _run_pipeline(self, data: InferenceData) -> Tuple[Tensor, Tensor, Tensor]:
        probability_distribution, observation_embedding = (
            self._get_probability_distribution_and_embedding_from_pixelcnn(data)
        )
        coefficients = self._sample_from_vqvae(probability_distribution)

        if self.guidance_type is GuidanceType.PROJECTION:
            coefficients = self._run_projection_guidance(coefficients, data)
        elif self.guidance_type is GuidanceType.PRIEST:
            coefficients, idx_min = self._run_priest(coefficients, data)

        trajectory_x, trajectory_y = (
            self.projection_guidance.coefficients_to_trajectory(
                coefficients[:, 0, :], coefficients[:, 1, :], position_only=True
            )
        )
        trajectory = torch.stack((trajectory_x, trajectory_y), dim=1)

        if self.scoring_network is not None:
            with torch.no_grad():
                scores = self.scoring_network.forward(
                    coefficients.unflatten(0, (-1, self.num_samples)),
                    condition=observation_embedding,
                ).squeeze()  # shape: (num_samples)
            scores = torch.exp(torch.log_softmax(scores, dim=0))
        else:
            scores = torch.zeros(self.num_samples)
            scores[int(idx_min)] = 1.0

        return (
            coefficients.transpose(-1, -2).cpu(),  # shape: (num_samples, 11, 2)
            trajectory.transpose(
                -1, -2
            ).cpu(),  # shape: (num_samples, trajectory_time_steps, 2)
            scores.cpu(),  # shape: (num_samples)
        )
