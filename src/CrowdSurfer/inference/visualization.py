import os

import numpy as np
import torch
from tqdm import tqdm

import configuration
from navigation.projection_guidance import ProjectionGuidance

from .dataset import InferenceDataset
from .plotting import plot_trajectories


class Visualization:
    def __init__(self, configuration: configuration.Configuration):
        self.dataset = InferenceDataset(
            dataset_configuration=configuration.dataset,
            projection_configuration=configuration.projection,
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.projection_guidance = ProjectionGuidance(
            num_obstacles=0,
            num_timesteps=configuration.dataset.trajectory_length,
            total_time=configuration.dataset.trajectory_time,
            obstacle_ellipse_semi_major_axis=0.5,
            obstacle_ellipse_semi_minor_axis=0.5,
            max_projection_iterations=2,
            device=self.device,
        )
        self.coefficient_configuration = configuration.dataset.coefficient_configuration[0]

    def run(self, bag_index: int) -> None:
        coefficients = self.dataset.get_all_coefficients_for_plotting(
            bag_index, self.coefficient_configuration
        )  # shape: (num_timesteps, [num_elites], 11, 2)
        if coefficients.ndim == 4:
            reshaped_coefficients = coefficients.reshape(-1, 11, 2)
            trajectories_x, trajectories_y = self.projection_guidance.coefficients_to_trajectory(
                coefficients_x=torch.tensor(reshaped_coefficients[:, :, 0], dtype=torch.float32).to(self.device),
                coefficients_y=torch.tensor(reshaped_coefficients[:, :, 1], dtype=torch.float32).to(self.device),
                position_only=True,
            )
            trajectories = np.stack((trajectories_x.cpu().numpy(), trajectories_y.cpu().numpy()), axis=-1).reshape(
                coefficients.shape[0], coefficients.shape[1], -1, 2
            )  # shape: (num_timesteps, num_elites, 30, 2)
        else:
            trajectories_x, trajectories_y = self.projection_guidance.coefficients_to_trajectory(
                coefficients_x=torch.tensor(coefficients[:, :, 0], dtype=torch.float32).to(self.device),
                coefficients_y=torch.tensor(coefficients[:, :, 1], dtype=torch.float32).to(self.device),
                position_only=True,
            )
            trajectories = np.stack(
                (trajectories_x.cpu().numpy(), trajectories_y.cpu().numpy()), axis=-1
            )  # shape: (num_timesteps, 30, 2)

        os.makedirs("visualization", exist_ok=True)

        plot_trajectories(
            trajectories=trajectories,
            scores=None,
            dataset=self.dataset,
            bag_index=bag_index,
            save_directory="visualization",
        )

    def run_all(self) -> None:
        for bag_index in tqdm(range(len(self.dataset.data_lengths))):
            self.run(bag_index)
