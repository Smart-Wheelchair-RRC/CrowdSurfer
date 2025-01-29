import os
from typing import Tuple

import numpy as np
from tqdm import tqdm

import configuration
from navigation.constants import DataDirectories

from .dataset import InferenceDataset
from .pipeline import InferenceData, Pipeline
from .plotting import plot_trajectories


class InferencePipeline(Pipeline):
    def __init__(self, configuration: configuration.Configuration):
        super().__init__(configuration)
        self.dataset = InferenceDataset(
            dataset_configuration=configuration.dataset,
            projection_configuration=configuration.projection,
        )

    def run(self, bag_index: int, plot: bool = False, save: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        coefficients_list = []
        trajectories = []
        scores = []

        # Create inference directory if not exists
        if save or plot:
            os.makedirs(DataDirectories.INFERENCE, exist_ok=True)

        for timestep in range(self.dataset.get_bag_length(bag_index)):
            data = self.dataset.get_data(bag_index, timestep, self.device)

            coefficients, trajectory, timestep_score = self._run_pipeline(
                InferenceData(
                    static_obstacles=data["static_obstacles"],
                    dynamic_obstacles=data["dynamic_obstacles"],
                    heading_to_goal=data["heading_to_goal"],
                    ego_velocity_for_projection=data["projection_ego_velocity"],
                    goal_position_for_projection=data["projection_goal_position"],
                    obstacle_positions_for_projection=data["projection_obstacle_positions"],
                    obstacle_velocities_for_projection=data["projection_obstacle_velocities"],
                )
            )

            coefficients_list.append(coefficients.numpy())
            trajectories.append(trajectory.numpy())
            scores.append(timestep_score.numpy())

        coefficients_list = np.stack(coefficients_list, axis=0)
        trajectories = np.stack(trajectories, axis=0)
        scores = np.stack(scores, axis=0)

        if save:
            np.save(
                f"inference/coefficients_{bag_index:08}.npy",
                coefficients_list,
            )
            np.save(
                f"inference/trajectories_{bag_index:08}.npy",
                trajectories,
            )
            np.save(
                f"inference/scores_{bag_index:08}.npy",
                scores,
            )

        if plot:
            plot_trajectories(
                trajectories=trajectories,
                scores=scores,
                dataset=self.dataset,
                bag_index=bag_index,
                save_directory=DataDirectories.INFERENCE,
            )

        return coefficients_list, trajectories, scores

    def run_all(self, plot: bool = False, save_arrays: bool = False) -> None:
        for bag_index in tqdm(range(len(self.dataset.data_lengths))):
            self.run(bag_index, plot=plot, save=save_arrays)
