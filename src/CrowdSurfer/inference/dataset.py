from typing import Dict

import numpy as np
from torch import Tensor, device

from configuration import CoefficientConfiguration, DatasetConfiguration, ProjectionConfiguration
from navigation.constants import DataDirectories, DataKeys
from navigation.dataset import OverallDataset


class InferenceDataset(OverallDataset):
    def __init__(self, dataset_configuration: DatasetConfiguration, projection_configuration: ProjectionConfiguration):
        super().__init__(
            dataset_configuration=dataset_configuration,
            projection_configuration=projection_configuration,
            type=DataDirectories.INFERENCE,
        )

    def get_bag_length(self, bag_index: int) -> int:
        return self.data_lengths[bag_index]

    def get_all_odometry_for_plotting(self, bag_index: int) -> np.ndarray:
        bag_file = self._get_bag_file(bag_index)
        return bag_file[DataKeys.ODOMETRY]

    def get_all_coefficients_for_plotting(
        self, bag_index: int, coefficient_configuration: CoefficientConfiguration
    ) -> np.ndarray:
        bag_file = self._get_bag_file(bag_index)

        if coefficient_configuration is CoefficientConfiguration.BEST_EXPERT:
            return bag_file[DataKeys.BEST_EXPERT_COEFFICIENTS]
        elif coefficient_configuration is CoefficientConfiguration.BEST_PRIEST:
            return bag_file[DataKeys.BEST_PRIEST_COEFFICIENTS]
        elif coefficient_configuration is CoefficientConfiguration.ELITE_EXPERT:
            return bag_file[DataKeys.ELITE_EXPERT_COEFFICIENTS]
        elif coefficient_configuration is CoefficientConfiguration.ELITE_PRIEST:
            return bag_file[DataKeys.ELITE_PRIEST_COEFFICIENTS]

    def get_data(self, bag_index: int, timestep: int, device: device) -> Dict[str, Tensor]:
        bag_file = self._get_bag_file(bag_index)
        trajectory_idx = timestep
        coefficient = (CoefficientConfiguration.BEST_PRIEST, 0)

        return {
            key: value.unsqueeze(0).to(device)
            for key, value in {
                **self._get_odometry_data(bag_file, trajectory_idx),
                **self._get_trajectory_data(bag_file, trajectory_idx, coefficient),
                **self._get_observation_data(bag_file, trajectory_idx),
                **self._get_trajectory_projection_data(bag_file, trajectory_idx),
                **self._get_obstacle_projection_data(bag_file, trajectory_idx),
            }.items()
        }

    def _get_obstacle_visualization_data(
        self, bag_file: Dict[str, np.ndarray], trajectory_idx: int
    ) -> Dict[str, np.ndarray]:
        point_cloud = np.transpose(bag_file[DataKeys.POINT_CLOUD][trajectory_idx])  # shape: (2, num_points)
        np.where(np.isnan(point_cloud), self.projection_padding_value, point_cloud)

        dynamic_obstacles = np.transpose(
            bag_file[DataKeys.DYNAMIC_OBSTACLES][trajectory_idx]
        )  # shape: (5, num_obstacles)
        np.where(np.isnan(dynamic_obstacles), self.projection_padding_value, dynamic_obstacles)

        obstacle_positions = np.zeros((2, self.num_projection_dynamic_obstacles))  # shape: (2, total_obstacles)
        obstacle_positions[:, : min(self.num_projection_dynamic_obstacles, dynamic_obstacles.shape[1])] = (
            dynamic_obstacles[1:3, : self.num_projection_dynamic_obstacles]
        )

        obstacle_velocities = np.zeros((2, self.num_projection_dynamic_obstacles))
        obstacle_velocities[:, : min(self.num_projection_dynamic_obstacles, dynamic_obstacles.shape[1])] = (
            dynamic_obstacles[3:5, : self.num_projection_dynamic_obstacles]
        )

        return {
            "point_cloud": point_cloud,
            "dynamic_obstacle_positions": obstacle_positions,
            "dynamic_obstacle_velocities": obstacle_velocities,
        }

    def get_plotting_data(self, bag_index: int, timestep: int) -> Dict[str, np.ndarray]:
        bag_file = self._get_bag_file(bag_index)
        trajectory_idx = timestep
        coefficient = (CoefficientConfiguration.BEST_PRIEST, 0)

        output = {
            key: value.numpy()
            for key, value in {
                **self._get_odometry_data(bag_file, trajectory_idx),
                **self._get_trajectory_data(bag_file, trajectory_idx, coefficient),
                **self._get_trajectory_projection_data(bag_file, trajectory_idx),
            }.items()
        }
        output.update(self._get_obstacle_visualization_data(bag_file, trajectory_idx))
        return output
