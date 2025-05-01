import os
from typing import Dict, List, Tuple, cast

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

import configuration
from configuration import CoefficientConfiguration, StaticObstacleType

from ..constants import DataDirectories, DataKeys


class OverallDataset(Dataset):
    def __init__(
        self,
        dataset_configuration: configuration.DatasetConfiguration,
        projection_configuration: configuration.ProjectionConfiguration,
        type: str = DataDirectories.TRAIN,
    ):
        self.directory = os.path.join(dataset_configuration.directory, type)

        self.trajectory_length = dataset_configuration.trajectory_length

        self.coefficient_configuration = dataset_configuration.coefficient_configuration
        self.num_coefficients = {
            CoefficientConfiguration.BEST_EXPERT: 1,
            CoefficientConfiguration.BEST_PRIEST: 1,
            CoefficientConfiguration.ELITE_EXPERT: dataset_configuration.num_elite_coefficients,
            CoefficientConfiguration.ELITE_PRIEST: dataset_configuration.num_elite_coefficients,
        }

        self.total_coefficients = sum(
            self.num_coefficients[coefficient_configuration]
            for coefficient_configuration in self.coefficient_configuration
        )

        self.num_observation_time_steps = 5
        self.padding_value = dataset_configuration.padding
        self.static_obstacle_type = dataset_configuration.static_obstacle_type

        self.projection_padding_value = projection_configuration.padding
        self.num_projection_dynamic_obstacles = (
            projection_configuration.max_dynamic_obstacles
        )
        self.num_projection_static_obstacles = (
            projection_configuration.max_static_obstacles
        )

        self.data_lengths: List[int] = []
        bag_file_names: List[str] = []

        index_file = os.path.join(self.directory, "index.txt")
        with open(index_file, "r") as f:
            for line in f:
                split_line = line.split()
                self.data_lengths.append(int(split_line[2]))
                bag_file_names.append(split_line[0])

        self.memmap_files: List[Dict[str, np.ndarray]] = []

        for bag_file_name in bag_file_names:
            bag_file_directory = os.path.join(self.directory, bag_file_name)
            npy_files = os.listdir(bag_file_directory)
            mmap_npy_files = {}
            for npy_file in npy_files:
                mmap_npy_files[npy_file[:-4]] = np.load(
                    os.path.join(bag_file_directory, npy_file),
                    mmap_mode="r",
                )
            self.memmap_files.append(mmap_npy_files)
        self.total_length = sum(self.data_lengths)

    def __len__(self):
        return self.total_length * self.total_coefficients

    def _get_indices(self, idx: int):
        if idx < 0 or idx >= len(self):
            raise IndexError

        # Find the bag file that contains the trajectory at index idx (where idx is the index of the coefficients), Each bag file contains N trajectories each with either 1 or 80
        bag_idx = 0
        trajectory_idx = (
            idx // self.total_coefficients
        )  # Get the index of the trajectory within the bag file

        coefficient_idx = (
            idx % self.total_coefficients
        )  # Get the index of the coefficient within the trajectory

        for coefficient_configuration in self.coefficient_configuration:
            num_coefficients = self.num_coefficients[coefficient_configuration]
            if coefficient_idx < num_coefficients:
                coefficient_configuration = coefficient_configuration
                break
            coefficient_idx -= num_coefficients

        while trajectory_idx >= self.data_lengths[bag_idx]:
            trajectory_idx -= self.data_lengths[bag_idx]
            bag_idx += 1

        return bag_idx, trajectory_idx, (coefficient_configuration, coefficient_idx)

    def _get_bag_file(self, bag_idx: int):
        # return np.load(os.path.join(self.directory, f"{bag_idx:08d}.npz"))
        return self.memmap_files[bag_idx]

    def _get_trajectory_data(
        self,
        bag_file: Dict[str, np.ndarray],
        trajectory_idx: int,
        coefficient: Tuple[CoefficientConfiguration, int],
    ) -> Dict[str, Tensor]:
        if coefficient[0] is CoefficientConfiguration.BEST_EXPERT:
            coefficients = torch.tensor(
                bag_file[DataKeys.BEST_EXPERT_COEFFICIENTS][trajectory_idx],
                dtype=torch.float32,
            ).transpose(-1, -2)
        elif coefficient[0] is CoefficientConfiguration.BEST_PRIEST:
            coefficients = torch.tensor(
                bag_file[DataKeys.BEST_PRIEST_COEFFICIENTS][trajectory_idx],
                dtype=torch.float32,
            ).transpose(-1, -2)
        elif coefficient[0] is CoefficientConfiguration.ELITE_EXPERT:
            coefficients = torch.tensor(
                bag_file[DataKeys.ELITE_EXPERT_COEFFICIENTS][trajectory_idx][
                    coefficient[1]
                ],
                dtype=torch.float32,
            ).transpose(-1, -2)
        elif coefficient[0] is CoefficientConfiguration.ELITE_PRIEST:
            coefficients = torch.tensor(
                bag_file[DataKeys.ELITE_PRIEST_COEFFICIENTS][trajectory_idx][
                    coefficient[1]
                ],
                dtype=torch.float32,
            ).transpose(-1, -2)

        # Translate coefficients to ensure the first timestep is at the origin
        coefficients -= coefficients[:, 0].unsqueeze(-1).clone()

        # Replace nans with zeros
        coefficients[torch.isnan(coefficients)] = 0.0

        expert_trajectory = torch.tensor(
            bag_file[DataKeys.EXPERT_TRAJECTORY][trajectory_idx],
            dtype=torch.float32,
        ).transpose(-1, -2)

        if expert_trajectory.shape[-1] != self.trajectory_length:
            expert_trajectory = cast(
                Tensor,
                torch.nn.functional.interpolate(
                    expert_trajectory.unsqueeze(0),
                    size=self.trajectory_length,
                    mode="linear",
                    align_corners=False,
                ),
            ).squeeze(0)  # shape: (2, trajectory_length)

        return {
            "coefficients": coefficients,
            "expert_trajectory": expert_trajectory,
        }

    def _get_trajectory_projection_data(
        self, bag_file: Dict[str, np.ndarray], trajectory_idx: int
    ) -> Dict[str, Tensor]:
        ego_velocity = torch.tensor(
            bag_file[DataKeys.EGO_VELOCITY][trajectory_idx],
            dtype=torch.float32,
        )

        goal_position = torch.tensor(
            bag_file[DataKeys.GOAL_POSITION][trajectory_idx],
            dtype=torch.float32,
        )

        return {
            "projection_ego_velocity": ego_velocity,
            "projection_goal_position": goal_position,
        }

    def _get_obstacle_projection_data(
        self, bag_file: Dict[str, np.ndarray], trajectory_idx: int
    ) -> Dict[str, Tensor]:
        point_cloud = torch.tensor(
            bag_file[DataKeys.POINT_CLOUD][trajectory_idx],
            dtype=torch.float32,
        ).transpose(-1, -2)  # shape: (2, num_points)
        point_cloud[torch.isnan(point_cloud)] = self.projection_padding_value

        dynamic_obstacles = torch.tensor(
            bag_file[DataKeys.DYNAMIC_OBSTACLES][trajectory_idx],
            dtype=torch.float32,
        ).transpose(-1, -2)  # shape: (5, num_obstacles)
        dynamic_obstacles[torch.isnan(dynamic_obstacles)] = (
            self.projection_padding_value
        )

        obstacle_positions = torch.zeros(
            2,
            self.num_projection_dynamic_obstacles
            + self.num_projection_static_obstacles,
        )  # shape: (2, total_obstacles)

        obstacle_positions[
            :, : min(self.num_projection_dynamic_obstacles, dynamic_obstacles.shape[1])
        ] = dynamic_obstacles[1:3, : self.num_projection_dynamic_obstacles]
        obstacle_positions[
            :,
            self.num_projection_dynamic_obstacles : self.num_projection_dynamic_obstacles
            + min(self.num_projection_static_obstacles, point_cloud.shape[1]),
        ] = point_cloud[:, : self.num_projection_static_obstacles]

        obstacle_velocities = torch.zeros(
            2,
            self.num_projection_dynamic_obstacles
            + self.num_projection_static_obstacles,
        )

        obstacle_velocities[
            :, : min(self.num_projection_dynamic_obstacles, dynamic_obstacles.shape[1])
        ] = dynamic_obstacles[3:5, : self.num_projection_dynamic_obstacles]

        return {
            "projection_obstacle_positions": obstacle_positions,
            "projection_obstacle_velocities": obstacle_velocities,
        }

    def _get_observation_data(
        self, bag_file: Dict[str, np.ndarray], trajectory_idx: int
    ) -> Dict[str, Tensor]:
        # Get the point cloud and dynamic obstacles for the current and previous timesteps
        # If the current timestep is the first timestep, the previous timestep is padded with zeros

        num_timesteps = self.num_observation_time_steps

        indices = np.arange(trajectory_idx - num_timesteps + 1, trajectory_idx + 1)
        invalid_indices = indices < 0
        indices[invalid_indices] = 0

        if self.static_obstacle_type is StaticObstacleType.POINT_CLOUD:
            static_obstacles = torch.tensor(
                bag_file[DataKeys.DOWNSAMPLED_POINT_CLOUD][trajectory_idx],
                dtype=torch.float32,
            ).transpose(-1, -2)  # shape: (2, num_points)
        elif self.static_obstacle_type is StaticObstacleType.OCCUPANCY_MAP:
            static_obstacles = torch.tensor(
                bag_file[DataKeys.RAY_TRACED_OCCUPANCY_MAP][trajectory_idx],
                dtype=torch.float32,
            ).unsqueeze(0)
            static_obstacles = torch.rot90(static_obstacles, 1, dims=(1, 2))
        else:
            raise ValueError(
                f"Invalid static obstacle type: {self.static_obstacle_type}"
            )

        dynamic_obstacles = torch.tensor(
            bag_file[DataKeys.DYNAMIC_OBSTACLES][indices],
            dtype=torch.float32,
        ).transpose(-1, -2)  # shape: (num_timesteps, 5, num_obstacles)
        dynamic_obstacles[torch.tensor(invalid_indices, dtype=torch.bool)] = (
            self.padding_value
        )

        # Replace invalid indices with padding value
        static_obstacles[torch.isnan(static_obstacles)] = self.padding_value
        dynamic_obstacles[torch.isnan(dynamic_obstacles)] = self.padding_value

        goal_position = torch.tensor(
            bag_file[DataKeys.GOAL_POSITION][trajectory_idx],
            dtype=torch.float32,
        )  # shape: (2,)

        heading_to_goal = torch.atan2(goal_position[1], goal_position[0])

        return {
            "static_obstacles": static_obstacles,
            "dynamic_obstacles": dynamic_obstacles[:, 1:],
            "heading_to_goal": heading_to_goal,
        }

    def _get_odometry_data(
        self, bag_file: Dict[str, np.ndarray], trajectory_idx: int
    ) -> Dict[str, Tensor]:
        odometry = torch.tensor(
            bag_file[DataKeys.ODOMETRY][trajectory_idx],
            dtype=torch.float32,
        )

        return {"odometry": odometry}

    def __getitem__(self, idx: int):
        bag_idx, trajectory_idx, coefficient = self._get_indices(idx)
        bag_file = self._get_bag_file(bag_idx)

        return {
            **self._get_odometry_data(bag_file, trajectory_idx),
            **self._get_trajectory_data(bag_file, trajectory_idx, coefficient),
            **self._get_observation_data(bag_file, trajectory_idx),
            **self._get_trajectory_projection_data(bag_file, trajectory_idx),
            **self._get_obstacle_projection_data(bag_file, trajectory_idx),
        }


class ScoringNetworkDataset(OverallDataset):
    def __init__(
        self,
        dataset_configuration: configuration.DatasetConfiguration,
        projection_configuration: configuration.ProjectionConfiguration,
        type: str = DataDirectories.TRAIN,
    ):
        super().__init__(
            dataset_configuration=dataset_configuration,
            projection_configuration=projection_configuration,
            type=type,
        )
        assert self.total_coefficients == 1, (
            "Scoring network does not support multiple or elite coefficients"
        )

    def __getitem__(self, idx) -> Dict[str, Tensor]:
        bag_idx, trajectory_idx, coefficient = self._get_indices(idx)
        bag_file = self._get_bag_file(bag_idx)

        return {
            **self._get_trajectory_data(bag_file, trajectory_idx, coefficient),
            **self._get_observation_data(bag_file, trajectory_idx),
            **self._get_trajectory_projection_data(bag_file, trajectory_idx),
            **self._get_obstacle_projection_data(bag_file, trajectory_idx),
        }


class PixelCNNDataset(OverallDataset):
    def __init__(
        self,
        dataset_configuration: configuration.DatasetConfiguration,
        projection_configuration: configuration.ProjectionConfiguration,
        type: str = DataDirectories.TRAIN,
    ):
        super().__init__(
            dataset_configuration=dataset_configuration,
            projection_configuration=projection_configuration,
            type=type,
        )

    def __getitem__(self, idx) -> Dict[str, Tensor]:
        bag_idx, trajectory_idx, coefficient = self._get_indices(idx)
        bag_file = self._get_bag_file(bag_idx)

        return {
            **self._get_trajectory_data(bag_file, trajectory_idx, coefficient),
            **self._get_observation_data(bag_file, trajectory_idx),
        }


class VQVAEDataset(OverallDataset):
    def __init__(
        self,
        dataset_configuration: configuration.DatasetConfiguration,
        projection_configuration: configuration.ProjectionConfiguration,
    ):
        super().__init__(
            dataset_configuration=dataset_configuration,
            projection_configuration=projection_configuration,
            type=DataDirectories.TRAIN,
        )

    def __getitem__(self, idx) -> Dict[str, Tensor]:
        bag_idx, trajectory_idx, coefficient = self._get_indices(idx)
        bag_file = self._get_bag_file(bag_idx)

        return {
            **self._get_trajectory_data(bag_file, trajectory_idx, coefficient),
            **self._get_trajectory_projection_data(bag_file, trajectory_idx),
        }


if __name__ == "__main__":
    pass
    # from tqdm import tqdm

    # Load the dataset
    # current_file_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # # dataset = VQVAEDataset(
    # #     os.path.join(current_file_dir, "data", "custom"), use_best_coefficients=False
    # # )
    # dataset = PixelCNNDataset(
    #     os.path.join(current_file_dir, "data", "custom", "validation"),
    #     use_best_coefficients=True,
    # )

    # data = dataset[5]
    # for d in data:
    #     # print(d)
    #     print(d.shape)
    # data = dataset[4000]

    # Get the first trajectory
    # print(len(dataset))
    # counter = 0
    # for data in tqdm(dataset):
    #     coefficients, trajectory, expert_trajectory, ego_velocity, goal_position = data
    #     if torch.sum(coefficients) == 0:
    #         counter += 1
    # print(counter)

    # print(
    #     coefficients.shape,
    #     trajectory.shape,
    #     torch.rad2deg(odometry[2]),
    #     ego_velocity.shape,
    #     goal_position,
    # )
