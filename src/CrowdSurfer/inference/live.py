import numpy as np
import open3d as o3d
import torch

from configuration import Configuration, Mode

from .pipeline import InferenceData, Pipeline
import matplotlib.pyplot as plt


class LivePipeline(Pipeline):
    def __init__(self, configuration: Configuration):
        assert configuration.mode == Mode.LIVE
        super().__init__(configuration)
        self.time_horizon = configuration.live.time_horizon
        self.threshold_distance = configuration.live.threshold_distance
        self.padding_obstacle = configuration.live.padding

    def _generate_occupancy_map(self, pcd: np.ndarray, map_height=60, map_width=60, resolution=0.1):
        occupancy_map = np.zeros((map_height, map_width), dtype=np.int8)
        origin = (map_width * resolution / 2, map_height * resolution / 2)
        print(pcd.shape)
        x = pcd[:, 0]
        y = pcd[:, 1]

        # Transformation to map frame
        map_x = np.round((x + origin[0]) / resolution).astype(int)
        map_y = np.round((y + origin[1]) / resolution).astype(int)

        valid_points = (map_x >= 0) & (map_x < map_width) & (map_y >= 0) & (map_y < map_height)
        map_x = map_x[valid_points]
        map_y = map_y[valid_points]

        print(x, y)
        if len(map_x) > 0:
            occupancy_map[map_y, map_x] = 100  # Occupied

        # PLACEHOLDER ego-agent position on map
        ego_x = int(map_width / 2)
        ego_y = int(map_height / 2)
        occupancy_map[ego_y, ego_x] = 50  # Ego-agent

        plt.imsave("/home/laksh/crowdsurfer_ws/outputs/test/occu.png", occupancy_map, cmap="gray")

        return occupancy_map

    def _process_point_cloud(self, point_cloud: np.ndarray):
        output_point_cloud = np.zeros((2, self.max_projection_static_obstacles))

        if point_cloud.shape[0] == 0 or point_cloud.shape[1] == 0:
            return output_point_cloud

        # output_point_cloud = point_cloud[:, :2][np.argsort(np.linalg.norm(point_cloud, axis=-1), axis=-1)][
        #     : self.max_projection_static_obstacles, :2
        # ]
        # print(output_point_cloud[:, :5])

        # return output_point_cloud.T

        pcd = o3d.geometry.PointCloud()
        # print(np.array(static_obstacles).shape)
        static_obstacles = np.stack((point_cloud[:, 0], point_cloud[:, 1], np.zeros_like(point_cloud[:, 0])))
        pcd.points = o3d.utility.Vector3dVector(static_obstacles.T)
        downsampled_pcd = pcd.voxel_down_sample(voxel_size=0.16)
        pcd_array = np.asarray(downsampled_pcd.points)
        pcd_array = pcd_array[:, :2][np.argsort(np.linalg.norm(pcd_array, axis=-1), axis=-1)][
            : self.max_projection_static_obstacles
        ]
        return pcd_array.T

    def run(
        self,
        goal: np.ndarray,
        ego_velocity: np.ndarray,
        ego_acceleration: np.ndarray,
        point_cloud: np.ndarray,
        dynamic_obstacles_n_steps: np.ndarray,
    ):
        occupancy_map = self._generate_occupancy_map(point_cloud)

        point_cloud = self._process_point_cloud(point_cloud)

        goal_position = torch.from_numpy(np.array(goal))

        obstacle_positions_for_projection = np.full(
            (
                2,
                self.max_projection_dynamic_obstacles + self.max_projection_static_obstacles,
            ),
            self.padding_obstacle,
        )

        obstacle_positions_for_projection[
            :,
            self.max_projection_dynamic_obstacles : self.max_projection_dynamic_obstacles + point_cloud.shape[1],
        ] = point_cloud

        # print(dynamic_obstacles_n_steps.shape[-1])

        obstacle_positions_for_projection[:, 0 : dynamic_obstacles_n_steps.shape[-1]] = dynamic_obstacles_n_steps[
            -1, 0:2, :
        ]

        obstacle_velocities_for_projection = np.zeros(
            (
                2,
                self.max_projection_dynamic_obstacles + self.max_projection_static_obstacles,
            )
        )

        obstacle_velocities_for_projection[:, 0 : dynamic_obstacles_n_steps.shape[-1]] = dynamic_obstacles_n_steps[
            -1, 2:, :
        ]

        data = InferenceData(
            static_obstacles=torch.from_numpy(occupancy_map)
            .unsqueeze(0)
            .unsqueeze(0)
            .type(torch.float32)
            .to(self.device),
            dynamic_obstacles=torch.from_numpy(np.array(dynamic_obstacles_n_steps))
            .unsqueeze(0)
            .type(torch.float32)
            .to(self.device),
            heading_to_goal=torch.atan2(goal_position[1], goal_position[0])
            .unsqueeze(0)
            .type(torch.float32)
            .to(self.device),
            ego_velocity_for_projection=torch.from_numpy(ego_velocity).unsqueeze(0).type(torch.float32).to(self.device),
            ego_acceleration_for_projection=torch.from_numpy(ego_acceleration)
            .unsqueeze(0)
            .type(torch.float32)
            .to(self.device),
            goal_position_for_projection=torch.from_numpy(goal).unsqueeze(0).type(torch.float32).to(self.device),
            obstacle_positions_for_projection=torch.from_numpy(obstacle_positions_for_projection)
            .unsqueeze(0)
            .type(torch.float32)
            .to(self.device),
            obstacle_velocities_for_projection=torch.from_numpy(obstacle_velocities_for_projection)
            .unsqueeze(0)
            .type(torch.float32)
            .to(self.device),
        )

        # for key, value in data.__dict__.items():
        #     print(key, value.shape)

        coefficients, trajectories, scores = self._run_pipeline(data)

        return (
            coefficients.detach().cpu().numpy(),
            trajectories.detach().cpu().numpy(),
            scores.detach().cpu().numpy(),
        )
