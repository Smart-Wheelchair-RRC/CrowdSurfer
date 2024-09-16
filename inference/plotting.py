import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

from .dataset import InferenceDataset


def _translate_points(points: np.ndarray, translation: np.ndarray) -> np.ndarray:
    # points shape: (..., 2)
    # translation shape: (..., 2)

    return points + translation


def _rotate_points(points: np.ndarray, angle: np.ndarray) -> np.ndarray:
    # points shape: (..., 2)
    # angle shape: (...)

    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    rotation_matrix = np.stack((cos_theta, -sin_theta, sin_theta, cos_theta), axis=-1)
    rotation_matrix = np.reshape(rotation_matrix, (*angle.shape, 2, 2))

    return np.einsum("...ij,...j->...i", rotation_matrix, points)


def plot_trajectories(
    trajectories: np.ndarray,  # shape: (N, [num_elites], 30, 2)
    scores: Optional[np.ndarray],  # shape: (N, [num_elites])
    dataset: InferenceDataset,
    bag_index: int,
    save_directory: str,
):
    assert (
        trajectories.shape[0] == dataset.data_lengths[bag_index]
    ), f"Expected trajectories to have shape ({dataset.data_lengths[bag_index]}, [num_elites], 30, 2), but got {trajectories.shape}"

    if scores is not None:
        assert scores.shape[0] == dataset.data_lengths[bag_index], (
            f"Expected scores to have shape ({dataset.data_lengths[bag_index]}, [num_elites]), "
            f"but got {scores.shape}"
        )
        assert trajectories.ndim == 4, (
            "Expected scores to be provided only when multiple trajectories are provided, "
            f"but got scores with shape {scores.shape} and trajectories with shape {trajectories.shape}"
        )

    all_odometry = dataset.get_all_odometry_for_plotting(bag_index)

    fig, ax = plt.subplots(figsize=(12, 8))

    ax.set_xlim(-8, 8)
    ax.set_ylim(-8, 8)
    ax.grid(True)
    ax.set_aspect("equal")

    odometry_plot = ax.plot([], [], c="grey", alpha=0.7, label="Odometry")[0]
    expert_trajectory_plot = ax.plot([], [], c="orange", alpha=0.7, label="Expert Trajectory")[0]
    point_cloud_plot = ax.scatter([], [], c="red", s=1, alpha=0.5, label="Point Cloud")
    dynamic_obstacles_plot = ax.scatter([], [], c="blue", s=30, alpha=0.7, label="Dynamic Obstacles")
    dynamic_obstacles_heading_plots = [
        ax.arrow(0, 0, 0, 0, head_width=0.05, head_length=0.2, fc="g", ec="g", alpha=0.7) for _ in range(10)
    ]
    goal_plot = ax.scatter([], [], c="green", s=30, alpha=0.7, label="Goal Position")
    if trajectories.ndim == 4:
        trajectory_plot = [
            ax.plot([], [], alpha=0.8, linewidth=0.6, color="black" if scores is None else "grey")[0]
            for _ in range(trajectories.shape[1])
        ]
    elif trajectories.ndim == 3:
        trajectory_plot = [ax.plot([], [], alpha=0.8, linewidth=1.5, color="black")[0]]
    else:
        raise ValueError(
            f"Expected trajectories to have shape (N, 30, 2) or (N, E, 30, 2), but got {trajectories.shape}"
        )
    heading_plot = ax.arrow(0, 0, 0, 0, head_width=0.1, head_length=0.1, fc="k", ec="k", alpha=0.3, label="Heading")

    def init():
        return (
            odometry_plot,
            expert_trajectory_plot,
            point_cloud_plot,
            dynamic_obstacles_plot,
            *dynamic_obstacles_heading_plots,
            goal_plot,
            *trajectory_plot,
            heading_plot,
        )

    def update(timestep: int):
        data = dataset.get_plotting_data(bag_index, timestep)

        odometry = data["odometry"]

        translated_odometry = _translate_points(all_odometry[:, :2], -odometry[:2])
        odometry_plot.set_data(translated_odometry[:, 0], translated_odometry[:, 1])

        rotated_expert_trajectory_plot = _rotate_points(data["expert_trajectory"].T, odometry[2]).T
        expert_trajectory_plot.set_data(rotated_expert_trajectory_plot)

        point_cloud = data["point_cloud"]
        rotated_point_cloud = _rotate_points(point_cloud.T, odometry[2])
        point_cloud_plot.set_offsets(rotated_point_cloud)

        dynamic_obstacles_positions = data["dynamic_obstacle_positions"][:, :10]
        rotated_dynamic_obstacles_positions = _rotate_points(dynamic_obstacles_positions.T, odometry[2])
        dynamic_obstacles_plot.set_offsets(rotated_dynamic_obstacles_positions)

        dynamic_obstacles_velocities = data["dynamic_obstacle_velocities"][:, :10]
        dynamic_obstacles_headings = np.arctan2(dynamic_obstacles_velocities[1, :], dynamic_obstacles_velocities[0, :])
        for i, heading_plot in enumerate(dynamic_obstacles_heading_plots):
            heading_plot.set_data(
                x=rotated_dynamic_obstacles_positions[i, 0],
                y=rotated_dynamic_obstacles_positions[i, 1],
                dx=0.5 * np.cos(dynamic_obstacles_headings[i]),
                dy=0.5 * np.sin(dynamic_obstacles_headings[i]),
            )

        goal_position = data["projection_goal_position"]
        rotated_goal_position = _rotate_points(goal_position.T, odometry[2])
        goal_plot.set_offsets(rotated_goal_position)

        if trajectories.ndim == 4:
            if scores is not None:
                best_trajectory_index = np.argmax(scores[timestep])
            for i in range(trajectories.shape[1] - 1 if scores is not None else trajectories.shape[1]):
                if scores is not None and i == best_trajectory_index:
                    continue
                rotated_trajectory = _rotate_points(trajectories[timestep, i], odometry[2]).T
                trajectory_plot[i].set_data(rotated_trajectory)
                trajectory_plot[i].set_color("black" if scores is None else "grey")
            if scores is not None:
                rotated_trajectory = _rotate_points(trajectories[timestep, best_trajectory_index], odometry[2]).T
                trajectory_plot[-1].set_data(rotated_trajectory)
                trajectory_plot[-1].set_color("black")
                trajectory_plot[-1].set_linewidth(1.5)

        elif trajectories.ndim == 3:
            rotated_trajectory = _rotate_points(trajectories[timestep], odometry[2]).T
            trajectory_plot[0].set_data(rotated_trajectory)

        heading_plot.set_data(
            x=0,
            y=0,
            dx=np.cos(odometry[2]),
            dy=np.sin(odometry[2]),
        )

        return (
            odometry_plot,
            expert_trajectory_plot,
            point_cloud_plot,
            dynamic_obstacles_plot,
            *dynamic_obstacles_heading_plots,
            goal_plot,
            *trajectory_plot,
            heading_plot,
        )

    anim = animation.FuncAnimation(
        fig, update, frames=dataset.get_bag_length(bag_index), init_func=init, blit=True, interval=100
    )
    anim.save(os.path.join(save_directory, f"visualization_{bag_index:08}.mp4"), writer="ffmpeg", fps=10)
    plt.close(fig)
