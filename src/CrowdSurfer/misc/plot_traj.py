import os

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from navigation.projection_guidance import ProjectionGuidance


def initialize_projection_guidance():
    return ProjectionGuidance(
        num_obstacles=0,
        num_timesteps=100,
        total_time=10,
        obstacle_ellipse_semi_major_axis=0,
        obstacle_ellipse_semi_minor_axis=0,
        max_projection_iterations=100,
        device=torch.device("cpu"),
    )


def generate_trajectory(pg, c_x, c_y):
    return pg.coefficients_to_trajectory(c_x, c_y, position_only=True)


def read_npz_files(directory):
    all_data = []
    npz_files = [f for f in os.listdir(directory) if f.endswith(".npz")]

    for npz_file in tqdm(npz_files, desc="Processing files"):
        file_path = os.path.join(directory, npz_file)

        with np.load(file_path) as data:
            best_priest_coefficients = data["best_priest_coefficients.npy"]
            dynamic_obstacle_metadata = data["dynamic_obstacle_metadata.npy"]
            pointcloud = data["point_cloud.npy"]

            file_data = {
                "filename": npz_file,
                "best_priest_coefficients": best_priest_coefficients,
                "dynamic_obstacle_metadata": dynamic_obstacle_metadata,
                "pointcloud": pointcloud,
            }

            all_data.append(file_data)

    return all_data


def create_animation(x_best, y_best, pointcloud, dynamic_obstacles, filename, output_dir):
    print(f"Shape of x_best: {x_best.shape}")
    print(f"Shape of y_best: {y_best.shape}")
    print(f"Shape of pointcloud: {pointcloud.shape}")
    print(f"Shape of dynamic_obstacles: {dynamic_obstacles.shape}")

    # Check for NaN values in pointcloud
    nan_count = np.isnan(pointcloud).sum()
    total_elements = pointcloud.size
    nan_percentage = (nan_count / total_elements) * 100

    print(f"NaN values in pointcloud: {nan_count}")
    print(f"Total elements in pointcloud: {total_elements}")
    print(f"Percentage of NaN values: {nan_percentage:.2f}%")

    if nan_count > 0:
        print("Warning: Pointcloud contains NaN values!")

    if x_best.ndim == 1:
        x_best = x_best.reshape(1, -1)
        y_best = y_best.reshape(1, -1)

    num_timesteps = x_best.shape[1]

    fig, ax = plt.subplots(figsize=(12, 8))

    # Update limits to include dynamic obstacles
    x_min = min(np.min(x_best), np.min(pointcloud[:, :, 0]), np.min(dynamic_obstacles[:, :, 1]))
    x_max = max(np.max(x_best), np.max(pointcloud[:, :, 0]), np.max(dynamic_obstacles[:, :, 1]))
    y_min = min(np.min(y_best), np.min(pointcloud[:, :, 1]), np.min(dynamic_obstacles[:, :, 2]))
    y_max = max(np.max(y_best), np.max(pointcloud[:, :, 1]), np.max(dynamic_obstacles[:, :, 2]))

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.grid(True)
    ax.set_aspect("equal")
    # Create animation for this file
    trajectory_lines = ax.plot([], [], alpha=0.5)[0]
    pointcloud_scatter = ax.scatter([], [], c="red", s=1, alpha=0.5, label="Point Cloud")
    obstacle_scatter = ax.scatter([], [], c="green", s=30, alpha=0.7, label="Dynamic Obstacles")

    title = ax.set_title("")

    def init():
        # for line in trajectory_lines:
        trajectory_lines.set_data([], [])
        pointcloud_scatter.set_offsets(np.empty((0, 2)))
        obstacle_scatter.set_offsets(np.empty((0, 2)))
        return trajectory_lines, pointcloud_scatter, obstacle_scatter, title

    def update(frame):
        title.set_text(f"PRIEST Plot for {filename} (Timestep {frame+1}/{num_timesteps})")

        # for i, line in enumerate(trajectory_lines):
        trajectory_lines.set_data(x_best[frame], y_best[frame])

        if frame < pointcloud.shape[0]:
            pointcloud_scatter.set_offsets(pointcloud[frame, :, :2])
        else:
            pointcloud_scatter.set_offsets(np.empty((0, 2)))

        # Update dynamic obstacles
        valid_obstacles = dynamic_obstacles[frame, dynamic_obstacles[frame, :, 0] != 0]
        obstacle_scatter.set_offsets(valid_obstacles[:, 1:3])  # Use columns 1 and 2 for x and y positions

        return trajectory_lines, pointcloud_scatter, obstacle_scatter, title

    anim = animation.FuncAnimation(fig, update, frames=num_timesteps, init_func=init, blit=True, interval=100)

    output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_animation.mp4")
    anim.save(output_path, writer="ffmpeg", fps=10)
    plt.close(fig)


npz_directory = "/home/themys/visualize_dataset/test_1"

output_directory = "/home/themys/visualize_dataset/trajectory_animations"
os.makedirs(output_directory, exist_ok=True)

trajectory_data = read_npz_files(npz_directory)

pg = initialize_projection_guidance()

for data in tqdm(trajectory_data, desc="Generating animations"):
    coefficients = data["best_priest_coefficients"]
    filename = data["filename"]
    pointcloud = data["pointcloud"]
    dynamic_obstacles = data["dynamic_obstacle_metadata"]

    print(f"Processing file: {filename}")
    print(f"Shape of coefficients: {coefficients.shape}")
    print(f"Shape of dynamic_obstacles: {dynamic_obstacles.shape}")

    x_best = []
    y_best = []

    for coeff_set in coefficients:
        c_x = coeff_set[:, 0]
        c_y = coeff_set[:, 1]
        x, y = generate_trajectory(pg, c_x, c_y)
        x_best.append(x.cpu().numpy())
        y_best.append(y.cpu().numpy())

    x_best = np.array(x_best)
    y_best = np.array(y_best)

    print(f"Shape of x_best after conversion: {x_best.shape}")
    print(f"Shape of y_best after conversion: {y_best.shape}")

    create_animation(x_best, y_best, pointcloud, dynamic_obstacles, filename, output_directory)

print(f"Processed {len(trajectory_data)} files.")
print(f"Trajectory, pointcloud, and dynamic obstacle animations saved in {output_directory}")
