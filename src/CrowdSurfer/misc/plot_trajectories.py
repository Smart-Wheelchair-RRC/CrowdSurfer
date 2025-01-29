import os
from functools import partial

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from navigation.priest_guidance import PriestPlanner
from navigation.projection_guidance import ProjectionGuidance


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


def initialize_projection_guidance():
    return ProjectionGuidance(
        num_obstacles=0,
        num_timesteps=30,
        total_time=3,
        obstacle_ellipse_semi_major_axis=0,
        obstacle_ellipse_semi_minor_axis=0,
        max_projection_iterations=100,
        device=torch.device("cpu"),
    )


def load_key(directory, key):
    return np.load(os.path.join(directory, f"{key}.npy"))


def generate_trajectory(pg, c_x, c_y):
    return pg.coefficients_to_trajectory(c_x, c_y, position_only=True)


def read_npz_files(directory):
    directory = os.path.join(directory, "processed")
    all_data = []
    npz_files = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]

    for npz_file in tqdm(npz_files, desc="Processing files"):
        file_path = os.path.join(directory, npz_file)
        print(file_path)
        get_data = partial(load_key, file_path)

        dynamic_obstacle_metadata = get_data("dynamic_obstacle_metadata")
        pointcloud = get_data("point_cloud")
        goal = get_data("goal_position")
        odometry = get_data("odometry")
        expert_trajectory = get_data("expert_trajectory")
        ego_velocity = get_data("ego_velocity")
        pixcnn_traj = get_data("PixCNN_trajectories")
        scores = get_data("scores")

        file_data = {
            "filename": npz_file,
            "dynamic_obstacle_metadata": dynamic_obstacle_metadata,
            "pointcloud": pointcloud,
            "goal": goal,
            "odometry": odometry,
            "expert_trajectory": expert_trajectory,
            "ego_velocity": ego_velocity,
            "Pixel_Priest": pixcnn_traj,
            "Scores": scores,
        }

        all_data.append(file_data)

    return all_data


def create_animation(
    x_best,
    y_best,
    x_elite,
    y_elite,
    pointcloud,
    dynamic_obstacles,
    goals,
    pointcloud_center,
    odometry,
    expert_trajectory,
    filename,
    output_dir,
):
    print(f"Shape of scored_x_best: {x_best.shape}")
    print(f"Shape of scored_y_best: {y_best.shape}")
    print(f"Shape of x_elite: {x_elite.shape}")
    print(f"Shape of y_elite: {y_elite.shape}")
    print(f"Shape of pointcloud: {pointcloud.shape}")
    print(f"Shape of dynamic_obstacles: {dynamic_obstacles.shape}")
    print(f"Shape of goals: {goals.shape}")

    # Check for NaN values in pointcloud
    nan_count = np.isnan(pointcloud).sum()
    total_elements = pointcloud.size
    nan_percentage = (nan_count / total_elements) * 100

    print(f"NaN values in pointcloud: {nan_count}")
    print(f"Total elements in pointcloud: {total_elements}")
    print(f"Percentage of NaN values: {nan_percentage:.2f}%")

    if nan_count > 0:
        print("Warning: Pointcloud contains NaN values!")

    num_timesteps, num_elites = x_elite.shape[:2]

    fig, ax = plt.subplots(figsize=(12, 8))

    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)

    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.grid(True)
    ax.set_aspect("equal")

    # Create animation for this file
    trajectory_lines = [ax.plot([], [], alpha=0.5, linewidth=0.5, color="grey")[0] for _ in range(num_elites)]
    best_traj_line = ax.plot([], [], alpha=1.0, linewidth=1.0, color="black", label="Best Trajectory")[0]

    pointcloud_scatter = ax.scatter([], [], c="red", s=1, alpha=0.5, label="Point Cloud")
    obstacle_scatter = ax.scatter([], [], c="green", s=30, alpha=0.7, label="Dynamic Obstacles")
    goal_scatter = ax.scatter([], [], c="blue", s=30, alpha=0.7, label="Goal Position")
    heading_arrow = ax.arrow(
        0,
        0,
        0,
        0,
        head_width=0.1,
        head_length=0.1,
        fc="k",
        ec="k",
        alpha=0.3,
        label="Heading",
    )
    odometry_plot = ax.plot(odometry[:, 0], odometry[:, 1], c="grey", alpha=0.7, label="Odometry")[0]
    expert_trajectory_plot = ax.plot(
        expert_trajectory[:, 0],
        expert_trajectory[:, 1],
        c="orange",
        alpha=0.7,
        label="Expert Trajectory",
    )[0]

    # Heading Arrows
    obstacle_arrows = [
        ax.arrow(0, 0, 0, 0, head_width=0.05, head_length=0.2, fc="g", ec="g", alpha=0.7) for _ in range(10)
    ]

    title = ax.set_title("")

    def init():
        for line in trajectory_lines:
            line.set_data([], [])
        best_traj_line.set_data([], [])
        pointcloud_scatter.set_offsets(np.empty((0, 2)))
        obstacle_scatter.set_offsets(np.empty((0, 2)))
        goal_scatter.set_offsets(np.empty((0, 2)))
        odometry_plot.set_data([], [])
        expert_trajectory_plot.set_data([], [])
        heading_arrow.set_data(x=0, y=0, dx=0, dy=0)
        for arrow in obstacle_arrows:
            arrow.set_data(x=0, y=0, dx=0, dy=0)
        return (
            heading_arrow,
            *trajectory_lines,
            best_traj_line,
            pointcloud_scatter,
            obstacle_scatter,
            goal_scatter,
            odometry_plot,
            expert_trajectory_plot,
            title,
            *obstacle_arrows,
        )

    def update(frame):
        title.set_text(f"PRIEST Plot for {filename} (Timestep {frame+1}/{num_timesteps})")

        for i, line in enumerate(trajectory_lines):
            traj = np.stack((x_elite[frame, i], y_elite[frame, i]), axis=-1)
            traj = _rotate_points(traj, odometry[frame, 2])
            line.set_data(traj[:, 0], traj[:, 1])

        # Update best trajectory
        best_traj = np.stack((x_best[frame], y_best[frame]), axis=-1)
        best_traj = _rotate_points(best_traj, odometry[frame, 2])
        best_traj_line.set_data(best_traj[:, 0], best_traj[:, 1])

        # unrotate pointcloud
        pcd_data = _rotate_points(pointcloud[frame, :, :2], odometry[frame, 2])

        if frame < pointcloud.shape[0]:
            pointcloud_scatter.set_offsets(pcd_data)
        else:
            pointcloud_scatter.set_offsets(np.empty((0, 2)))

        # Update dynamic obstacles and their headings
        valid_obstacles = dynamic_obstacles[frame, dynamic_obstacles[frame, :, 0] != np.nan]
        rotated_obstacles = _rotate_points(valid_obstacles[:, 1:3], odometry[frame, 2])
        obstacle_scatter.set_offsets(rotated_obstacles)

        for i, arrow in enumerate(obstacle_arrows):
            if i < len(valid_obstacles):
                obstacle = valid_obstacles[i]
                rotated_pos = rotated_obstacles[i]
                vx, vy = obstacle[3:5]
                heading = np.arctan2(vy, vx)
                arrow.set_data(
                    x=rotated_pos[0],
                    y=rotated_pos[1],
                    dx=0.2 * np.cos(heading),
                    dy=0.2 * np.sin(heading),
                )
            else:
                arrow.set_data(x=0, y=0, dx=0, dy=0)

        goal_scatter.set_offsets(_rotate_points(goals[frame, :2], odometry[frame, 2]))
        curr_odom = odometry[:, :2] - odometry[frame, :2]
        odometry_plot.set_data(curr_odom[:, 0], curr_odom[:, 1])
        curr_expert = _rotate_points(expert_trajectory[frame, :, :2], odometry[frame, 2])
        expert_trajectory_plot.set_data(curr_expert[:, 0], curr_expert[:, 1])
        heading = odometry[frame, 2]
        heading_arrow.set_data(
            x=0,
            y=0,
            dx=np.cos(heading),
            dy=np.sin(heading),
        )
        return (
            heading_arrow,
            *trajectory_lines,
            best_traj_line,
            pointcloud_scatter,
            obstacle_scatter,
            goal_scatter,
            odometry_plot,
            expert_trajectory_plot,
            title,
            *obstacle_arrows,
        )

    anim = animation.FuncAnimation(fig, update, frames=num_timesteps, init_func=init, blit=True, interval=100)

    output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_animation.mp4")
    anim.save(output_path, writer="ffmpeg", fps=10)
    plt.close(fig)


# Change to your directory where npz files are stored
npz_directory = "data/custom"

# Change to your desired output directory
output_directory = "data/custom/animations"
os.makedirs(output_directory, exist_ok=True)

trajectory_data = read_npz_files(npz_directory)

pg = initialize_projection_guidance()
planner = PriestPlanner(
    t_fin=3,
    num=30,
    num_waypoints=31,
    weight_track=1.2,
)

for data in tqdm(trajectory_data, desc="Generating PRIEST animations"):
    # elite_coefficients = data["elite_priest_coefficients"]
    filename = data["filename"]
    pointcloud = data["pointcloud"]
    dynamic_obstacles = data["dynamic_obstacle_metadata"]
    goal = data["goal"]
    odom = data["odometry"]
    expert_trajectory = data["expert_trajectory"]
    # expert_coeff = data["expert_coeff"]

    # Select non nan values from pointcloud
    nan_index = ~np.isnan(pointcloud[:, :, :2]).any(axis=2)
    means = []
    for i in range(pointcloud.shape[0]):
        current_mean = np.mean(pointcloud[i, nan_index[i], :2], axis=0)
        means.append(current_mean)

    pointcloud_center = np.array(means)
    print(f"Shape of pointcloud_center: {pointcloud_center.shape}")

    print(f"Processing file: {filename}")
    # print(f"Shape of elite_coefficients: {elite_coefficients.shape}")
    print(f"Shape of dynamic_obstacles: {dynamic_obstacles.shape}")
    print(f"Shape of pointcloud: {pointcloud.shape}")

    pixelcnn_trajectories = data["Pixel_Priest"]
    traj_score = data["Scores"]

    x_elite = []
    y_elite = []
    x_best = []
    y_best = []

    # FOR PRIEST TRAJECTORIES ON PIXELCNN

    # pixelcnn shape (num timesteps, num samples, 30, 2)
    # trajectories = []
    # for sample in tqdm(pixelcnn_trajectories.transpose((1, 0, 2, 3))):
    #     trajectory = create_priest_trajectories_from_observations(
    #         data=data, planner=planner, pixelcnn_trajectories=sample
    #     )
    #     trajectories.append(trajectory)

    # trajectories = np.array(trajectories).transpose((1, 0, 2, 3))
    # goal_to_add = np.tile(
    #     goal[:, np.newaxis, np.newaxis, :], (1, trajectories.shape[1], 1, 1)
    # )
    # trajectories = np.concatenate((trajectories, goal_to_add), axis=2)

    # FOR PRIEST COEFFICIENTS ON PIXELCNN

    # num_timesteps, num_trajectories, traj_length, coords = pixelcnn_trajectories.shape
    # optimized_trajectories = np.zeros_like(pixelcnn_trajectories)

    # for i in tqdm(range(num_trajectories), desc="Processing trajectories"):
    #     trajectory = pixelcnn_trajectories[:, i, :, :]  # Shape: (919, 30, 2)
    #     bernstein_coeffs = create_priest_trajectories_from_observations(
    #         data=data, planner=planner, pixelcnn_trajectories=trajectory) # Shape: (919,11,2)

    #     for t in range(num_timesteps):
    #         c_x = bernstein_coeffs[t, :, 0]
    #         c_y = bernstein_coeffs[t, :, 1]

    #         x, y = generate_trajectory(pg, c_x, c_y) # Shape: (30,)

    #         x_np = x.cpu().numpy() if isinstance(x, torch.Tensor) else x
    #         y_np = y.cpu().numpy() if isinstance(y, torch.Tensor) else y

    #         generated_traj = np.column_stack((x_np, y_np)) # Shape: (30,2)
    #         if len(generated_traj) < traj_length:
    #             pad_length = traj_length - len(generated_traj)
    #             generated_traj = np.pad(generated_traj, ((0, pad_length), (0, 0)), mode='constant', constant_values=0)
    #         elif len(generated_traj) > traj_length:
    #             generated_traj = generated_traj[:traj_length]

    #         optimized_trajectories[t, i, :, :] = generated_traj

    #     trajectories = optimized_trajectories # Shape: (919,10,30,2)

    # FOR DIRECT PLOTTING OF PROJECTED PIXELCNN TRAJECTORIES

    for timestep, timestep_coeffs in enumerate(pixelcnn_trajectories):
        x_timestep = []
        y_timestep = []
        # To select the best trajectory using scoring network
        timestep_score = np.argmax(traj_score[timestep])
        scored_x = timestep_coeffs[timestep_score, :, 0]
        scored_y = timestep_coeffs[timestep_score, :, 1]
        x_best.append(scored_x)
        y_best.append(scored_y)
        for elite_coeff in timestep_coeffs:
            c_x = elite_coeff[:, 0]
            c_y = elite_coeff[:, 1]
            # x, y = generate_trajectory(pg, c_x, c_y)
            x_timestep.append(c_x)
            y_timestep.append(c_y)
        x_elite.append(x_timestep)
        y_elite.append(y_timestep)

    x_elite = np.array(x_elite)
    y_elite = np.array(y_elite)
    x_best = np.array(x_best)
    y_best = np.array(y_best)

    # FOR VISUALIZING EXPERT TRAJECTORY USING BERNSTEIN COEFFICIENTS
    # expert_x = []
    # expert_y = []
    # # print("Expert_coeff Shape: ", expert_coeff.shape)
    # for timestep_coeff in expert_coeff:
    #     c_x = timestep_coeff[:, 0]
    #     c_y = timestep_coeff[:, 1]
    #     x, y = generate_trajectory(pg, c_x, c_y)
    #     expert_x.append(x)
    #     expert_y.append(y)

    # x_best = np.array(expert_x)
    # y_best = np.array(expert_y)
    # print(x_best.shape)
    # print(y_best.shape)

    # print(f"Shape of x_elite after conversion: {x_elite.shape}")
    # print(f"Shape of y_elite after conversion: {y_elite.shape}")

    # ANIMATION

    create_animation(
        # trajectories[:, :, :, 0],
        # trajectories[:, :, :, 1],
        x_best,
        y_best,
        x_elite,
        y_elite,
        pointcloud,
        dynamic_obstacles,
        goal,
        pointcloud_center,
        odom,
        expert_trajectory,
        filename,
        output_directory,
    )

print(f"Processed {len(trajectory_data)} files.")
print(f"npz file animations saved in {output_directory}")
