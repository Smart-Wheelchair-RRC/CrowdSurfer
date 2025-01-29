from typing import Optional

import torch
from torch import Tensor

from .projection_guidance import ProjectionGuidance


def project_coefficients(
    projection_guidance: ProjectionGuidance,
    coefficients: Tensor,
    initial_ego_position_x: Tensor,
    initial_ego_position_y: Tensor,
    initial_ego_velocity_x: Tensor,
    initial_ego_velocity_y: Tensor,
    initial_ego_acceleration_x: Tensor,
    initial_ego_acceleration_y: Tensor,
    final_ego_position_x: Tensor,
    final_ego_position_y: Tensor,
    obstacle_positions_x: Optional[Tensor],
    obstacle_positions_y: Optional[Tensor],
    obstacle_velocities_x: Optional[Tensor],
    obstacle_velocities_y: Optional[Tensor],
) -> Tensor:
    outputs = projection_guidance.project_trajectory(
        initial_ego_position_x=initial_ego_position_x,
        initial_ego_position_y=initial_ego_position_y,
        initial_ego_velocity_x=initial_ego_velocity_x,
        initial_ego_velocity_y=initial_ego_velocity_y,
        initial_ego_acceleration_x=initial_ego_acceleration_x,
        initial_ego_acceleration_y=initial_ego_acceleration_y,
        final_ego_position_x=final_ego_position_x,
        final_ego_position_y=final_ego_position_y,
        obstacle_positions_x=(
            obstacle_positions_x
            if obstacle_positions_x is not None
            else torch.tensor([]).tile(coefficients.shape[0], 1).to(coefficients.device)
        ),
        obstacle_positions_y=(
            obstacle_positions_y
            if obstacle_positions_y is not None
            else torch.tensor([]).tile(coefficients.shape[0], 1).to(coefficients.device)
        ),
        obstacle_velocities_x=(
            obstacle_velocities_x
            if obstacle_velocities_x is not None
            else torch.tensor([]).tile(coefficients.shape[0], 1).to(coefficients.device)
        ),
        obstacle_velocities_y=(
            obstacle_velocities_y
            if obstacle_velocities_y is not None
            else torch.tensor([]).tile(coefficients.shape[0], 1).to(coefficients.device)
        ),
        maximum_velocity=6.0,
        maximum_acceleration=2.0,
        initial_solution_x=coefficients[:, 0, :],
        initial_solution_y=coefficients[:, 1, :],
    )
    return torch.stack(outputs[:2], dim=1)


def project_trajectory(
    projection_guidance: ProjectionGuidance,
    trajectory: Tensor,
    coefficients: Optional[Tensor] = None,
) -> Tensor:
    outputs = projection_guidance.project_trajectory(
        initial_ego_position_x=trajectory[:, 0, 0],
        initial_ego_position_y=trajectory[:, 1, 0],
        initial_ego_velocity_x=torch.zeros(trajectory.shape[0]).to(trajectory.device),
        initial_ego_velocity_y=torch.zeros(trajectory.shape[0]).to(trajectory.device),
        initial_ego_acceleration_x=torch.zeros(trajectory.shape[0]).to(trajectory.device),
        initial_ego_acceleration_y=torch.zeros(trajectory.shape[0]).to(trajectory.device),
        final_ego_position_x=trajectory[:, 0, -1],
        final_ego_position_y=trajectory[:, 1, -1],
        obstacle_positions_x=torch.tensor([]).tile(trajectory.shape[0], -1).to(trajectory.device),
        obstacle_positions_y=torch.tensor([]).tile(trajectory.shape[0], -1).to(trajectory.device),
        obstacle_velocities_x=torch.tensor([]).tile(trajectory.shape[0], -1).to(trajectory.device),
        obstacle_velocities_y=torch.tensor([]).tile(trajectory.shape[0], -1).to(trajectory.device),
        maximum_velocity=1.0,
        maximum_acceleration=1.0,
        warm_ego_position_trajectory_x=trajectory[:, 0, :] if coefficients is None else None,
        warm_ego_position_trajectory_y=trajectory[:, 1, :] if coefficients is None else None,
        initial_solution_x=coefficients[:, 0, :] if coefficients is not None else None,
        initial_solution_y=coefficients[:, 1, :] if coefficients is not None else None,
    )
    return torch.stack(outputs[:2], dim=1)
