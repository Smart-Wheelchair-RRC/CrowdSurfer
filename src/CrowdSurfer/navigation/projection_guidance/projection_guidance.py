from dataclasses import dataclass
from typing import List, Optional, Tuple, Union, cast, overload

import torch
from torch import Tensor
from typing_extensions import Literal

from .bernstein_polynomials import bernstein_polynomials_generator


def conditional_compilation(condition):
    def decorator(func):
        if not condition:
            return func
        return torch.compile(func, options={"triton.cudagraphs": True}, fullgraph=True)

    return decorator


COMPILE = False


@dataclass
class Carry:
    primal_solution_x: Tensor  # shape (BATCH_SIZE, 11)
    primal_solution_y: Tensor  # shape (BATCH_SIZE, 11)
    collision_alpha_vector: Tensor  # shape (BATCH_SIZE, NUM_OBSTACLES, num_timesteps)
    collision_d_vector: Tensor  # shape (BATCH_SIZE, NUM_OBSTACLES, num_timesteps)
    velocity_alpha_vector: Tensor  # shape (BATCH_SIZE, num_timesteps)
    velocity_d_vector: Tensor  # shape (BATCH_SIZE, num_timesteps)
    acceleration_alpha_vector: Tensor  # shape (BATCH_SIZE, num_timesteps)
    acceleration_d_vector: Tensor  # shape (BATCH_SIZE, num_timesteps)
    lambda_x: Tensor  # shape (BATCH_SIZE, 11)
    lambda_y: Tensor  # shape (BATCH_SIZE, 11)


@dataclass
class Output:
    ego_position_trajectory_x: Tensor  # shape (BATCH_SIZE, num_timesteps)
    ego_position_trajectory_y: Tensor  # shape (BATCH_SIZE, num_timesteps)
    ego_velocity_trajectory_x: Tensor  # shape (BATCH_SIZE, num_timesteps)
    ego_velocity_trajectory_y: Tensor  # shape (BATCH_SIZE, num_timesteps)
    ego_acceleration_trajectory_x: Tensor  # shape (BATCH_SIZE, num_timesteps)
    ego_acceleration_trajectory_y: Tensor  # shape (BATCH_SIZE, num_timesteps)
    error_residuals: Tensor  # shape (BATCH_SIZE,)


class ProjectionGuidance:
    def __init__(
        self,
        num_obstacles: int,
        num_timesteps: int,
        total_time: float,
        obstacle_ellipse_semi_major_axis: float,
        obstacle_ellipse_semi_minor_axis: float,
        max_projection_iterations: int,
        device: torch.device,
    ):
        self.NUM_OBSTACLES = num_obstacles
        self.OBSTACLE_ELLIPSE_SEMI_MAJOR_AXIS = obstacle_ellipse_semi_major_axis
        self.OBSTACLE_ELLIPSE_SEMI_MINOR_AXIS = obstacle_ellipse_semi_minor_axis
        self.DEVICE = device
        self.DT = torch.linspace(0, total_time, num_timesteps).to(self.DEVICE)  # shape (num_timesteps,)

        self.MAX_PROJECTION_ITERATIONS = max_projection_iterations

        (
            self.BERNSTEIN_POLYNOMIALS,  # shape (num_timesteps, 11)
            self.BERNSTEIN_POLYNOMIALS_FIRST_DIFFERENTIAL,  # shape (num_timesteps, 11)
            self.BERNSTEIN_POLYNOMIALS_SECOND_DIFFERENTIAL,  # shape (num_timesteps, 11)
        ) = bernstein_polynomials_generator(tmin=0, tmax=total_time, t_actual=self.DT)

        INITIAL_COST_MATRIX = (
            self.BERNSTEIN_POLYNOMIALS.T @ self.BERNSTEIN_POLYNOMIALS
            + 1e-4 * torch.eye(11).to(device)
            + (1 * self.BERNSTEIN_POLYNOMIALS_SECOND_DIFFERENTIAL.T @ self.BERNSTEIN_POLYNOMIALS_SECOND_DIFFERENTIAL)
        )  # shape (11, 11)

        self.INVERSE_INITIAL_COST_MATRIX_X = cast(Tensor, torch.linalg.inv(INITIAL_COST_MATRIX))  # shape (11, 11)

        self.INVERSE_INITIAL_COST_MATRIX_Y = (
            self.INVERSE_INITIAL_COST_MATRIX_X  # shape (11, 11)
        )

        self.RHO_PROJECTION = 1.0
        self.RHO_COLLISION = 1.0
        self.RHO_INEQUALITIES = 1.0

        self.A_EQUALITY = torch.stack(
            [
                self.BERNSTEIN_POLYNOMIALS[0],  # shape (11,)
                self.BERNSTEIN_POLYNOMIALS_FIRST_DIFFERENTIAL[0],  # shape (11,)
                self.BERNSTEIN_POLYNOMIALS_SECOND_DIFFERENTIAL[0],  # shape (11,)
                self.BERNSTEIN_POLYNOMIALS[-1],  # shape (11,)
            ],
            dim=0,
        )  # shape (4, 11)
        self.A_PROJECTION = torch.eye(11).to(device)  # shape (11, 11)
        self.A_COLLISION = torch.tile(
            self.BERNSTEIN_POLYNOMIALS[None, :, :], (self.NUM_OBSTACLES, 1, 1)
        )  # shape (NUM_OBSTACLES, num_timesteps, 11)
        self.A_VELOCITY = (
            self.BERNSTEIN_POLYNOMIALS_FIRST_DIFFERENTIAL  # shape (num_timesteps, 11)
        )
        self.A_ACCELERATION = (
            self.BERNSTEIN_POLYNOMIALS_SECOND_DIFFERENTIAL  # shape (num_timesteps, 11)
        )

        PROJECTION_COST_MATRIX_X = (
            self.RHO_PROJECTION * (self.A_PROJECTION.T @ self.A_PROJECTION)
            + self.RHO_COLLISION * (self.A_COLLISION.reshape(-1, 11).T @ self.A_COLLISION.reshape(-1, 11))
            + self.RHO_INEQUALITIES * (self.A_VELOCITY.T @ self.A_VELOCITY)
            + self.RHO_INEQUALITIES * (self.A_ACCELERATION.T @ self.A_ACCELERATION)
        )  # shape (11, 11)

        PROJECTION_COST_MATRIX_X = torch.concatenate(
            [
                torch.concatenate(
                    [
                        PROJECTION_COST_MATRIX_X,
                        self.A_EQUALITY.T,
                    ],
                    dim=1,
                ),  # shape (11, 15)
                torch.concatenate(
                    [
                        self.A_EQUALITY,
                        torch.zeros((self.A_EQUALITY.shape[0], self.A_EQUALITY.shape[0])).to(device),
                    ],
                    dim=1,
                ),  # shape (4, 15)
            ],
            dim=0,
        )  # shape (15, 15)

        self.INVERSE_PROJECTION_COST_MATRIX_X = cast(Tensor, torch.inverse(PROJECTION_COST_MATRIX_X))  # shape (15, 15)
        self.INVERSE_PROJECTION_COST_MATRIX_Y = (
            self.INVERSE_PROJECTION_COST_MATRIX_X  # shape (15, 15)
        )

    @conditional_compilation(COMPILE)
    def predict_obstacle_trajectories(
        self,
        obstacle_positions_x: Tensor,  # shape (BATCH_SIZE, NUM_OBSTACLES)
        obstacle_positions_y: Tensor,  # shape (BATCH_SIZE, NUM_OBSTACLES,)
        obstacle_velocities_x: Tensor,  # shape (BATCH_SIZE, NUM_OBSTACLES)
        obstacle_velocities_y: Tensor,  # shape (BATCH_SIZE, NUM_OBSTACLES)
        initial_ego_position_x: Tensor,  # shape (BATCH_SIZE,)
        initial_ego_position_y: Tensor,  # shape (BATCH_SIZE,)
    ) -> Tuple[Tensor, Tensor]:
        obstacle_positions_trajectory_x = (
            obstacle_positions_x[:, :, None] + obstacle_velocities_x[:, :, None] * self.DT[None, None, :]
        )  # shape (BATCH_SIZE, NUM_OBSTACLES, num_timesteps)

        obstacle_positions_trajectory_y = (
            obstacle_positions_y[:, :, None] + obstacle_velocities_y[:, :, None] * self.DT[None, None, :]
        )  # shape (BATCH_SIZE, NUM_OBSTACLES, num_timesteps)

        distances = torch.sqrt(
            (initial_ego_position_x[:, None, None] - obstacle_positions_trajectory_x) ** 2
            + (initial_ego_position_y[:, None, None] - obstacle_positions_trajectory_y) ** 2
        )  # shape (BATCH_SIZE, NUM_OBSTACLES, num_timesteps)

        distances = torch.argsort(
            distances
        )  # sort, closest to farthest # shape (BATCH_SIZE, NUM_OBSTACLES, num_timesteps)

        return (
            obstacle_positions_trajectory_x.take(distances),  # shape (BATCH_SIZE, NUM_OBSTACLES, num_timesteps)
            obstacle_positions_trajectory_y.take(distances),  # shape (BATCH_SIZE, NUM_OBSTACLES, num_timesteps)
        )

    @conditional_compilation(COMPILE)
    def generate_alpha_d_lambda(
        self,
        ego_position_trajectory_x: Tensor,  # shape (BATCH_SIZE, num_timesteps)
        ego_position_trajectory_y: Tensor,  # shape (BATCH_SIZE, num_timesteps)
        ego_velocity_trajectory_x: Tensor,  # shape (BATCH_SIZE, num_timesteps)
        ego_velocity_trajectory_y: Tensor,  # shape (BATCH_SIZE, num_timesteps)
        ego_acceleration_trajectory_x: Tensor,  # shape (BATCH_SIZE, num_timesteps)
        ego_acceleration_trajectory_y: Tensor,  # shape (BATCH_SIZE, num_timesteps)
        obstacle_positions_trajectory_x: Tensor,  # shape (BATCH_SIZE, NUM_OBSTACLES, num_timesteps)
        obstacle_positions_trajectory_y: Tensor,  # shape (BATCH_SIZE, NUM_OBSTACLES, num_timesteps)
        maximum_velocity: float,
        maximum_acceleration: float,
        lambda_x: Tensor,  # shape (BATCH_SIZE, 11)
        lambda_y: Tensor,  # shape (BATCH_SIZE, 11)
    ) -> Tuple[
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
    ]:
        ## Construct collision d vector
        ego_to_obstacle_distance_x = (
            ego_position_trajectory_x[:, None, :] - obstacle_positions_trajectory_x
        )  # shape (BATCH_SIZE, NUM_OBSTACLES, num_timesteps)

        ego_to_obstacle_distance_y = (
            ego_position_trajectory_y[:, None, :] - obstacle_positions_trajectory_y
        )  # shape (BATCH_SIZE, NUM_OBSTACLES, num_timesteps)

        collision_alpha_vector = torch.arctan2(
            self.OBSTACLE_ELLIPSE_SEMI_MINOR_AXIS * ego_to_obstacle_distance_y,
            self.OBSTACLE_ELLIPSE_SEMI_MAJOR_AXIS * ego_to_obstacle_distance_x,
        )  # shape (BATCH_SIZE, NUM_OBSTACLES, num_timesteps)

        collision_d_vector = (
            1.0
            * self.RHO_COLLISION
            * (
                (self.OBSTACLE_ELLIPSE_SEMI_MINOR_AXIS * ego_to_obstacle_distance_x * torch.cos(collision_alpha_vector))
                + (
                    self.OBSTACLE_ELLIPSE_SEMI_MAJOR_AXIS
                    * ego_to_obstacle_distance_y
                    * torch.sin(collision_alpha_vector)
                )
            )
        ) / (
            1.0
            * self.RHO_COLLISION
            * (
                ((self.OBSTACLE_ELLIPSE_SEMI_MAJOR_AXIS**2) * (torch.sin(collision_alpha_vector) ** 2))
                + ((self.OBSTACLE_ELLIPSE_SEMI_MINOR_AXIS**2) * (torch.cos(collision_alpha_vector) ** 2))
            )
        )  # shape (BATCH_SIZE, NUM_OBSTACLES, num_timesteps)

        collision_d_vector = torch.clip(collision_d_vector, min=1)  # shape (BATCH_SIZE, NUM_OBSTACLES, num_timesteps)

        resulting_collision_e_vector_x = (
            ego_to_obstacle_distance_x
            - self.OBSTACLE_ELLIPSE_SEMI_MINOR_AXIS * collision_d_vector * torch.cos(collision_alpha_vector)
        )  # shape (BATCH_SIZE, NUM_OBSTACLES, num_timesteps)

        resulting_collision_e_vector_y = (
            ego_to_obstacle_distance_y
            - self.OBSTACLE_ELLIPSE_SEMI_MAJOR_AXIS * collision_d_vector * torch.sin(collision_alpha_vector)
        )  # shape (BATCH_SIZE, NUM_OBSTACLES, num_timesteps)

        ## Construct velocity d vector

        velocity_alpha_vector = torch.arctan2(
            ego_velocity_trajectory_y, ego_velocity_trajectory_x
        )  # shape (BATCH_SIZE, num_timesteps)

        velocity_d_vector = (
            1.0
            * self.RHO_INEQUALITIES
            * (
                ego_velocity_trajectory_x * torch.cos(velocity_alpha_vector)
                + ego_velocity_trajectory_y * torch.sin(velocity_alpha_vector)
            )
        ) / (
            1.0
            * self.RHO_INEQUALITIES
            * (torch.cos(velocity_alpha_vector) ** 2 + torch.sin(velocity_alpha_vector) ** 2)
        )  # shape (BATCH_SIZE, num_timesteps)

        velocity_d_vector = torch.clip(velocity_d_vector, max=maximum_velocity)  # shape (BATCH_SIZE, num_timesteps)

        resulting_velocity_e_vector_x = ego_velocity_trajectory_x - velocity_d_vector * torch.cos(
            velocity_alpha_vector
        )  # shape (BATCH_SIZE, num_timesteps)

        resulting_velocity_e_vector_y = ego_velocity_trajectory_y - velocity_d_vector * torch.sin(
            velocity_alpha_vector
        )  # shape (BATCH_SIZE, num_timesteps)

        ## Construct acceleration d vector
        acceleration_alpha_vector = torch.arctan2(
            ego_acceleration_trajectory_y, ego_acceleration_trajectory_x
        )  # shape (BATCH_SIZE, num_timesteps)

        acceleration_d_vector = (
            1.0
            * self.RHO_INEQUALITIES
            * (
                ego_acceleration_trajectory_x * torch.cos(acceleration_alpha_vector)
                + ego_acceleration_trajectory_y * torch.sin(acceleration_alpha_vector)
            )
        ) / (
            1.0
            * self.RHO_INEQUALITIES
            * (torch.cos(acceleration_alpha_vector) ** 2 + torch.sin(acceleration_alpha_vector) ** 2)
        )  # shape (BATCH_SIZE, num_timesteps)

        acceleration_d_vector = torch.clip(
            acceleration_d_vector, max=maximum_acceleration
        )  # shape (BATCH_SIZE, num_timesteps)

        resulting_acceleration_e_vector_x = ego_acceleration_trajectory_x - acceleration_d_vector * torch.cos(
            acceleration_alpha_vector
        )  # shape (BATCH_SIZE, num_timesteps)

        resulting_acceleration_e_vector_y = ego_acceleration_trajectory_y - acceleration_d_vector * torch.sin(
            acceleration_alpha_vector
        )  # shape (BATCH_SIZE, num_timesteps)

        updated_lambda_x = (
            lambda_x
            - self.RHO_COLLISION
            * (self.A_COLLISION.reshape(-1, 11).T @ resulting_collision_e_vector_x.flatten(start_dim=1).T).T
            - self.RHO_INEQUALITIES * (self.A_VELOCITY.T @ resulting_velocity_e_vector_x.T).T
            - self.RHO_INEQUALITIES * (self.A_ACCELERATION.T @ resulting_acceleration_e_vector_x.T).T
        )  # shape (BATCH_SIZE, 11)

        updated_lambda_y = (
            lambda_y
            - self.RHO_COLLISION
            * (self.A_COLLISION.reshape(-1, 11).T @ resulting_collision_e_vector_y.flatten(start_dim=1).T).T
            - self.RHO_INEQUALITIES * (self.A_VELOCITY.T @ resulting_velocity_e_vector_y.T).T
            - self.RHO_INEQUALITIES * (self.A_ACCELERATION.T @ resulting_acceleration_e_vector_y.T).T
        )  # shape (BATCH_SIZE, 11)

        return (
            collision_alpha_vector,  # shape (BATCH_SIZE, NUM_OBSTACLES, num_timesteps)
            collision_d_vector,  # shape (BATCH_SIZE, NUM_OBSTACLES, num_timesteps)
            velocity_alpha_vector,  # shape (BATCH_SIZE, num_timesteps)
            velocity_d_vector,  # shape (BATCH_SIZE, num_timesteps)
            acceleration_alpha_vector,  # shape (BATCH_SIZE, num_timesteps)
            acceleration_d_vector,  # shape (BATCH_SIZE, num_timesteps)
            updated_lambda_x,  # shape (BATCH_SIZE, 11)
            updated_lambda_y,  # shape (BATCH_SIZE, 11)
            resulting_collision_e_vector_x,  # shape (BATCH_SIZE, NUM_OBSTACLES, num_timesteps)
            resulting_collision_e_vector_y,  # shape (BATCH_SIZE, NUM_OBSTACLES, num_timesteps)
            resulting_velocity_e_vector_x,  # shape (BATCH_SIZE, num_timesteps)
            resulting_velocity_e_vector_y,  # shape (BATCH_SIZE, num_timesteps)
            resulting_acceleration_e_vector_x,  # shape (BATCH_SIZE, num_timesteps)
            resulting_acceleration_e_vector_y,  # shape (BATCH_SIZE, num_timesteps)
        )

    @conditional_compilation(COMPILE)
    def compute_error_residuals(
        self,
        resulting_collision_e_vector_x: Tensor,  # shape (BATCH_SIZE, NUM_OBSTACLES, num_timesteps)
        resulting_collision_e_vector_y: Tensor,  # shape (BATCH_SIZE, NUM_OBSTACLES, num_timesteps)
        resulting_velocity_e_vector_x: Tensor,  # shape (BATCH_SIZE, num_timesteps)
        resulting_velocity_e_vector_y: Tensor,  # shape (BATCH_SIZE, num_timesteps)
        resulting_acceleration_e_vector_x: Tensor,  # shape (BATCH_SIZE, num_timesteps)
        resulting_acceleration_e_vector_y: Tensor,  # shape (BATCH_SIZE, num_timesteps)
    ) -> Tensor:
        return cast(
            Tensor,
            1
            * torch.norm(
                torch.concatenate(
                    [
                        resulting_collision_e_vector_x.flatten(start_dim=1),
                        resulting_collision_e_vector_y.flatten(start_dim=1),
                    ],
                    dim=1,
                ),
                dim=1,
            )
            + torch.norm(
                torch.concatenate(
                    [resulting_velocity_e_vector_x, resulting_velocity_e_vector_y],
                    dim=1,
                ),
                dim=1,
            )
            + torch.norm(
                torch.concatenate(
                    [
                        resulting_acceleration_e_vector_x,
                        resulting_acceleration_e_vector_y,
                    ],
                    dim=1,
                ),
                dim=1,
            ),
        )  # shape (BATCH_SIZE,)

    @overload
    def coefficients_to_trajectory(
        self,
        coefficients_x: Tensor,  # shape (BATCH_SIZE, 11)
        coefficients_y: Tensor,  # shape (BATCH_SIZE, 11)
        position_only: Literal[True],
    ) -> Tuple[Tensor, Tensor]: ...

    @overload
    def coefficients_to_trajectory(
        self,
        coefficients_x: Tensor,  # shape (BATCH_SIZE, 11)
        coefficients_y: Tensor,  # shape (BATCH_SIZE, 11)
        position_only: Literal[False],
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]: ...

    @overload
    def coefficients_to_trajectory(
        self,
        coefficients_x: Tensor,  # shape (BATCH_SIZE, 11)
        coefficients_y: Tensor,  # shape (BATCH_SIZE, 11)
        position_only: bool = False,
    ) -> Union[Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor], Tuple[Tensor, Tensor]]: ...

    @conditional_compilation(COMPILE)
    def coefficients_to_trajectory(
        self,
        coefficients_x: Tensor,  # shape (BATCH_SIZE, 11)
        coefficients_y: Tensor,  # shape (BATCH_SIZE, 11)
        position_only: bool = False,
    ) -> Union[Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor], Tuple[Tensor, Tensor]]:
        ego_position_trajectory_x = (
            self.BERNSTEIN_POLYNOMIALS @ coefficients_x.T
        ).T  # shape (BATCH_SIZE, num_timesteps)
        ego_position_trajectory_y = (
            self.BERNSTEIN_POLYNOMIALS @ coefficients_y.T
        ).T  # shape (BATCH_SIZE, num_timesteps)

        if position_only:
            return (
                ego_position_trajectory_x,
                ego_position_trajectory_y,
            )

        ego_velocity_trajectory_x = (
            self.BERNSTEIN_POLYNOMIALS_FIRST_DIFFERENTIAL @ coefficients_x.T
        ).T  # shape (BATCH_SIZE, num_timesteps)
        ego_velocity_trajectory_y = (
            self.BERNSTEIN_POLYNOMIALS_FIRST_DIFFERENTIAL @ coefficients_y.T
        ).T  # shape (BATCH_SIZE, num_timesteps)

        ego_acceleration_trajectory_x = (
            self.BERNSTEIN_POLYNOMIALS_SECOND_DIFFERENTIAL @ coefficients_x.T
        ).T  # shape (BATCH_SIZE, num_timesteps)
        ego_acceleration_trajectory_y = (
            self.BERNSTEIN_POLYNOMIALS_SECOND_DIFFERENTIAL @ coefficients_y.T
        ).T  # shape (BATCH_SIZE, num_timesteps)

        return (
            ego_position_trajectory_x,  # shape (BATCH_SIZE, num_timesteps)
            ego_position_trajectory_y,  # shape (BATCH_SIZE, num_timesteps)
            ego_velocity_trajectory_x,  # shape (BATCH_SIZE, num_timesteps)
            ego_velocity_trajectory_y,  # shape (BATCH_SIZE, num_timesteps)
            ego_acceleration_trajectory_x,  # shape (BATCH_SIZE, num_timesteps)
            ego_acceleration_trajectory_y,  # shape (BATCH_SIZE, num_timesteps)
        )

    @conditional_compilation(COMPILE)
    def project_initial_solution_to_trajectory(
        self,
        ego_position_solution_x: Tensor,  # shape (BATCH_SIZE, 11)
        ego_position_solution_y: Tensor,  # shape (BATCH_SIZE, 11)
        obstacle_positions_trajectory_x: Tensor,  # shape (BATCH_SIZE, NUM_OBSTACLES, num_timesteps)
        obstacle_positions_trajectory_y: Tensor,  # shape (BATCH_SIZE, NUM_OBSTACLES, num_timesteps)
        collision_alpha_vector: Tensor,  # shape (BATCH_SIZE, NUM_OBSTACLES, num_timesteps)
        collision_d_vector: Tensor,  # shape (BATCH_SIZE, NUM_OBSTACLES, num_timesteps)
        velocity_alpha_vector: Tensor,  # shape (BATCH_SIZE, num_timesteps)
        velocity_d_vector: Tensor,  # shape (BATCH_SIZE, num_timesteps)
        acceleration_alpha_vector: Tensor,  # shape (BATCH_SIZE, num_timesteps)
        acceleration_d_vector: Tensor,  # shape (BATCH_SIZE, num_timesteps)
        lambda_x: Tensor,  # shape (BATCH_SIZE, 11)
        lambda_y: Tensor,  # shape (BATCH_SIZE, 11)
        boundary_conditions_x: Tensor,  # shape (BATCH_SIZE, 4)
        boundary_conditions_y: Tensor,  # shape (BATCH_SIZE, 4)
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        collision_e_vector_x = (
            obstacle_positions_trajectory_x
            + (
                collision_d_vector * torch.cos(collision_alpha_vector) * self.OBSTACLE_ELLIPSE_SEMI_MINOR_AXIS
            )  # shape (BATCH_SIZE, NUM_OBSTACLES, num_timesteps)
        )  # shape (BATCH_SIZE, NUM_OBSTACLES, num_timesteps)

        collision_e_vector_y = (
            obstacle_positions_trajectory_y
            + (
                collision_d_vector * torch.sin(collision_alpha_vector) * self.OBSTACLE_ELLIPSE_SEMI_MAJOR_AXIS
            )  # shape (BATCH_SIZE, NUM_OBSTACLES, num_timesteps)
        )  # shape (BATCH_SIZE, NUM_OBSTACLES, num_timesteps)

        velocity_e_vector_x = velocity_d_vector * torch.cos(velocity_alpha_vector)  # shape (BATCH_SIZE, num_timesteps)
        velocity_e_vector_y = velocity_d_vector * torch.sin(velocity_alpha_vector)  # shape (BATCH_SIZE, num_timesteps)

        acceleration_e_vector_x = acceleration_d_vector * torch.cos(
            acceleration_alpha_vector
        )  # shape (BATCH_SIZE, num_timesteps)
        acceleration_e_vector_y = acceleration_d_vector * torch.sin(
            acceleration_alpha_vector
        )  # shape (BATCH_SIZE, num_timesteps)

        linear_cost_x = (
            -1 * self.RHO_PROJECTION * (self.A_PROJECTION.T @ ego_position_solution_x.T).T
            - lambda_x
            - self.RHO_COLLISION * (self.A_COLLISION.reshape(-1, 11).T @ collision_e_vector_x.flatten(start_dim=1).T).T
            - self.RHO_INEQUALITIES * (self.A_VELOCITY.T @ velocity_e_vector_x.T).T
            - self.RHO_INEQUALITIES * (self.A_ACCELERATION.T @ acceleration_e_vector_x.T).T
        )  # shape (BATCH_SIZE, 11)
        linear_cost_y = (
            -1 * self.RHO_PROJECTION * (self.A_PROJECTION.T @ ego_position_solution_y.T).T
            - lambda_y
            - self.RHO_COLLISION * (self.A_COLLISION.reshape(-1, 11).T @ collision_e_vector_y.flatten(start_dim=1).T).T
            - self.RHO_INEQUALITIES * (self.A_VELOCITY.T @ velocity_e_vector_y.T).T
            - self.RHO_INEQUALITIES * (self.A_ACCELERATION.T @ acceleration_e_vector_y.T).T
        )  # shape (BATCH_SIZE, 11)

        primal_solution_x = (
            self.INVERSE_PROJECTION_COST_MATRIX_X
            @ torch.concatenate([-1 * linear_cost_x, boundary_conditions_x], dim=1).T  # shape (15, BATCH_SIZE)
        ).T  # shape (BATCH_SIZE, 15)
        primal_solution_y = (
            self.INVERSE_PROJECTION_COST_MATRIX_Y
            @ torch.concatenate([-1 * linear_cost_y, boundary_conditions_y], dim=1).T  # shape (15, BATCH_SIZE)
        ).T  # shape (BATCH_SIZE, 15)

        primal_solution_x = primal_solution_x[:, :11]  # shape (BATCH_SIZE, 11)
        primal_solution_y = primal_solution_y[:, :11]  # shape (BATCH_SIZE, 11)

        (
            ego_position_trajectory_x,
            ego_position_trajectory_y,
            ego_velocity_trajectory_x,
            ego_velocity_trajectory_y,
            ego_acceleration_trajectory_x,
            ego_acceleration_trajectory_y,
        ) = self.coefficients_to_trajectory(
            coefficients_x=primal_solution_x,
            coefficients_y=primal_solution_y,
            position_only=False,
        )

        return (
            primal_solution_x,  # shape (BATCH_SIZE, 11)
            primal_solution_y,  # shape (BATCH_SIZE, 11)
            ego_position_trajectory_x,  # shape (BATCH_SIZE, num_timesteps)
            ego_position_trajectory_y,  # shape (BATCH_SIZE, num_timesteps)
            ego_velocity_trajectory_x,  # shape (BATCH_SIZE, num_timesteps)
            ego_velocity_trajectory_y,  # shape (BATCH_SIZE, num_timesteps)
            ego_acceleration_trajectory_x,  # shape (BATCH_SIZE, num_timesteps)
            ego_acceleration_trajectory_y,  # shape (BATCH_SIZE, num_timesteps)
        )

    @conditional_compilation(COMPILE)
    def convert_warm_trajectory_to_initial_solution(
        self,
        warm_ego_position_trajectory_x: Tensor,  # shape (BATCH_SIZE, num_timesteps)
        warm_ego_position_trajectory_y: Tensor,  # shape (BATCH_SIZE, num_timesteps)
    ) -> Tuple[Tensor, Tensor]:
        linear_cost_x = (
            -1 * (self.BERNSTEIN_POLYNOMIALS.T @ warm_ego_position_trajectory_x.T).T
        )  # shape (BATCH_SIZE, 11)
        linear_cost_y = (
            -1 * (self.BERNSTEIN_POLYNOMIALS.T @ warm_ego_position_trajectory_y.T).T
        )  # shape (BATCH_SIZE, 11)

        initial_solution_x = (self.INVERSE_INITIAL_COST_MATRIX_X @ -linear_cost_x.T).T  # shape (BATCH_SIZE, 11)

        initial_solution_y = (self.INVERSE_INITIAL_COST_MATRIX_Y @ -linear_cost_y.T).T  # shape (BATCH_SIZE, 11)

        return (
            initial_solution_x,  # shape (BATCH_SIZE, 11)
            initial_solution_y,  # shape (BATCH_SIZE, 11)
        )

    @conditional_compilation(COMPILE)
    def project_trajectory(
        self,
        initial_ego_position_x: Tensor,  # shape (BATCH_SIZE,)
        initial_ego_position_y: Tensor,  # shape (BATCH_SIZE,)
        initial_ego_velocity_x: Tensor,  # shape (BATCH_SIZE,)
        initial_ego_velocity_y: Tensor,  # shape (BATCH_SIZE,)
        initial_ego_acceleration_x: Tensor,  # shape (BATCH_SIZE,)
        initial_ego_acceleration_y: Tensor,  # shape (BATCH_SIZE,)
        final_ego_position_x: Tensor,  # shape (BATCH_SIZE,)
        final_ego_position_y: Tensor,  # shape (BATCH_SIZE,)
        obstacle_positions_x: Tensor,  # shape (BATCH_SIZE, NUM_OBSTACLES,)
        obstacle_positions_y: Tensor,  # shape (BATCH_SIZE, NUM_OBSTACLES,)
        obstacle_velocities_x: Tensor,  # shape (BATCH_SIZE, NUM_OBSTACLES,)
        obstacle_velocities_y: Tensor,  # shape (BATCH_SIZE, NUM_OBSTACLES,)
        maximum_velocity: float,
        maximum_acceleration: float,
        warm_ego_position_trajectory_x: Optional[Tensor] = None,  # shape (BATCH_SIZE, num_timesteps)
        warm_ego_position_trajectory_y: Optional[Tensor] = None,  # shape (BATCH_SIZE, num_timesteps)
        initial_solution_x: Optional[Tensor] = None,  # shape (BATCH_SIZE, 11)
        initial_solution_y: Optional[Tensor] = None,  # shape (BATCH_SIZE, 11)
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        assert (warm_ego_position_trajectory_x is not None and warm_ego_position_trajectory_y is not None) or (
            initial_solution_x is not None and initial_solution_y is not None
        ), "Either warm_ego_position_trajectory or initial_solution must be provided"

        assert (warm_ego_position_trajectory_x is None and warm_ego_position_trajectory_y is None) or (
            initial_solution_x is None and initial_solution_y is None
        ), "Only one of warm_ego_position_trajectory or initial_solution must be provided"

        boundary_conditions_x = torch.stack(
            [
                initial_ego_position_x,
                initial_ego_velocity_x,
                initial_ego_acceleration_x,
                final_ego_position_x,
            ],
            dim=1,
        )  # shape (BATCH_SIZE, 4)
        boundary_conditions_y = torch.stack(
            [
                initial_ego_position_y,
                initial_ego_velocity_y,
                initial_ego_acceleration_y,
                final_ego_position_y,
            ],
            dim=1,
        )  # shape (BATCH_SIZE, 4)

        if initial_solution_x is None and initial_solution_y is None:
            assert warm_ego_position_trajectory_x is not None and warm_ego_position_trajectory_y is not None
            (
                initial_solution_x,  # shape (BATCH_SIZE, 11)
                initial_solution_y,  # shape (BATCH_SIZE, 11)
            ) = self.convert_warm_trajectory_to_initial_solution(
                warm_ego_position_trajectory_x=warm_ego_position_trajectory_x,
                warm_ego_position_trajectory_y=warm_ego_position_trajectory_y,
            )

        assert initial_solution_x is not None and initial_solution_y is not None

        (
            obstacle_positions_trajectory_x,  # shape (BATCH_SIZE, NUM_OBSTACLES, num_timesteps)
            obstacle_positions_trajectory_y,  # shape (BATCH_SIZE, NUM_OBSTACLES, num_timesteps)
        ) = self.predict_obstacle_trajectories(
            obstacle_positions_x=obstacle_positions_x,
            obstacle_positions_y=obstacle_positions_y,
            obstacle_velocities_x=obstacle_velocities_x,
            obstacle_velocities_y=obstacle_velocities_y,
            initial_ego_position_x=initial_ego_position_x,
            initial_ego_position_y=initial_ego_position_y,
        )

        lambda_x = torch.zeros((initial_solution_x.shape[0], 11)).to(self.DEVICE)  # shape (BATCH_SIZE, 11)
        lambda_y = torch.zeros((initial_solution_x.shape[0], 11)).to(self.DEVICE)  # shape (BATCH_SIZE, 11)

        initial_ego_position_trajectory_x = (
            self.BERNSTEIN_POLYNOMIALS @ initial_solution_x.T
        ).T  # shape (BATCH_SIZE, num_timesteps)
        initial_ego_position_trajectory_y = (
            self.BERNSTEIN_POLYNOMIALS @ initial_solution_y.T
        ).T  # shape (BATCH_SIZE, num_timesteps)
        initial_ego_velocity_trajectory_x = (
            self.BERNSTEIN_POLYNOMIALS_FIRST_DIFFERENTIAL @ initial_solution_x.T
        ).T  # shape (BATCH_SIZE, num_timesteps)
        initial_ego_velocity_trajectory_y = (
            self.BERNSTEIN_POLYNOMIALS_FIRST_DIFFERENTIAL @ initial_solution_y.T
        ).T  # shape (BATCH_SIZE, num_timesteps)
        initial_ego_acceleration_trajectory_x = (
            self.BERNSTEIN_POLYNOMIALS_SECOND_DIFFERENTIAL @ initial_solution_x.T
        ).T  # shape (BATCH_SIZE, num_timesteps)
        initial_ego_acceleration_trajectory_y = (
            self.BERNSTEIN_POLYNOMIALS_SECOND_DIFFERENTIAL @ initial_solution_y.T
        ).T  # shape (BATCH_SIZE, num_timesteps)

        (
            initial_collision_alpha_vector,  # shape (BATCH_SIZE, NUM_OBSTACLES, num_timesteps)
            initial_collision_d_vector,  # shape (BATCH_SIZE, NUM_OBSTACLES, num_timesteps)
            initial_velocity_alpha_vector,  # shape (BATCH_SIZE, num_timesteps)
            initial_velocity_d_vector,  # shape (BATCH_SIZE, num_timesteps)
            initial_acceleration_alpha_vector,  # shape (BATCH_SIZE, num_timesteps)
            initial_acceleration_d_vector,  # shape (BATCH_SIZE, num_timesteps)
            initial_lambda_x,  # shape (BATCH_SIZE, 11)
            initial_lambda_y,  # shape (BATCH_SIZE, 11)
            _,
            _,
            _,
            _,
            _,
            _,
        ) = self.generate_alpha_d_lambda(
            ego_position_trajectory_x=initial_ego_position_trajectory_x,
            ego_position_trajectory_y=initial_ego_position_trajectory_y,
            ego_velocity_trajectory_x=initial_ego_velocity_trajectory_x,
            ego_velocity_trajectory_y=initial_ego_velocity_trajectory_y,
            ego_acceleration_trajectory_x=initial_ego_acceleration_trajectory_x,
            ego_acceleration_trajectory_y=initial_ego_acceleration_trajectory_y,
            obstacle_positions_trajectory_x=obstacle_positions_trajectory_x,
            obstacle_positions_trajectory_y=obstacle_positions_trajectory_y,
            maximum_velocity=maximum_velocity,
            maximum_acceleration=maximum_acceleration,
            lambda_x=lambda_x,
            lambda_y=lambda_y,
        )

        # @torch.compile(options={"triton.cudagraphs": True}, fullgraph=True)
        def _project_trajectory(carry: Carry) -> Tuple[Carry, Output]:
            (
                next_primal_solution_x,  # shape (BATCH_SIZE, 11)
                next_primal_solution_y,  # shape (BATCH_SIZE, 11)
                next_ego_position_trajectory_x,  # shape (BATCH_SIZE, num_timesteps)
                next_ego_position_trajectory_y,  # shape (BATCH_SIZE, num_timesteps)
                next_ego_velocity_trajectory_x,  # shape (BATCH_SIZE, num_timesteps)
                next_ego_velocity_trajectory_y,  # shape (BATCH_SIZE, num_timesteps)
                next_ego_acceleration_trajectory_x,  # shape (BATCH_SIZE, num_timesteps)
                next_ego_acceleration_trajectory_y,  # shape (BATCH_SIZE, num_timesteps)
            ) = self.project_initial_solution_to_trajectory(
                ego_position_solution_x=carry.primal_solution_x,
                ego_position_solution_y=carry.primal_solution_y,
                obstacle_positions_trajectory_x=obstacle_positions_trajectory_x,
                obstacle_positions_trajectory_y=obstacle_positions_trajectory_y,
                collision_alpha_vector=carry.collision_alpha_vector,
                collision_d_vector=carry.collision_d_vector,
                velocity_alpha_vector=carry.velocity_alpha_vector,
                velocity_d_vector=carry.velocity_d_vector,
                acceleration_alpha_vector=carry.acceleration_alpha_vector,
                acceleration_d_vector=carry.acceleration_d_vector,
                lambda_x=carry.lambda_x,
                lambda_y=carry.lambda_y,
                boundary_conditions_x=boundary_conditions_x,
                boundary_conditions_y=boundary_conditions_y,
            )

            (
                next_collision_alpha_vector,  # shape (BATCH_SIZE, NUM_OBSTACLES, num_timesteps)
                next_collision_d_vector,  # shape (BATCH_SIZE, NUM_OBSTACLES, num_timesteps)
                next_velocity_alpha_vector,  # shape (BATCH_SIZE, num_timesteps)
                next_velocity_d_vector,  # shape (BATCH_SIZE, num_timesteps)
                next_acceleration_alpha_vector,  # shape (BATCH_SIZE, num_timesteps)
                next_acceleration_d_vector,  # shape (BATCH_SIZE, num_timesteps)
                next_lambda_x,  # shape (BATCH_SIZE, 11)
                next_lambda_y,  # shape (BATCH_SIZE, 11)
                next_resulting_collision_e_vector_x,  # shape (BATCH_SIZE, NUM_OBSTACLES, num_timesteps)
                next_resulting_collision_e_vector_y,  # shape (BATCH_SIZE, NUM_OBSTACLES, num_timesteps)
                next_resulting_velocity_e_vector_x,  # shape (BATCH_SIZE, num_timesteps)
                next_resulting_velocity_e_vector_y,  # shape (BATCH_SIZE, num_timesteps)
                next_resulting_acceleration_e_vector_x,  # shape (BATCH_SIZE, num_timesteps)
                next_resulting_acceleration_e_vector_y,  # shape (BATCH_SIZE, num_timesteps)
            ) = self.generate_alpha_d_lambda(
                ego_position_trajectory_x=next_ego_position_trajectory_x,
                ego_position_trajectory_y=next_ego_position_trajectory_y,
                ego_velocity_trajectory_x=next_ego_velocity_trajectory_x,
                ego_velocity_trajectory_y=next_ego_velocity_trajectory_y,
                ego_acceleration_trajectory_x=next_ego_acceleration_trajectory_x,
                ego_acceleration_trajectory_y=next_ego_acceleration_trajectory_y,
                obstacle_positions_trajectory_x=obstacle_positions_trajectory_x,
                obstacle_positions_trajectory_y=obstacle_positions_trajectory_y,
                maximum_velocity=maximum_velocity,
                maximum_acceleration=maximum_acceleration,
                lambda_x=carry.lambda_x,
                lambda_y=carry.lambda_y,
            )

            next_error_residuals = self.compute_error_residuals(
                resulting_collision_e_vector_x=next_resulting_collision_e_vector_x,
                resulting_collision_e_vector_y=next_resulting_collision_e_vector_y,
                resulting_velocity_e_vector_x=next_resulting_velocity_e_vector_x,
                resulting_velocity_e_vector_y=next_resulting_velocity_e_vector_y,
                resulting_acceleration_e_vector_x=next_resulting_acceleration_e_vector_x,
                resulting_acceleration_e_vector_y=next_resulting_acceleration_e_vector_y,
            )  # shape (BATCH_SIZE,)

            return Carry(
                primal_solution_x=next_primal_solution_x,
                primal_solution_y=next_primal_solution_y,
                collision_alpha_vector=next_collision_alpha_vector,
                collision_d_vector=next_collision_d_vector,
                velocity_alpha_vector=next_velocity_alpha_vector,
                velocity_d_vector=next_velocity_d_vector,
                acceleration_alpha_vector=next_acceleration_alpha_vector,
                acceleration_d_vector=next_acceleration_d_vector,
                lambda_x=next_lambda_x,
                lambda_y=next_lambda_y,
            ), Output(
                ego_position_trajectory_x=next_ego_position_trajectory_x,
                ego_position_trajectory_y=next_ego_position_trajectory_y,
                ego_velocity_trajectory_x=next_ego_velocity_trajectory_x,
                ego_velocity_trajectory_y=next_ego_velocity_trajectory_y,
                ego_acceleration_trajectory_x=next_ego_acceleration_trajectory_x,
                ego_acceleration_trajectory_y=next_ego_acceleration_trajectory_y,
                error_residuals=next_error_residuals,
            )

        # Run the projection loop
        outputs: List[Output] = []
        carry = Carry(
            primal_solution_x=initial_solution_x,
            primal_solution_y=initial_solution_y,
            collision_alpha_vector=initial_collision_alpha_vector,
            collision_d_vector=initial_collision_d_vector,
            velocity_alpha_vector=initial_velocity_alpha_vector,
            velocity_d_vector=initial_velocity_d_vector,
            acceleration_alpha_vector=initial_acceleration_alpha_vector,
            acceleration_d_vector=initial_acceleration_d_vector,
            lambda_x=initial_lambda_x,
            lambda_y=initial_lambda_y,
        )
        for _ in range(self.MAX_PROJECTION_ITERATIONS):
            carry, output = _project_trajectory(carry)
            outputs.append(output)

        return (
            carry.primal_solution_x,  # shape (BATCH_SIZE, 11)
            carry.primal_solution_y,  # shape (BATCH_SIZE, 11)
            outputs[-1].ego_position_trajectory_x,  # shape (BATCH_SIZE, num_timesteps)
            outputs[-1].ego_position_trajectory_y[-1],  # shape (BATCH_SIZE, num_timesteps)
            outputs[-1].ego_velocity_trajectory_x[-1],  # shape (BATCH_SIZE, num_timesteps)
            outputs[-1].ego_velocity_trajectory_y[-1],  # shape (BATCH_SIZE, num_timesteps)
            outputs[-1].ego_acceleration_trajectory_x[-1],  # shape (BATCH_SIZE, num_timesteps)
            outputs[-1].ego_acceleration_trajectory_y[-1],  # shape (BATCH_SIZE, num_timesteps)
            outputs[-1].error_residuals,  # shape (MAX_PROJECTION_ITERATIONS, BATCH_SIZE,)
        )
