from dataclasses import dataclass
from functools import partial
from typing import Tuple, cast

import jax
import jax.lax as lax
import jax.numpy as jnp
from jax import Array

from .jax_bernstein_polynomials import bernstein_coeff_order10_new_jax as bernstein_polynomials_generator


@partial(
    jax.tree_util.register_dataclass,
    data_fields=[
        "primal_solution_x",
        "primal_solution_y",
        "collision_alpha_vector",
        "collision_d_vector",
        "velocity_alpha_vector",
        "velocity_d_vector",
        "acceleration_alpha_vector",
        "acceleration_d_vector",
        "lambda_x",
        "lambda_y",
    ],
    meta_fields=[],
)
@dataclass
class LaxCarry:
    primal_solution_x: Array  # shape (BATCH_SIZE, 11)
    primal_solution_y: Array  # shape (BATCH_SIZE, 11)
    collision_alpha_vector: Array  # shape (BATCH_SIZE, NUM_OBSTACLES, num_timesteps)
    collision_d_vector: Array  # shape (BATCH_SIZE, NUM_OBSTACLES, num_timesteps)
    velocity_alpha_vector: Array  # shape (BATCH_SIZE, num_timesteps)
    velocity_d_vector: Array  # shape (BATCH_SIZE, num_timesteps)
    acceleration_alpha_vector: Array  # shape (BATCH_SIZE, num_timesteps)
    acceleration_d_vector: Array  # shape (BATCH_SIZE, num_timesteps)
    lambda_x: Array  # shape (BATCH_SIZE, 11)
    lambda_y: Array  # shape (BATCH_SIZE, 11)


@partial(
    jax.tree_util.register_dataclass,
    data_fields=[
        "ego_position_trajectory_x",
        "ego_position_trajectory_y",
        "ego_velocity_trajectory_x",
        "ego_velocity_trajectory_y",
        "ego_acceleration_trajectory_x",
        "ego_acceleration_trajectory_y",
        "error_residuals",
    ],
    meta_fields=[],
)
@dataclass
class LaxOutput:
    ego_position_trajectory_x: Array  # shape (BATCH_SIZE, num_timesteps)
    ego_position_trajectory_y: Array  # shape (BATCH_SIZE, num_timesteps)
    ego_velocity_trajectory_x: Array  # shape (BATCH_SIZE, num_timesteps)
    ego_velocity_trajectory_y: Array  # shape (BATCH_SIZE, num_timesteps)
    ego_acceleration_trajectory_x: Array  # shape (BATCH_SIZE, num_timesteps)
    ego_acceleration_trajectory_y: Array  # shape (BATCH_SIZE, num_timesteps)
    error_residuals: Array  # shape (BATCH_SIZE,)


class ProjectionGuidance:
    def __init__(
        self,
        num_obstacles: int,
        num_timesteps: int,
        total_time: float,
        batch_size: int,
        obstacle_ellipse_semi_major_axis: float,
        obstacle_ellipse_semi_minor_axis: float,
        max_projection_iterations: int,
    ):
        self.NUM_OBSTACLES = num_obstacles
        self.OBSTACLE_ELLIPSE_SEMI_MAJOR_AXIS = obstacle_ellipse_semi_major_axis
        self.OBSTACLE_ELLIPSE_SEMI_MINOR_AXIS = obstacle_ellipse_semi_minor_axis

        self.DT = jnp.linspace(0, total_time, num_timesteps)  # shape (num_timesteps,)

        self.BATCH_SIZE = batch_size

        self.MAX_PROJECTION_ITERATIONS = max_projection_iterations

        (
            self.BERNSTEIN_POLYNOMIALS,  # shape (num_timesteps, 11)
            self.BERNSTEIN_POLYNOMIALS_FIRST_DIFFERENTIAL,  # shape (num_timesteps, 11)
            self.BERNSTEIN_POLYNOMIALS_SECOND_DIFFERENTIAL,  # shape (num_timesteps, 11)
        ) = bernstein_polynomials_generator(tmin=0, tmax=total_time, t_actual=self.DT)

        INITIAL_COST_MATRIX = (
            self.BERNSTEIN_POLYNOMIALS.T @ self.BERNSTEIN_POLYNOMIALS
            + 1e-4 * jnp.identity(11)
            + (1 * self.BERNSTEIN_POLYNOMIALS_SECOND_DIFFERENTIAL.T @ self.BERNSTEIN_POLYNOMIALS_SECOND_DIFFERENTIAL)
        )  # shape (11, 11)

        self.INVERSE_INITIAL_COST_MATRIX_X = cast(Array, jnp.linalg.inv(INITIAL_COST_MATRIX))  # shape (11, 11)

        self.INVERSE_INITIAL_COST_MATRIX_Y = (
            self.INVERSE_INITIAL_COST_MATRIX_X  # shape (11, 11)
        )

        self.RHO_PROJECTION = 1.0
        self.RHO_COLLISION = 1.0
        self.RHO_INEQUALITIES = 1.0

        self.A_EQUALITY = jnp.stack(
            [
                self.BERNSTEIN_POLYNOMIALS[0],  # shape (11,)
                self.BERNSTEIN_POLYNOMIALS_FIRST_DIFFERENTIAL[0],  # shape (11,)
                self.BERNSTEIN_POLYNOMIALS_SECOND_DIFFERENTIAL[0],  # shape (11,)
                self.BERNSTEIN_POLYNOMIALS[-1],  # shape (11,)
            ],
            axis=0,
        )  # shape (4, 11)
        self.A_PROJECTION = jnp.identity(11)  # shape (11, 11)
        self.A_COLLISION = jnp.tile(
            self.BERNSTEIN_POLYNOMIALS[jnp.newaxis, :, :], (self.NUM_OBSTACLES, 1, 1)
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

        PROJECTION_COST_MATRIX_X = jnp.concatenate(
            [
                jnp.concatenate(
                    [
                        PROJECTION_COST_MATRIX_X,
                        self.A_EQUALITY.T,
                    ],
                    axis=1,
                ),  # shape (11, 15)
                jnp.concatenate(
                    [
                        self.A_EQUALITY,
                        jnp.zeros((self.A_EQUALITY.shape[0], self.A_EQUALITY.shape[0])),
                    ],
                    axis=1,
                ),  # shape (4, 15)
            ],
            axis=0,
        )  # shape (15, 15)

        self.INVERSE_PROJECTION_COST_MATRIX_X = cast(Array, jnp.linalg.inv(PROJECTION_COST_MATRIX_X))  # shape (15, 15)
        self.INVERSE_PROJECTION_COST_MATRIX_Y = (
            self.INVERSE_PROJECTION_COST_MATRIX_X  # shape (15, 15)
        )

    @partial(jax.jit, static_argnums=(0,))
    def predict_obstacle_trajectories(
        self,
        obstacle_positions_x: Array,  # shape (NUM_OBSTACLES,)
        obstacle_positions_y: Array,  # shape (NUM_OBSTACLES,)
        obstacle_velocities_x: Array,  # shape (NUM_OBSTACLES,)
        obstacle_velocities_y: Array,  # shape (NUM_OBSTACLES,)
        initial_ego_position_x: float,
        initial_ego_position_y: float,
    ) -> Tuple[Array, Array]:
        obstacle_positions_trajectory_x = (
            obstacle_positions_x[:, jnp.newaxis] + obstacle_velocities_x[:, jnp.newaxis] * self.DT[jnp.newaxis, :]
        )  # shape (NUM_OBSTACLES, num_timesteps)

        obstacle_positions_trajectory_y = (
            obstacle_positions_y[:, jnp.newaxis] + obstacle_velocities_y[:, jnp.newaxis] * self.DT[jnp.newaxis, :]
        )  # shape (NUM_OBSTACLES, num_timesteps)

        distances = jnp.sqrt(
            (initial_ego_position_x - obstacle_positions_trajectory_x) ** 2
            + (initial_ego_position_y - obstacle_positions_trajectory_y) ** 2
        )  # shape (NUM_OBSTACLES, num_timesteps)

        distances = jnp.argsort(distances)  # sort, closest to farthest # shape (NUM_OBSTACLES, num_timesteps)

        return (
            obstacle_positions_trajectory_x.take(distances),  # shape (NUM_OBSTACLES, num_timesteps)
            obstacle_positions_trajectory_y.take(distances),  # shape (NUM_OBSTACLES, num_timesteps)
        )

    @partial(jax.jit, static_argnums=(0,))
    def generate_alpha_d_lambda(
        self,
        ego_position_trajectory_x: Array,  # shape (BATCH_SIZE, num_timesteps)
        ego_position_trajectory_y: Array,  # shape (BATCH_SIZE, num_timesteps)
        ego_velocity_trajectory_x: Array,  # shape (BATCH_SIZE, num_timesteps)
        ego_velocity_trajectory_y: Array,  # shape (BATCH_SIZE, num_timesteps)
        ego_acceleration_trajectory_x: Array,  # shape (BATCH_SIZE, num_timesteps)
        ego_acceleration_trajectory_y: Array,  # shape (BATCH_SIZE, num_timesteps)
        obstacle_positions_trajectory_x: Array,  # shape (NUM_OBSTACLES, num_timesteps)
        obstacle_positions_trajectory_y: Array,  # shape (NUM_OBSTACLES, num_timesteps)
        maximum_velocity: float,
        maximum_acceleration: float,
        lambda_x: Array,  # shape (BATCH_SIZE, 11)
        lambda_y: Array,  # shape (BATCH_SIZE, 11)
    ) -> Tuple[
        Array,
        Array,
        Array,
        Array,
        Array,
        Array,
        Array,
        Array,
        Array,
        Array,
        Array,
        Array,
        Array,
        Array,
    ]:
        ## Construct collision d vector
        ego_to_obstacle_distance_x = (
            ego_position_trajectory_x[:, jnp.newaxis, :] - obstacle_positions_trajectory_x[jnp.newaxis, :, :]
        )  # shape (BATCH_SIZE, NUM_OBSTACLES, num_timesteps)

        ego_to_obstacle_distance_y = (
            ego_position_trajectory_y[:, jnp.newaxis, :] - obstacle_positions_trajectory_y[jnp.newaxis, :, :]
        )  # shape (BATCH_SIZE, NUM_OBSTACLES, num_timesteps)

        collision_alpha_vector = jnp.arctan2(
            self.OBSTACLE_ELLIPSE_SEMI_MINOR_AXIS * ego_to_obstacle_distance_y,
            self.OBSTACLE_ELLIPSE_SEMI_MAJOR_AXIS * ego_to_obstacle_distance_x,
        )  # shape (BATCH_SIZE, NUM_OBSTACLES, num_timesteps)

        collision_d_vector = (
            1.0
            * self.RHO_COLLISION
            * (
                self.OBSTACLE_ELLIPSE_SEMI_MAJOR_AXIS * ego_to_obstacle_distance_x * jnp.cos(collision_alpha_vector)
                + self.OBSTACLE_ELLIPSE_SEMI_MINOR_AXIS * ego_to_obstacle_distance_y * jnp.sin(collision_alpha_vector)
            )
        ) / (
            1.0
            * self.RHO_COLLISION
            * (
                (self.OBSTACLE_ELLIPSE_SEMI_MAJOR_AXIS**2) * (jnp.cos(collision_alpha_vector) ** 2)
                + (self.OBSTACLE_ELLIPSE_SEMI_MINOR_AXIS**2) * (jnp.sin(collision_alpha_vector) ** 2)
            )
        )  # shape (BATCH_SIZE, NUM_OBSTACLES, num_timesteps)

        collision_d_vector = jnp.clip(collision_d_vector, min=1)  # shape (BATCH_SIZE, NUM_OBSTACLES, num_timesteps)

        resulting_collision_e_vector_x = (
            ego_to_obstacle_distance_x
            - self.OBSTACLE_ELLIPSE_SEMI_MAJOR_AXIS * collision_d_vector * jnp.cos(collision_alpha_vector)
        )  # shape (BATCH_SIZE, NUM_OBSTACLES, num_timesteps)

        resulting_collision_e_vector_y = (
            ego_to_obstacle_distance_y
            - self.OBSTACLE_ELLIPSE_SEMI_MINOR_AXIS * collision_d_vector * jnp.sin(collision_alpha_vector)
        )  # shape (BATCH_SIZE, NUM_OBSTACLES, num_timesteps)

        ## Construct velocity d vector

        velocity_alpha_vector = jnp.arctan2(
            ego_velocity_trajectory_y, ego_velocity_trajectory_x
        )  # shape (BATCH_SIZE, num_timesteps)

        velocity_d_vector = (
            1.0
            * self.RHO_INEQUALITIES
            * (
                ego_velocity_trajectory_x * jnp.cos(velocity_alpha_vector)
                + ego_velocity_trajectory_y * jnp.sin(velocity_alpha_vector)
            )
        ) / (
            1.0 * self.RHO_INEQUALITIES * (jnp.cos(velocity_alpha_vector) ** 2 + jnp.sin(velocity_alpha_vector) ** 2)
        )  # shape (BATCH_SIZE, num_timesteps)

        velocity_d_vector = jnp.clip(velocity_d_vector, max=maximum_velocity)  # shape (BATCH_SIZE, num_timesteps)

        resulting_velocity_e_vector_x = ego_velocity_trajectory_x - velocity_d_vector * jnp.cos(
            velocity_alpha_vector
        )  # shape (BATCH_SIZE, num_timesteps)

        resulting_velocity_e_vector_y = ego_velocity_trajectory_y - velocity_d_vector * jnp.sin(
            velocity_alpha_vector
        )  # shape (BATCH_SIZE, num_timesteps)

        ## Construct acceleration d vector
        acceleration_alpha_vector = jnp.arctan2(
            ego_acceleration_trajectory_y, ego_acceleration_trajectory_x
        )  # shape (BATCH_SIZE, num_timesteps)

        acceleration_d_vector = (
            1.0
            * self.RHO_INEQUALITIES
            * (
                ego_acceleration_trajectory_x * jnp.cos(acceleration_alpha_vector)
                + ego_acceleration_trajectory_y * jnp.sin(acceleration_alpha_vector)
            )
        ) / (
            1.0
            * self.RHO_INEQUALITIES
            * (jnp.cos(acceleration_alpha_vector) ** 2 + jnp.sin(acceleration_alpha_vector) ** 2)
        )  # shape (BATCH_SIZE, num_timesteps)

        acceleration_d_vector = jnp.clip(
            acceleration_d_vector, max=maximum_acceleration
        )  # shape (BATCH_SIZE, num_timesteps)

        resulting_acceleration_e_vector_x = ego_acceleration_trajectory_x - acceleration_d_vector * jnp.cos(
            acceleration_alpha_vector
        )  # shape (BATCH_SIZE, num_timesteps)

        resulting_acceleration_e_vector_y = ego_acceleration_trajectory_y - acceleration_d_vector * jnp.sin(
            acceleration_alpha_vector
        )  # shape (BATCH_SIZE, num_timesteps)

        updated_lambda_x = (
            lambda_x
            - self.RHO_COLLISION
            * (self.A_COLLISION.reshape(-1, 11).T @ resulting_collision_e_vector_x.reshape(self.BATCH_SIZE, -1).T).T
            - self.RHO_INEQUALITIES * (self.A_VELOCITY.T @ resulting_velocity_e_vector_x.T).T
            - self.RHO_INEQUALITIES * (self.A_ACCELERATION.T @ resulting_acceleration_e_vector_x.T).T
        )  # shape (BATCH_SIZE, 11)

        updated_lambda_y = (
            lambda_y
            - self.RHO_COLLISION
            * (self.A_COLLISION.reshape(-1, 11).T @ resulting_collision_e_vector_y.reshape(self.BATCH_SIZE, -1).T).T
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

    @partial(jax.jit, static_argnums=(0,))
    def compute_error_residuals(
        self,
        resulting_collision_e_vector_x: Array,  # shape (BATCH_SIZE, NUM_OBSTACLES, num_timesteps)
        resulting_collision_e_vector_y: Array,  # shape (BATCH_SIZE, NUM_OBSTACLES, num_timesteps)
        resulting_velocity_e_vector_x: Array,  # shape (BATCH_SIZE, num_timesteps)
        resulting_velocity_e_vector_y: Array,  # shape (BATCH_SIZE, num_timesteps)
        resulting_acceleration_e_vector_x: Array,  # shape (BATCH_SIZE, num_timesteps)
        resulting_acceleration_e_vector_y: Array,  # shape (BATCH_SIZE, num_timesteps)
    ) -> Array:
        return cast(
            Array,
            1
            * jnp.linalg.norm(
                jnp.concatenate(
                    [
                        resulting_collision_e_vector_x.reshape(self.BATCH_SIZE, -1),
                        resulting_collision_e_vector_y.reshape(self.BATCH_SIZE, -1),
                    ],
                    axis=1,
                ),
                axis=1,
            )
            + jnp.linalg.norm(
                jnp.concatenate(
                    [resulting_velocity_e_vector_x, resulting_velocity_e_vector_y],
                    axis=1,
                ),
                axis=1,
            )
            + jnp.linalg.norm(
                jnp.concatenate(
                    [
                        resulting_acceleration_e_vector_x,
                        resulting_acceleration_e_vector_y,
                    ],
                    axis=1,
                ),
                axis=1,
            ),
        )  # shape (BATCH_SIZE,)

    @partial(jax.jit, static_argnums=(0,))
    def project_initial_solution_to_trajectory(
        self,
        ego_position_solution_x: Array,  # shape (BATCH_SIZE, 11)
        ego_position_solution_y: Array,  # shape (BATCH_SIZE, 11)
        obstacle_positions_trajectory_x: Array,  # shape (NUM_OBSTACLES, num_timesteps)
        obstacle_positions_trajectory_y: Array,  # shape (NUM_OBSTACLES, num_timesteps)
        collision_alpha_vector: Array,  # shape (BATCH_SIZE, NUM_OBSTACLES, num_timesteps)
        collision_d_vector: Array,  # shape (BATCH_SIZE, NUM_OBSTACLES, num_timesteps)
        velocity_alpha_vector: Array,  # shape (BATCH_SIZE, num_timesteps)
        velocity_d_vector: Array,  # shape (BATCH_SIZE, num_timesteps)
        acceleration_alpha_vector: Array,  # shape (BATCH_SIZE, num_timesteps)
        acceleration_d_vector: Array,  # shape (BATCH_SIZE, num_timesteps)
        lambda_x: Array,  # shape (BATCH_SIZE, 11)
        lambda_y: Array,  # shape (BATCH_SIZE, 11)
        boundary_conditions_x: Array,  # shape (BATCH_SIZE, 4)
        boundary_conditions_y: Array,  # shape (BATCH_SIZE, 4)
    ) -> Tuple[Array, Array, Array, Array, Array, Array, Array, Array]:
        collision_e_vector_x = (
            obstacle_positions_trajectory_x[jnp.newaxis, :, :]
            + (
                collision_d_vector * jnp.cos(collision_alpha_vector) * self.OBSTACLE_ELLIPSE_SEMI_MAJOR_AXIS
            )  # shape (BATCH_SIZE, NUM_OBSTACLES, num_timesteps)
        )  # shape (BATCH_SIZE, NUM_OBSTACLES, num_timesteps)

        collision_e_vector_y = (
            obstacle_positions_trajectory_y[jnp.newaxis, :, :]
            + (
                collision_d_vector * jnp.sin(collision_alpha_vector) * self.OBSTACLE_ELLIPSE_SEMI_MINOR_AXIS
            )  # shape (BATCH_SIZE, NUM_OBSTACLES, num_timesteps)
        )  # shape (BATCH_SIZE, NUM_OBSTACLES, num_timesteps)

        velocity_e_vector_x = velocity_d_vector * jnp.cos(velocity_alpha_vector)  # shape (BATCH_SIZE, num_timesteps)
        velocity_e_vector_y = velocity_d_vector * jnp.sin(velocity_alpha_vector)  # shape (BATCH_SIZE, num_timesteps)

        acceleration_e_vector_x = acceleration_d_vector * jnp.cos(
            acceleration_alpha_vector
        )  # shape (BATCH_SIZE, num_timesteps)
        acceleration_e_vector_y = acceleration_d_vector * jnp.sin(
            acceleration_alpha_vector
        )  # shape (BATCH_SIZE, num_timesteps)

        linear_cost_x = (
            -1 * self.RHO_PROJECTION * (self.A_PROJECTION.T @ ego_position_solution_x.T).T
            - lambda_x
            - self.RHO_COLLISION
            * (self.A_COLLISION.reshape(-1, 11).T @ collision_e_vector_x.reshape(self.BATCH_SIZE, -1).T).T
            - self.RHO_INEQUALITIES * (self.A_VELOCITY.T @ velocity_e_vector_x.T).T
            - self.RHO_INEQUALITIES * (self.A_ACCELERATION.T @ acceleration_e_vector_x.T).T
        )  # shape (BATCH_SIZE, 11)
        linear_cost_y = (
            -1 * self.RHO_PROJECTION * (self.A_PROJECTION.T @ ego_position_solution_y.T).T
            - lambda_y
            - self.RHO_COLLISION
            * (self.A_COLLISION.reshape(-1, 11).T @ collision_e_vector_y.reshape(self.BATCH_SIZE, -1).T).T
            - self.RHO_INEQUALITIES * (self.A_VELOCITY.T @ velocity_e_vector_y.T).T
            - self.RHO_INEQUALITIES * (self.A_ACCELERATION.T @ acceleration_e_vector_y.T).T
        )  # shape (BATCH_SIZE, 11)

        primal_solution_x = (
            self.INVERSE_PROJECTION_COST_MATRIX_X
            @ jnp.concatenate([-1 * linear_cost_x, boundary_conditions_x], axis=1).T  # shape (15, BATCH_SIZE)
        ).T  # shape (BATCH_SIZE, 15)
        primal_solution_y = (
            self.INVERSE_PROJECTION_COST_MATRIX_Y
            @ jnp.concatenate([-1 * linear_cost_y, boundary_conditions_y], axis=1).T  # shape (15, BATCH_SIZE)
        ).T  # shape (BATCH_SIZE, 15)

        primal_solution_x = primal_solution_x[:, :11]  # shape (BATCH_SIZE, 11)
        primal_solution_y = primal_solution_y[:, :11]  # shape (BATCH_SIZE, 11)

        ego_position_trajectory_x = (
            self.BERNSTEIN_POLYNOMIALS @ primal_solution_x.T
        ).T  # shape (BATCH_SIZE, num_timesteps)
        ego_velocity_trajectory_x = (
            self.BERNSTEIN_POLYNOMIALS_FIRST_DIFFERENTIAL @ primal_solution_x.T
        ).T  # shape (BATCH_SIZE, num_timesteps)
        ego_acceleration_trajectory_x = (
            self.BERNSTEIN_POLYNOMIALS_SECOND_DIFFERENTIAL @ primal_solution_x.T
        ).T  # shape (BATCH_SIZE, num_timesteps)

        ego_position_trajectory_y = (
            self.BERNSTEIN_POLYNOMIALS @ primal_solution_y.T
        ).T  # shape (BATCH_SIZE, num_timesteps)
        ego_velocity_trajectory_y = (
            self.BERNSTEIN_POLYNOMIALS_FIRST_DIFFERENTIAL @ primal_solution_y.T
        ).T  # shape (BATCH_SIZE, num_timesteps)
        ego_acceleration_trajectory_y = (
            self.BERNSTEIN_POLYNOMIALS_SECOND_DIFFERENTIAL @ primal_solution_y.T
        ).T  # shape (BATCH_SIZE, num_timesteps)

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

    @partial(jax.jit, static_argnums=(0,))
    def convert_warm_trajectory_to_initial_solution(
        self,
        warm_ego_position_trajectory_x: Array,  # shape (BATCH_SIZE, num_timesteps)
        warm_ego_position_trajectory_y: Array,  # shape (BATCH_SIZE, num_timesteps)
    ) -> Tuple[Array, Array]:
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

    @partial(jax.jit, static_argnums=(0,))
    def project_trajectory(
        self,
        initial_ego_position_x: float,
        initial_ego_position_y: float,
        initial_ego_velocity_x: float,
        initial_ego_velocity_y: float,
        initial_ego_acceleration_x: float,
        initial_ego_acceleration_y: float,
        final_ego_position_x: float,
        final_ego_position_y: float,
        obstacle_positions_x: Array,  # shape (NUM_OBSTACLES,)
        obstacle_positions_y: Array,  # shape (NUM_OBSTACLES,)
        obstacle_velocities_x: Array,  # shape (NUM_OBSTACLES,)
        obstacle_velocities_y: Array,  # shape (NUM_OBSTACLES,)
        maximum_velocity: float,
        maximum_acceleration: float,
        warm_ego_position_trajectory_x: Array,  # shape (BATCH_SIZE, num_timesteps)
        warm_ego_position_trajectory_y: Array,  # shape (BATCH_SIZE, num_timesteps)
    ) -> Tuple[Array, Array, Array, Array, Array, Array, Array, Array, Array]:
        boundary_conditions_x = jnp.stack(
            [
                initial_ego_position_x * jnp.ones(self.BATCH_SIZE),  # shape (BATCH_SIZE,)
                initial_ego_velocity_x * jnp.ones(self.BATCH_SIZE),  # shape (BATCH_SIZE,)
                initial_ego_acceleration_x * jnp.ones(self.BATCH_SIZE),  # shape (BATCH_SIZE,)
                final_ego_position_x * jnp.ones(self.BATCH_SIZE),  # shape (BATCH_SIZE,)
            ],
            axis=1,
        )  # shape (BATCH_SIZE, 4)
        boundary_conditions_y = jnp.stack(
            [
                initial_ego_position_y * jnp.ones(self.BATCH_SIZE),  # shape (BATCH_SIZE,)
                initial_ego_velocity_y * jnp.ones(self.BATCH_SIZE),  # shape (BATCH_SIZE,)
                initial_ego_acceleration_y * jnp.ones(self.BATCH_SIZE),  # shape (BATCH_SIZE,)
                final_ego_position_y * jnp.ones(self.BATCH_SIZE),  # shape (BATCH_SIZE,)
            ],
            axis=1,
        )  # shape (BATCH_SIZE, 4)

        (
            initial_solution_x,  # shape (BATCH_SIZE, 11)
            initial_solution_y,  # shape (BATCH_SIZE, 11)
        ) = self.convert_warm_trajectory_to_initial_solution(
            warm_ego_position_trajectory_x=warm_ego_position_trajectory_x,
            warm_ego_position_trajectory_y=warm_ego_position_trajectory_y,
        )

        (
            obstacle_positions_trajectory_x,  # shape (NUM_OBSTACLES, num_timesteps)
            obstacle_positions_trajectory_y,  # shape (NUM_OBSTACLES, num_timesteps)
        ) = self.predict_obstacle_trajectories(
            obstacle_positions_x=obstacle_positions_x,
            obstacle_positions_y=obstacle_positions_y,
            obstacle_velocities_x=obstacle_velocities_x,
            obstacle_velocities_y=obstacle_velocities_y,
            initial_ego_position_x=initial_ego_position_x,
            initial_ego_position_y=initial_ego_position_y,
        )

        lambda_x = jnp.zeros((self.BATCH_SIZE, 11))  # shape (BATCH_SIZE, 11)
        lambda_y = jnp.zeros((self.BATCH_SIZE, 11))  # shape (BATCH_SIZE, 11)

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

        def _lax_project_trajectory(carry: LaxCarry, iteration: int) -> Tuple[LaxCarry, LaxOutput]:
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

            return LaxCarry(
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
            ), LaxOutput(
                ego_position_trajectory_x=next_ego_position_trajectory_x,
                ego_position_trajectory_y=next_ego_position_trajectory_y,
                ego_velocity_trajectory_x=next_ego_velocity_trajectory_x,
                ego_velocity_trajectory_y=next_ego_velocity_trajectory_y,
                ego_acceleration_trajectory_x=next_ego_acceleration_trajectory_x,
                ego_acceleration_trajectory_y=next_ego_acceleration_trajectory_y,
                error_residuals=next_error_residuals,
            )

        initial_carry = LaxCarry(
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

        output_carry, output = lax.scan(
            _lax_project_trajectory,
            initial_carry,
            jnp.arange(self.MAX_PROJECTION_ITERATIONS),
        )

        return (
            output_carry.primal_solution_x,  # shape (BATCH_SIZE, 11)
            output_carry.primal_solution_y,  # shape (BATCH_SIZE, 11)
            output.ego_position_trajectory_x[-1],  # shape (BATCH_SIZE, num_timesteps)
            output.ego_position_trajectory_y[-1],  # shape (BATCH_SIZE, num_timesteps)
            output.ego_velocity_trajectory_x[-1],  # shape (BATCH_SIZE, num_timesteps)
            output.ego_velocity_trajectory_y[-1],  # shape (BATCH_SIZE, num_timesteps)
            output.ego_acceleration_trajectory_x[-1],  # shape (BATCH_SIZE, num_timesteps)
            output.ego_acceleration_trajectory_y[-1],  # shape (BATCH_SIZE, num_timesteps)
            output.error_residuals,  # shape (MAX_PROJECTION_ITERATIONS, BATCH_SIZE,)
        )


if __name__ == "__main__":
    # Testing
    guidance = ProjectionGuidance(
        num_obstacles=2,
        num_timesteps=1000,
        total_time=10.0,
        batch_size=1,
        obstacle_ellipse_semi_major_axis=1.0,
        obstacle_ellipse_semi_minor_axis=0.5,
        max_projection_iterations=13,
    )

    import time

    # Use random variables to test pipeline
    outputs = None
    for _ in range(10):
        start = time.perf_counter()
        outputs = guidance.project_trajectory(
            initial_ego_position_x=0.0,
            initial_ego_position_y=0.0,
            initial_ego_velocity_x=0.0,
            initial_ego_velocity_y=0.0,
            initial_ego_acceleration_x=0.0,
            initial_ego_acceleration_y=0.0,
            final_ego_position_x=10.0,
            final_ego_position_y=10.0,
            obstacle_positions_x=jnp.array([1.0, 2.0]),
            obstacle_positions_y=jnp.array([1.0, 2.0]),
            obstacle_velocities_x=jnp.array([0.0, 0.0]),
            obstacle_velocities_y=jnp.array([0.0, 0.0]),
            maximum_velocity=1.0,
            maximum_acceleration=1.0,
            warm_ego_position_trajectory_x=jnp.zeros((1, 1000)),
            warm_ego_position_trajectory_y=jnp.zeros((1, 1000)),
        )
        print(time.perf_counter() - start)

    # print(outputs[2], outputs[2].shape)
