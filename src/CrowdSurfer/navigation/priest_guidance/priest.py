from dataclasses import dataclass
from typing import Optional, Union

import jax.numpy as jnp
import numpy as np
from jax import random

from .priest_core import PriestPlannerCore


@dataclass
class OptimizationPacket:
    initial_state: jnp.ndarray
    dynamic_obstacle_x_positions: Union[np.ndarray, jnp.ndarray]
    dynamic_obstacle_y_positions: Union[np.ndarray, jnp.ndarray]
    dynamic_obstacle_x_velocities: Union[np.ndarray, jnp.ndarray]
    dynamic_obstacle_y_velocities: Union[np.ndarray, jnp.ndarray]
    static_obstacle_x_positions: Union[np.ndarray, jnp.ndarray]
    static_obstacle_y_positions: Union[np.ndarray, jnp.ndarray]
    x_waypoint: Union[np.ndarray, jnp.ndarray]
    y_waypoint: Union[np.ndarray, jnp.ndarray]
    arc_vec: Union[np.ndarray, jnp.ndarray]
    x_diff: Union[np.ndarray, jnp.ndarray]
    y_diff: Union[np.ndarray, jnp.ndarray]
    custom_x_coefficients: Optional[Union[np.ndarray, jnp.ndarray]] = None
    custom_y_coefficients: Optional[Union[np.ndarray, jnp.ndarray]] = None

    def __post_init__(self):
        self.dynamic_obstacle_x_positions = self.dynamic_obstacle_x_positions.copy()
        self.dynamic_obstacle_y_positions = self.dynamic_obstacle_y_positions.copy()
        self.dynamic_obstacle_x_velocities = self.dynamic_obstacle_x_velocities.copy()
        self.dynamic_obstacle_y_velocities = self.dynamic_obstacle_y_velocities.copy()
        self.static_obstacle_x_positions = self.static_obstacle_x_positions.copy()
        self.static_obstacle_y_positions = self.static_obstacle_y_positions.copy()
        self.x_waypoint = self.x_waypoint.copy()
        self.y_waypoint = self.y_waypoint.copy()
        self.arc_vec = self.arc_vec.copy()
        self.x_diff = self.x_diff.copy()
        self.y_diff = self.y_diff.copy()
        if self.custom_x_coefficients is not None:
            self.custom_x_coefficients = self.custom_x_coefficients.copy()
        if self.custom_y_coefficients is not None:
            self.custom_y_coefficients = self.custom_y_coefficients.copy()


class PriestPlanner:
    def __init__(
        self,
        num_dynamic_obstacles=10,
        num_static_obstacles=100,
        time_horizon=5,
        trajectory_length=50,
        max_velocity=1.0,
        min_velocity=0.0,
        max_acceleration=1.0,
        max_inner_iterations=13,
        max_outer_iterations=12,
        tracking_weight=0.1,
        smoothness_weight=0.2,
        static_obstacle_semi_minor_axis=0.5,
        static_obstacle_semi_major_axis=0.5,
        dynamic_obstacle_semi_minor_axis=0.68,
        dynamic_obstacle_semi_major_axis=0.68,
        trajectory_batch_size=110,
        desired_velocity=1.0,
        num_waypoints=1000,
    ):
        self.num_dynamic_obstacles = num_dynamic_obstacles
        self.num_static_obstacles = num_static_obstacles
        self.time_horizon, self.trajectory_length, self.trajectory_batch_size = (
            time_horizon,
            trajectory_length,
            trajectory_batch_size,
        )
        self.desired_velocity = desired_velocity
        self.num_waypoints = num_waypoints

        self.priest = PriestPlannerCore(
            static_obstacle_semi_minor_axis,
            static_obstacle_semi_major_axis,
            dynamic_obstacle_semi_minor_axis,
            dynamic_obstacle_semi_major_axis,
            max_velocity,
            min_velocity,
            max_acceleration,
            self.num_static_obstacles,
            self.num_dynamic_obstacles,
            self.time_horizon,
            self.trajectory_length,
            self.trajectory_batch_size,
            max_inner_iterations,
            max_outer_iterations,
            smoothness_weight,
            tracking_weight,
            self.num_waypoints,
            self.desired_velocity,
        )

        self.key = random.PRNGKey(0)
        self.x_waypoint, self.y_waypoint = None, None

    @staticmethod
    def _add_padding(
        array: Union[np.ndarray, jnp.ndarray], max_size: int, padding_value: float
    ) -> np.ndarray:
        output = np.ones(max_size) * padding_value
        output[: array.shape[0]] = array[:max_size]
        return output

    def optimize(
        self,
        optimization_packet: OptimizationPacket,
    ):
        (
            x_obs_trajectory,
            y_obs_trajectory,
            x_obs_trajectory_proj,
            y_obs_trajectory_proj,
            x_obs_trajectory_dy,
            y_obs_trajectory_dy,
        ) = self.priest.compute_obs_traj_prediction(
            self._add_padding(
                optimization_packet.dynamic_obstacle_x_positions,
                self.num_dynamic_obstacles,
                1000,
            ),
            self._add_padding(
                optimization_packet.dynamic_obstacle_y_positions,
                self.num_dynamic_obstacles,
                1000,
            ),
            self._add_padding(
                optimization_packet.dynamic_obstacle_x_velocities,
                self.num_dynamic_obstacles,
                0,
            ),
            self._add_padding(
                optimization_packet.dynamic_obstacle_y_velocities,
                self.num_dynamic_obstacles,
                0,
            ),
            self._add_padding(
                optimization_packet.static_obstacle_x_positions,
                self.num_static_obstacles,
                1000,
            ),
            self._add_padding(
                optimization_packet.static_obstacle_y_positions,
                self.num_static_obstacles,
                1000,
            ),
            jnp.zeros(self.num_static_obstacles),
            jnp.zeros(self.num_static_obstacles),
            optimization_packet.initial_state[0],
            optimization_packet.initial_state[1],
        )

        if (
            optimization_packet.custom_x_coefficients is not None
            and optimization_packet.custom_y_coefficients is not None
        ):
            self.priest.ellite_num_const = (
                optimization_packet.custom_x_coefficients.shape[0]
            )
            # self.priest.initial_up_sampling = 1

            traj_guess = self.priest.compute_traj_guess_from_coefficients(
                # optimization_packet.initial_state,
                # self.desired_velocity,
                optimization_packet.x_waypoint,
                optimization_packet.y_waypoint,
                optimization_packet.custom_x_coefficients,
                optimization_packet.custom_y_coefficients,
                # optimization_packet.arc_vec,
            )
        else:
            x_guess_per, y_guess_per = self.priest.compute_warm_traj(
                optimization_packet.initial_state,
                self.desired_velocity,
                self.x_waypoint,
                self.y_waypoint,
                self.arc_vec,
                self.x_diff,
                self.y_diff,
            )
            traj_guess = self.priest.compute_traj_guess(
                optimization_packet.initial_state,
                x_obs_trajectory,
                y_obs_trajectory,
                x_obs_trajectory_dy,
                y_obs_trajectory_dy,
                self.desired_velocity,
                optimization_packet.x_waypoint,
                optimization_packet.y_waypoint,
                optimization_packet.arc_vec,
                x_guess_per,
                y_guess_per,
                optimization_packet.x_diff,
                optimization_packet.y_diff,
            )

        (
            sol_x_bar,
            sol_y_bar,
            x_guess,
            y_guess,
            xdot_guess,
            ydot_guess,
            xddot_guess,
            yddot_guess,
            c_mean,
            c_cov,
            x_fin,
            y_fin,
        ) = traj_guess

        lamda_x = jnp.zeros((self.trajectory_batch_size, self.priest.nvar))
        lamda_y = jnp.zeros((self.trajectory_batch_size, self.priest.nvar))

        x_elite, y_elite, c_x_elite, c_y_elite, idx_min = self.priest.compute_cem(
            self.key,
            optimization_packet.initial_state,
            x_fin,
            y_fin,
            lamda_x,
            lamda_y,
            x_obs_trajectory,
            y_obs_trajectory,
            x_obs_trajectory_proj,
            y_obs_trajectory_proj,
            x_obs_trajectory_dy,
            y_obs_trajectory_dy,
            sol_x_bar,
            sol_y_bar,
            x_guess,
            y_guess,
            xdot_guess,
            ydot_guess,
            xddot_guess,
            yddot_guess,
            optimization_packet.x_waypoint,
            optimization_packet.y_waypoint,
            optimization_packet.arc_vec,
            c_mean,
            c_cov,
        )

        c_x_best = c_x_elite[idx_min]
        c_y_best = c_y_elite[idx_min]
        x_best = x_elite[idx_min]
        y_best = y_elite[idx_min]

        return (
            c_x_best,
            c_y_best,
            x_best,
            y_best,
            c_x_elite,
            c_y_elite,
            x_elite,
            y_elite,
            idx_min,
        )

    def update_waypoints(
        self,
        initial_x_position: float,
        initial_y_position: float,
        goal_x_position: float,
        goal_y_position: float,
        custom_x_waypoint: Optional[Union[np.ndarray, jnp.ndarray]] = None,
        custom_y_waypoint: Optional[Union[np.ndarray, jnp.ndarray]] = None,
        use_eager_waypoints: bool = True,
    ):
        if custom_x_waypoint is not None and custom_y_waypoint is not None:
            # extrapolation_length = 1
            extrapolation_length = self.time_horizon * self.desired_velocity
            extrapolation_steps = 20

            # Extrapolate waypoints when close to goal in the case of eager waypoints
            if use_eager_waypoints:
                current_arc_length, _, _, _ = self.priest.path_spline(
                    custom_x_waypoint, custom_y_waypoint
                )
                num_waypoints_before_threshold = self.num_waypoints
                if current_arc_length < self.time_horizon * self.desired_velocity:
                    num_waypoints_before_threshold = int(
                        self.num_waypoints
                        * current_arc_length
                        / (self.time_horizon * self.desired_velocity)
                    )
                    extrapolation_length -= current_arc_length

                extrapolation_steps = (
                    self.num_waypoints - num_waypoints_before_threshold
                )

            # angle_index = int((custom_x_waypoint.shape[0] - 1) * 0.9)
            angle_index = -2
            extrapolation_angle = jnp.arctan2(
                custom_y_waypoint[-1] - custom_y_waypoint[angle_index],
                custom_x_waypoint[-1] - custom_x_waypoint[angle_index],
            )

            custom_x_waypoint = jnp.concatenate(
                [
                    custom_x_waypoint[:-1],
                    jnp.linspace(
                        custom_x_waypoint[-1],
                        custom_x_waypoint[-1]
                        + extrapolation_length * np.cos(extrapolation_angle),
                        extrapolation_steps,
                    ),
                ]
            )
            custom_y_waypoint = jnp.concatenate(
                [
                    custom_y_waypoint[:-1],
                    jnp.linspace(
                        custom_y_waypoint[-1],
                        custom_y_waypoint[-1]
                        + extrapolation_length * np.sin(extrapolation_angle),
                        extrapolation_steps,
                    ),
                ]
            )

            self.num_waypoints = custom_x_waypoint.shape[0]
            self.x_waypoint = custom_x_waypoint
            self.y_waypoint = custom_y_waypoint
        else:
            theta_des = np.arctan2(
                goal_y_position - initial_y_position,
                goal_x_position - initial_x_position,
            )
            self.x_waypoint = jnp.linspace(
                initial_x_position,
                (initial_x_position if use_eager_waypoints else goal_x_position)
                + (self.time_horizon * self.desired_velocity) * jnp.cos(theta_des),
                self.num_waypoints if use_eager_waypoints else 1000,
            )
            self.y_waypoint = jnp.linspace(
                initial_y_position,
                (initial_y_position if use_eager_waypoints else goal_y_position)
                + (self.time_horizon * self.desired_velocity) * jnp.sin(theta_des),
                self.num_waypoints if use_eager_waypoints else 1000,
            )

        self.arc_length, self.arc_vec, self.x_diff, self.y_diff = (
            self.priest.path_spline(self.x_waypoint, self.y_waypoint)
        )

    def run_optimization(
        self,
        initial_x_position: float,
        initial_y_position: float,
        initial_x_velocity: float,
        initial_y_velocity: float,
        initial_x_acceleration: float,
        initial_y_acceleration: float,
        goal_x_position: float,
        goal_y_position: float,
        dynamic_obstacle_x_positions: Optional[np.ndarray],
        dynamic_obstacle_y_positions: Optional[np.ndarray],
        dynamic_obstacle_x_velocities: Optional[np.ndarray],
        dynamic_obstacle_y_velocities: Optional[np.ndarray],
        static_obstacle_x_positions: Optional[np.ndarray],
        static_obstacle_y_positions: Optional[np.ndarray],
        custom_x_waypoint: Optional[np.ndarray] = None,
        custom_y_waypoint: Optional[np.ndarray] = None,
        custom_x_coefficients: Optional[np.ndarray] = None,
        custom_y_coefficients: Optional[np.ndarray] = None,
        update_waypoints: bool = True,
        use_eager_waypoints: bool = True,
    ):
        if dynamic_obstacle_x_positions is None:
            dynamic_obstacle_x_positions = np.ones(self.num_dynamic_obstacles) * 1000
        if dynamic_obstacle_y_positions is None:
            dynamic_obstacle_y_positions = np.ones(self.num_dynamic_obstacles) * 1000
        if dynamic_obstacle_x_velocities is None:
            dynamic_obstacle_x_velocities = np.zeros(self.num_dynamic_obstacles)
        if dynamic_obstacle_y_velocities is None:
            dynamic_obstacle_y_velocities = np.zeros(self.num_dynamic_obstacles)
        if static_obstacle_x_positions is None:
            static_obstacle_x_positions = np.ones(self.num_static_obstacles) * 1000
        if static_obstacle_y_positions is None:
            static_obstacle_y_positions = np.ones(self.num_static_obstacles) * 1000

        initial_state = jnp.array(
            [
                initial_x_position,
                initial_y_position,
                initial_x_velocity,
                initial_y_velocity,
                initial_x_acceleration,
                initial_y_acceleration,
            ]
        )

        if update_waypoints:
            self.update_waypoints(
                initial_x_position=initial_x_position,
                initial_y_position=initial_y_position,
                goal_x_position=goal_x_position,
                goal_y_position=goal_y_position,
                custom_x_waypoint=custom_x_waypoint,
                custom_y_waypoint=custom_y_waypoint,
                use_eager_waypoints=use_eager_waypoints,
            )

        assert self.x_waypoint is not None and self.y_waypoint is not None

        return self.optimize(
            OptimizationPacket(
                initial_state=initial_state,
                dynamic_obstacle_x_positions=dynamic_obstacle_x_positions,
                dynamic_obstacle_y_positions=dynamic_obstacle_y_positions,
                dynamic_obstacle_x_velocities=dynamic_obstacle_x_velocities,
                dynamic_obstacle_y_velocities=dynamic_obstacle_y_velocities,
                static_obstacle_x_positions=static_obstacle_x_positions,
                static_obstacle_y_positions=static_obstacle_y_positions,
                x_waypoint=self.x_waypoint,
                y_waypoint=self.y_waypoint,
                arc_vec=self.arc_vec,
                x_diff=self.x_diff,
                y_diff=self.y_diff,
                custom_x_coefficients=custom_x_coefficients,
                custom_y_coefficients=custom_y_coefficients,
            )
        )
