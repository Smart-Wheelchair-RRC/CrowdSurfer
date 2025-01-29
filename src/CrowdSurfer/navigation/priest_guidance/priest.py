import jax.numpy as jnp
import numpy as np
import open3d
from jax import random

from .mpc_non_dy import batch_crowd_nav


class PriestPlanner:
    def __init__(
        self,
        num_dynamic_obstacles=10,
        num_obstacles=100,
        t_fin=5,
        num=50,
        max_velocity=1.0,
        min_velocity=0.0,
        max_acceleration=1.0,
        max_iter=1,
        max_iter_cem=12,
        weight_track=0.1,
        weight_smoothness=0.2,
        a_obs_1=0.5,
        b_obs_1=0.5,
        a_obs_2=0.68,
        b_obs_2=0.68,
        num_batch=110,
        v_des=1.0,
        num_waypoints=1000,
    ):
        # Initial and final conditions
        self.x_init, self.y_init = 0, 0
        self.vx_init, self.vy_init = 0.0, 0.0
        self.ax_init, self.ay_init = 0.0, 0.0
        self.x_fin, self.y_fin = 0.0, 0.0  # Might have to change these later
        self.theta_init = 0.0

        # MPC parameters
        self.num_dynamic_obstacles = num_dynamic_obstacles
        self.num_obstacles = num_obstacles
        self.v_max, self.v_min = max_velocity, min_velocity
        self.a_max = max_acceleration
        self.maxiter, self.maxiter_cem = max_iter, max_iter_cem
        self.weight_track, self.weight_smoothness = weight_track, weight_smoothness
        self.a_obs_1, self.b_obs_1 = a_obs_1, b_obs_1
        self.a_obs_2, self.b_obs_2 = a_obs_2, b_obs_2
        self.t_fin, self.num, self.num_batch = t_fin, num, num_batch
        self.v_des = v_des
        self.maxiter_mpc = 300
        self.num_waypoints = num_waypoints

        # Initialize obstacle data
        self.x_obs_init_dy = np.ones(self.num_dynamic_obstacles) * 100
        self.y_obs_init_dy = np.ones(self.num_dynamic_obstacles) * 100
        self.vx_obs_dy = np.zeros(self.num_dynamic_obstacles)
        self.vy_obs_dy = np.zeros(self.num_dynamic_obstacles)

        self.x_obs_init = np.ones(self.num_obstacles) * 100
        self.y_obs_init = np.ones(self.num_obstacles) * 100
        self.vx_obs = jnp.zeros(self.num_obstacles)
        self.vy_obs = jnp.zeros(self.num_obstacles)

        self.pcd = open3d.geometry.PointCloud()

        self.prob = batch_crowd_nav(
            self.a_obs_1,
            self.b_obs_1,
            self.a_obs_2,
            self.b_obs_2,
            self.v_max,
            self.v_min,
            self.a_max,
            self.num_obstacles,
            self.num_dynamic_obstacles,
            self.t_fin,
            self.num,
            self.num_batch,
            self.maxiter,
            self.maxiter_cem,
            self.weight_smoothness,
            self.weight_track,
            self.num_waypoints,
            self.v_des,
        )

        self.key = random.PRNGKey(0)

    def update_obstacle_pointcloud(self, downpcd_array):
        num_down_samples = downpcd_array.shape[0]

        self.x_obs_init[:num_down_samples] = downpcd_array[:, 0]
        self.y_obs_init[:num_down_samples] = downpcd_array[:, 1]
        self.x_obs_init[num_down_samples:] = 100
        self.y_obs_init[num_down_samples:] = 100

    def run_optimization(
        self,
        custom_x_waypoint=None,
        custom_y_waypoint=None,
        custom_x_coefficient=None,
        custom_y_coefficient=None,
    ):
        initial_state = jnp.array(
            [
                self.x_init,
                self.y_init,
                self.vx_init,
                self.vy_init,
                self.ax_init,
                self.ay_init,
            ]
        )
        x_obs_init_dy = self.x_obs_init_dy.copy()
        y_obs_init_dy = self.y_obs_init_dy.copy()
        x_obs_init = self.x_obs_init.copy()
        y_obs_init = self.y_obs_init.copy()
        vx_obs_dy = self.vx_obs_dy.copy()
        vy_obs_dy = self.vy_obs_dy.copy()
        vx_obs = self.vx_obs.copy()
        vy_obs = self.vy_obs.copy()

        theta_des = np.arctan2(self.y_fin - self.y_init, self.x_fin - self.x_init)

        if custom_x_waypoint is not None and custom_y_waypoint is not None:
            x_waypoint = custom_x_waypoint
            y_waypoint = custom_y_waypoint

            # Fix waypoints if shorter than num_waypoints
            arc_length, _, _, _ = self.prob.path_spline(x_waypoint, y_waypoint)

            if arc_length < self.t_fin * self.v_des:
                # Extrapolate waypoints
                num_waypoints_before_threshold = int(self.num_waypoints * arc_length / (self.t_fin * self.v_des))
                angle_index = int((x_waypoint.shape[0] - 1) * 0.75)
                final_theta = np.arctan2(
                    y_waypoint[-1] - y_waypoint[angle_index], x_waypoint[-1] - x_waypoint[angle_index]
                )
                x_waypoint = np.concatenate(
                    [
                        x_waypoint,
                        np.linspace(
                            x_waypoint[-1],
                            x_waypoint[-1] + ((self.t_fin * self.v_des) - arc_length) * np.cos(final_theta),
                            self.num_waypoints - num_waypoints_before_threshold,
                        ),
                    ]
                )
                y_waypoint = np.concatenate(
                    [
                        y_waypoint,
                        np.linspace(
                            y_waypoint[-1],
                            y_waypoint[-1] + ((self.t_fin * self.v_des) - arc_length) * np.sin(final_theta),
                            self.num_waypoints - num_waypoints_before_threshold,
                        ),
                    ]
                )
        else:
            x_waypoint = jnp.linspace(
                self.x_init,
                self.x_init + (self.t_fin * self.v_des) * jnp.cos(theta_des),
                self.num_waypoints,
            )
            y_waypoint = jnp.linspace(
                self.y_init,
                self.y_init + (self.t_fin * self.v_des) * jnp.sin(theta_des),
                self.num_waypoints,
            )

        arc_length, arc_vec, x_diff, y_diff = self.prob.path_spline(x_waypoint, y_waypoint)

        if custom_x_coefficient is None or custom_y_coefficient is None:
            x_guess_per, y_guess_per = self.prob.compute_warm_traj(
                initial_state,
                self.v_des,
                x_waypoint,
                y_waypoint,
                arc_vec,
                x_diff,
                y_diff,
            )

        (
            x_obs_trajectory,
            y_obs_trajectory,
            x_obs_trajectory_proj,
            y_obs_trajectory_proj,
            x_obs_trajectory_dy,
            y_obs_trajectory_dy,
        ) = self.prob.compute_obs_traj_prediction(
            jnp.asarray(x_obs_init_dy).flatten(),
            jnp.asarray(y_obs_init_dy).flatten(),
            vx_obs_dy,
            vy_obs_dy,
            jnp.asarray(x_obs_init).flatten(),
            jnp.asarray(y_obs_init).flatten(),
            vx_obs,
            vy_obs,
            initial_state[0],
            initial_state[1],
        )

        if custom_x_coefficient is not None and custom_y_coefficient is not None:
            self.prob.ellite_num_const = custom_x_coefficient.shape[0]

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
        ) = (
            self.prob.compute_traj_guess(
                initial_state,
                x_obs_trajectory,
                y_obs_trajectory,
                x_obs_trajectory_dy,
                y_obs_trajectory_dy,
                self.v_des,
                x_waypoint,
                y_waypoint,
                arc_vec,
                x_guess_per,
                y_guess_per,
                x_diff,
                y_diff,
            )
            if (custom_x_coefficient is None or custom_y_coefficient is None)
            else (
                self.prob.compute_traj_guess_from_waypoints(
                    x_coefficients=custom_x_coefficient,
                    y_coefficients=custom_y_coefficient,
                    x_waypoint=x_waypoint,
                    y_waypoint=y_waypoint,
                )
            )
        )

        lamda_x = jnp.zeros((self.num_batch, self.prob.nvar))
        lamda_y = jnp.zeros((self.num_batch, self.prob.nvar))

        x_elite, y_elite, c_x_elite, c_y_elite, idx_min = self.prob.compute_cem(
            self.key,
            initial_state,
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
            x_waypoint,
            y_waypoint,
            arc_vec,
            c_mean,
            c_cov,
        )

        c_x_best = c_x_elite[idx_min]
        c_y_best = c_y_elite[idx_min]
        x_best = x_elite[idx_min]
        y_best = y_elite[idx_min]

        return c_x_best, c_y_best, x_best, y_best, c_x_elite, c_y_elite, x_elite, y_elite, idx_min
