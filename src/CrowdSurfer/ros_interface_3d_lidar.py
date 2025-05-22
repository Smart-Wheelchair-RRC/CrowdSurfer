#!/usr/bin/env python3

import os
import time

# import message_filters
import numpy as np
import rospy

# from tf.transformations import euler_from_quaternion
import tf
from configuration import (
    Configuration,
    DynamicObstaclesMessageType,
    Mode,
    check_configuration,
    initialize_configuration,
)
from geometry_msgs.msg import PoseStamped, Twist
from inference import LivePipeline
from nav_msgs.msg import Path
from pedsim_msgs.msg import AgentStates, TrackedPersons  # type: ignore
from sensor_msgs.msg import PointCloud2

# from sensor_msgs.msg import LaserScan, PointCloud
from std_msgs.msg import Empty, Header
from tf.transformations import quaternion_matrix
from visualization_msgs.msg import MarkerArray

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from dataclasses import dataclass
from typing import Optional, Union

# from processing import per_timestep_processing_functions as processing_functions
from std_srvs.srv import Empty as EmptyService

np.float = np.float64

from ros_numpy.point_cloud2 import pointcloud2_to_xyz_array

pause_physics_service = None
unpause_physics_service = None


@dataclass
class PlanningData:
    """
    Dataclass to store the data required for planning
    All values are in ego frame, except odometry which is in world frame
    Dynamic obstacles include the past 5 timesteps of dynamic obstacles
    """

    goal: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    point_cloud: np.ndarray
    # laser_scan: np.ndarray
    dynamic_obstacles: np.ndarray
    update_waypoints: bool = False  # True
    sub_goal: Optional[np.ndarray] = None
    # global_path: Optional[np.ndarray] = None


class ROSInterface:
    def __init__(self, configuration: Configuration):
        self.pipeline = LivePipeline(configuration)
        self.total_rollout_time = 0
        self.num_rollouts = 0
        self.time_horizon = configuration.dataset.trajectory_time
        self.threshold_distance = configuration.live.threshold_distance
        self.obstacle_padding = configuration.projection.padding
        self.use_global_path = configuration.live.use_global_path
        self.max_static_obstacles = configuration.projection.max_static_obstacles
        self.max_angular_velocity = configuration.live.max_angular_velocity
        self.max_velocity = configuration.projection.max_velocity

        self.BERNSTEIN_POLYNOMIALS_FIRST_DIFFERENTIAL = (
            self.pipeline.projection_guidance.BERNSTEIN_POLYNOMIALS_FIRST_DIFFERENTIAL.detach()
            .cpu()
            .numpy()
        )
        self.BERNSTEIN_POLYNOMIALS_SECOND_DIFFERENTIAL = (
            self.pipeline.projection_guidance.BERNSTEIN_POLYNOMIALS_SECOND_DIFFERENTIAL.detach()
            .cpu()
            .numpy()
        )

        self.world_frame = configuration.live.world_frame
        self.robot_base_frame = configuration.live.robot_base_frame

        self.num_control_samples = 6

        self.planning_data = PlanningData(
            goal=np.zeros((2), dtype=np.float32),
            velocity=np.zeros((2), dtype=np.float32),
            acceleration=np.zeros((2), dtype=np.float32),
            # laser_scan=np.full(
            #     (2, 1080), configuration.projection.padding, dtype=np.float32
            # ),
            point_cloud=np.full(
                (2, 1000), configuration.projection.padding, dtype=np.float32
            ),
            dynamic_obstacles=np.concatenate(
                (
                    np.full(
                        (5, 2, self.pipeline.max_projection_dynamic_obstacles),
                        configuration.projection.padding,
                        dtype=np.float32,
                    ),
                    np.zeros(
                        (5, 2, self.pipeline.max_projection_dynamic_obstacles),
                        dtype=np.float32,
                    ),
                ),
                axis=1,
            ),
        )
        self.goal = np.zeros((2), dtype=np.float32)

        # Setup Subscribers
        # self.transform_listener = tf.TransformListener()
        # self.laser_scan_subscriber = message_filters.Subscriber(
        #     configuration.live.laser_scan_topic, LaserScan
        # )

        self.listener = tf.TransformListener()

        self.point_cloud_subscriber = rospy.Subscriber(
            configuration.live.point_cloud_topic, PointCloud2, self.point_cloud_callback
        )

        # time_synchronizer = message_filters.ApproximateTimeSynchronizer(
        #     [self.point_cloud_subscriber, self.laser_scan_subscriber],
        #     queue_size=20,
        #     slop=1,
        #     allow_headerless=True,
        # )
        # time_synchronizer.registerCallback(self.point_cloud_and_laser_scan_callback)

        # self.point_cloud_subscriber = rospy.Subscriber(
        #     configuration.live.point_cloud_topic, PointCloud, self.point_cloud_callback
        # )

        if (
            configuration.live.dynamic_obstacle_message_type
            is DynamicObstaclesMessageType.MARKER_ARRAY
        ):
            self.dynamic_obstacle_subscriber = rospy.Subscriber(
                configuration.live.dynamic_obstacle_topic,
                MarkerArray,
                self.dynamic_obstacle_callback,
            )
        elif (
            configuration.live.dynamic_obstacle_message_type
            is DynamicObstaclesMessageType.AGENT_STATES
        ):
            self.dynamic_obstacle_subscriber = rospy.Subscriber(
                configuration.live.dynamic_obstacle_topic,
                AgentStates,
                self.dynamic_obstacle_callback,
            )
        elif (
            configuration.live.dynamic_obstacle_message_type
            is DynamicObstaclesMessageType.TRACKED_PERSONS
        ):
            self.sub_dynamic = rospy.Subscriber(
                configuration.live.dynamic_obstacle_topic,
                TrackedPersons,
                self.dynamic_obstacle_callback,
            )
        else:
            raise NotImplementedError(
                f"For dynamic obstacles, support for message type {configuration.live.dynamic_obstacle_message_type.name} hasn't been implemented yet"
            )

        self.goal_subscriber = rospy.Subscriber(
            configuration.live.goal_topic,
            PoseStamped,
            self.goal_callback,
            queue_size=10,
        )

        if self.use_global_path:
            # self.global_path_subscriber = rospy.Subscriber(
            #     configuration.live.global_path_topic, Path, self.global_path_callback
            # )
            self.sub_goal_subscriber = rospy.Subscriber(
                configuration.live.sub_goal_topic, PoseStamped, self.sub_goal_callback
            )

        # Setup Publishers
        self.velocity_publisher = rospy.Publisher(
            configuration.live.velocity_command_topic, Twist, queue_size=10
        )
        self.path_publisher = rospy.Publisher(
            configuration.live.path_topic, Path, queue_size=10
        )
        self.goal_reached_publisher = rospy.Publisher(
            "/reached_goal", Empty, queue_size=10
        )

        self.received_goal = False

    def _reached_goal(self, threshold_distance: float):
        if np.linalg.norm(self.planning_data.goal) <= threshold_distance:
            self.received_goal = False
            self.goal_reached_publisher.publish()
            if self.num_rollouts != 0:
                print("Reached Goal")
                print(
                    "Average Rollout Time: ",
                    self.total_rollout_time / self.num_rollouts,
                )
            self.total_rollout_time = 0
            self.num_rollouts = 0
            return True
        return False

    def sub_goal_callback(self, sub_goal: PoseStamped):
        self.planning_data.sub_goal = np.array(
            [sub_goal.pose.position.x, sub_goal.pose.position.y]
        )

    # def global_path_callback(self, global_path: Path):
    #     print("Received plan")
    #     waypoints = global_path.poses if global_path.poses else []
    #     waypoints = [np.array([waypoint.pose.position.x, waypoint.pose.position.y]) for waypoint in waypoints]
    #     waypoints = np.array(waypoints).T
    #     self.planning_data.global_path = waypoints

    def goal_callback(self, goal: PoseStamped):
        self.goal = np.array([goal.pose.position.x, goal.pose.position.y])
        self.planning_data.goal = np.array([goal.pose.position.x, goal.pose.position.y])
        self.received_goal = True
        # print("RECEIVED GOAL")

    # def _process_laser_scan(self, laser_scan_message: LaserScan):
    #     max_useful_range = 50
    #     maximum_points = 1080
    #     ranges = np.array(laser_scan_message.ranges)

    #     # Check for zero or very small angle_increment
    #     if abs(laser_scan_message.angle_increment) < 1e-10:
    #         num_points = len(ranges)
    #         angles = np.linspace(
    #             laser_scan_message.angle_min, laser_scan_message.angle_max, num_points
    #         )
    #     else:
    #         angles = np.arange(
    #             laser_scan_message.angle_min,
    #             laser_scan_message.angle_max + laser_scan_message.angle_increment,
    #             laser_scan_message.angle_increment,
    #         )

    #     # Ensure angles and ranges have the same length. If not, set as same.
    #     if len(angles) != len(ranges):
    #         min_len = min(len(angles), len(ranges))
    #         angles = angles[:min_len]
    #         ranges = ranges[:min_len]

    #     valid_ranges = (
    #         (ranges >= laser_scan_message.range_min)
    #         & (
    #             (ranges <= laser_scan_message.range_max)
    #             | (np.isinf(laser_scan_message.range_max))
    #         )
    #         & (ranges <= max_useful_range)
    #     )
    #     valid_ranges_array = ranges[valid_ranges]
    #     valid_angles_array = angles[valid_ranges]

    #     # Pad remaining positions for consistent array length
    #     padded_ranges = np.pad(
    #         valid_ranges_array,
    #         (0, maximum_points - len(valid_ranges_array)),
    #         mode="constant",
    #         constant_values=np.nan,
    #     )[:maximum_points]
    #     padded_angles = np.pad(
    #         valid_angles_array,
    #         (0, maximum_points - len(valid_angles_array)),
    #         mode="constant",
    #         constant_values=np.nan,
    #     )[:maximum_points]

    #     return np.stack((padded_ranges, padded_angles), axis=-1)

    # def point_cloud_and_laser_scan_callback(
    #     self,
    #     point_cloud_message: PointCloud,
    #     laser_scan_message: LaserScan,
    # ):
    #     point_cloud_array = []

    #     points = point_cloud_message.points if point_cloud_message.points else []
    #     for point in points:
    #         point_cloud_array.append([point.x, point.y, 0])

    #     self.planning_data.point_cloud = np.array(point_cloud_array)

    #     self.planning_data.laser_scan = self._process_laser_scan(laser_scan_message)

    def point_cloud_callback(self, point_cloud_message: PointCloud2):
        self.planning_data.point_cloud = pointcloud2_to_xyz_array(point_cloud_message)

    def _process_dynamic_obstacles_from_marker_array(
        self, dynamic_obstacles: MarkerArray
    ):
        if len(dynamic_obstacles.markers) > 0:
            pose_array = []
            vel_array = []

            for obs in dynamic_obstacles.markers:
                if obs.ns == "dynamic_obstacle_velocity_text":
                    pose_array.append([obs.pose.position.x, obs.pose.position.y, 1])

                    parts = obs.text.split(",")
                    vx_str = parts[0].split("Vx=")[1].strip()
                    vx = float(vx_str)
                    vy_str = parts[1].split("Vy=")[1].strip()
                    vy = float(vy_str)

                    vel_array.append([vx, vy])

            # apply transform
            frame = dynamic_obstacles.markers[0].header.frame_id
            try:
                (trans, rot) = self.listener.lookupTransform(
                    frame, self.robot_base_frame, rospy.Time(0)
                )
            except (
                tf.LookupException,
                tf.ConnectivityException,
                tf.ExtrapolationException,
            ):
                print(
                    f"Lookup failed between {frame} and {self.robot_base_frame} for dynamic obstacle velocity transformation"
                )
                return []

            rot = quaternion_matrix(np.array(rot))
            rot = rot[:3, :3].T

            T = np.eye(3)
            T[:2, :2] = rot[:2, :2]
            T[:2, 2] = trans[:2]

            pose_array = np.array(pose_array)
            vel_array = np.array(vel_array)

            pose_array = (T @ pose_array.T).T
            pose_array = pose_array[:, :2]
            vel_array = (rot[:2, :2] @ vel_array.T).T

            obs_array = np.hstack((pose_array, vel_array))

            # print(obs_array)
            return obs_array

        return []

    def _process_dynamic_obstacles_from_agent_states(
        self, dynamic_obstacles: AgentStates
    ):
        obstacles_list = []
        agent_states = (
            dynamic_obstacles.agent_states if dynamic_obstacles.agent_states else []
        )
        for agent_state in agent_states:
            obstacles_list.append(
                [
                    agent_state.pose.position.x,
                    agent_state.pose.position.y,
                    agent_state.twist.linear.x,
                    agent_state.twist.linear.y,
                ]
            )
        return obstacles_list

    def _process_dynamic_obstacles_from_tracked_persons(
        self, dynamic_obstacles: TrackedPersons
    ):
        obstacles_list = []
        tracks = dynamic_obstacles.tracks if dynamic_obstacles.tracks else []
        for t in tracks:
            obstacles_list.append(
                [
                    t.pose.pose.position.x,
                    t.pose.pose.position.y,
                    t.twist.twist.linear.x,
                    t.twist.twist.linear.y,
                ]
            )
        return obstacles_list

    def dynamic_obstacle_callback(
        self, dynamic_obstacle_message: Union[MarkerArray, AgentStates]
    ):
        if isinstance(dynamic_obstacle_message, MarkerArray):
            obstacles_list = self._process_dynamic_obstacles_from_marker_array(
                dynamic_obstacle_message
            )
        elif isinstance(dynamic_obstacle_message, AgentStates):
            obstacles_list = self._process_dynamic_obstacles_from_agent_states(
                dynamic_obstacle_message
            )
        elif isinstance(dynamic_obstacle_message, TrackedPersons):
            obstacles_list = self._process_dynamic_obstacles_from_tracked_persons(
                dynamic_obstacle_message
            )
        else:
            raise ValueError(
                f"Unsupported message type {type(dynamic_obstacle_message)}"
            )

        self.planning_data.dynamic_obstacles[:-1] = (
            self.planning_data.dynamic_obstacles[1:]
        )
        self.planning_data.dynamic_obstacles[-1, :2, :] = self.obstacle_padding
        self.planning_data.dynamic_obstacles[-1, 2:, :] = 0

        # get closest dynamic obstacles
        obstacles = np.array(obstacles_list)

        if obstacles.shape[0] > 0:
            obstacles = obstacles[
                np.argsort(np.linalg.norm(obstacles[:, :2], axis=-1), axis=-1)
            ][: self.pipeline.max_projection_dynamic_obstacles]
            obstacles = obstacles.T

            self.planning_data.dynamic_obstacles[-1, :, : obstacles.shape[1]] = (
                obstacles
            )

        # return self.planning_data.dynamic_obstacles

    def _check_dynamic_obstacle_distance_for_stopping(
        self, threshold_distance: float = 1.2, num_obstacles: int = 1
    ):
        current_obstacles = self.planning_data.dynamic_obstacles[-1, :2, :]

        if current_obstacles.shape[1] == 0:
            return False

        # Filter out obstacles with negative values
        valid_obstacles = current_obstacles[:, (current_obstacles >= 0).all(axis=0)]

        if valid_obstacles.shape[1] == 0:
            return False

        dist = np.linalg.norm(valid_obstacles, axis=0)
        number_of_obstacles_within_threshold = np.sum(dist <= threshold_distance)

        if number_of_obstacles_within_threshold >= num_obstacles:
            print("Stopping")
            return True

        return False

    def plan(self):
        if (
            not self._reached_goal(self.threshold_distance)
            and (self.planning_data.sub_goal is not None or not self.use_global_path)
            and self.received_goal
        ):
            # pause_physics_service()
            start_time = time.perf_counter()

            coefficients, trajectories, scores = self.pipeline.run(
                goal=self.planning_data.goal
                if self.planning_data.sub_goal is None
                else self.planning_data.sub_goal,
                ego_velocity=self.planning_data.velocity,
                ego_acceleration=self.planning_data.acceleration,
                laser_scan=None,
                point_cloud=self.planning_data.point_cloud,
                # occupancy_map=processing_functions.generate_occupancy_map_from_point_cloud(
                #     self.planning_data.point_cloud
                # ),
                # downsampled_point_cloud=processing_functions.downsample_point_cloud(
                #     self.planning_data.point_cloud, max_downsampled_points=self.max_static_obstacles, padding_value=None
                # ).T,
                dynamic_obstacles_n_steps=self.planning_data.dynamic_obstacles,
            )

            self.total_rollout_time = time.perf_counter() - start_time
            self.num_rollouts += 1

            best_index = np.argmax(scores)
            best_coeffs = coefficients[best_index]
            x_best, y_best = best_coeffs[:, 0], best_coeffs[:, 1]

            best_traj = trajectories[best_index]
            self.publish_path_message(best_traj)

            vx_control, vy_control, ax_control, ay_control, norm_v_t, angle_v_t = (
                self.compute_controls(x_best * 0.8, y_best * 0.8)
            )

            self.planning_data.velocity = np.array(
                (vx_control, vy_control), dtype=np.float32
            )
            self.planning_data.acceleration = np.array(
                (ax_control, ay_control), dtype=np.float32
            )

            # zeta = self.convert_angle(self.planning_data.odometry[2]) - self.convert_angle(angle_v_t)
            zeta = self.convert_angle(0) - self.convert_angle(angle_v_t)
            v_t_control = norm_v_t * np.cos(zeta)
            v_t_control = min(max(v_t_control, 0), self.max_velocity)
            # v_t_control = max((min(v_t_control, 0.8), -0.8))

            # omega_control = -zeta / (
            #     self.num_control_samples * self.time_horizon * 0.01
            # )
            omega_control = -zeta / (self.num_control_samples * 15 * 0.01)
            # omega_control = max(min(omega_control.item(), 10), -10)

            if np.abs(omega_control) > self.max_angular_velocity:
                omega_control = np.sign(omega_control) * self.max_angular_velocity

            cmd_vel = Twist()

            # if self._check_dynamic_obstacle_distance_for_stopping():
            #     cmd_vel.linear.x = 0
            #     cmd_vel.angular.z = omega_control * 1.5
            # else:
            cmd_vel.linear.x = v_t_control
            cmd_vel.angular.z = omega_control

            print("Control", v_t_control, omega_control)
            self.velocity_publisher.publish(cmd_vel)
        else:
            self.publish_zero_velocity()

    def convert_angle(self, angle):
        angle = np.unwrap(np.array([angle]), discont=np.pi, axis=0, period=2 * np.pi)

        return angle.item()

    def compute_controls(self, x_best, y_best):
        xdot_best = np.dot(
            self.BERNSTEIN_POLYNOMIALS_FIRST_DIFFERENTIAL,
            x_best,
        )
        ydot_best = np.dot(
            self.BERNSTEIN_POLYNOMIALS_FIRST_DIFFERENTIAL,
            y_best,
        )

        xddot_best = np.dot(self.BERNSTEIN_POLYNOMIALS_SECOND_DIFFERENTIAL, x_best)
        yddot_best = np.dot(self.BERNSTEIN_POLYNOMIALS_SECOND_DIFFERENTIAL, y_best)

        vx_control = np.mean(xdot_best[: self.num_control_samples])
        vy_control = np.mean(ydot_best[: self.num_control_samples])

        ax_control = np.mean(xddot_best[: self.num_control_samples])
        ay_control = np.mean(yddot_best[: self.num_control_samples])

        norm_v_t = np.sqrt(vx_control**2 + vy_control**2)
        angle_v_t = np.arctan2(vy_control, vx_control)

        return vx_control, vy_control, ax_control, ay_control, norm_v_t, angle_v_t

    def publish_path_message(self, traj):
        path = Path()
        header = Header()
        header.frame_id = self.robot_base_frame

        path.header = header

        for p in traj:
            pose = PoseStamped()
            pose.pose.position.x = p[0]
            pose.pose.position.y = p[1]
            path.poses.append(pose)  # type: ignore

        self.path_publisher.publish(path)

    def publish_zero_velocity(self):
        for _ in range(5):
            self.velocity_publisher.publish()


@initialize_configuration
def main(configuration: Configuration) -> None:
    check_configuration(configuration)

    if configuration.mode is Mode.LIVE_3DLIDAR:
        rospy.init_node("Planner")

        interface = ROSInterface(configuration)
        rate = rospy.Rate(10)
        rospy.on_shutdown(interface.publish_zero_velocity)

        while not rospy.is_shutdown():
            interface.plan()
            rate.sleep()
    else:
        raise ValueError(f"Mode {configuration.mode} not supported for live inference.")


if __name__ == "__main__":
    pause_physics_service = rospy.ServiceProxy("/gazebo/pause_physics", EmptyService)
    unpause_physics_service = rospy.ServiceProxy(
        "/gazebo/unpause_physics", EmptyService
    )
    main()
