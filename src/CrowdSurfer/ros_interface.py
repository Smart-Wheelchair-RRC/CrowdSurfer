#!/usr/bin/env python3

import os

import message_filters
import numpy as np
import rospy

import time

# from tf.transformations import euler_from_quaternion
import tf
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry, Path
from pedsim_msgs.msg import AgentStates
from sensor_msgs.msg import LaserScan, PointCloud
from std_msgs.msg import Header
from tf.transformations import euler_from_quaternion, quaternion_matrix
from visualization_msgs.msg import MarkerArray

from configuration import (
    Configuration,
    Dynamic_Obstacles_Msg_Type,
    Mode,
    check_configuration,
    initialize_configuration,
)
from inference import LivePipeline

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


class ROSInterface:
    def __init__(self, configuration: Configuration):
        self.pipeline = LivePipeline(configuration)

        self.Pdot = self.pipeline.projection_guidance.BERNSTEIN_POLYNOMIALS_FIRST_DIFFERENTIAL.detach().cpu().numpy()
        self.Pddot = self.pipeline.projection_guidance.BERNSTEIN_POLYNOMIALS_SECOND_DIFFERENTIAL.detach().cpu().numpy()

        ## use if acceleration output is required
        # self.Pddot = (
        #     self.pipeline.projection_guidance.BERNSTEIN_POLYNOMIALS_SECOND_DIFFERENTIAL.detach()
        #     .cpu()
        #     .numpy()
        # )
        # self.config = configuration

        self.total_rollout_time = 0
        self.num_rollouts = 0

        self.world_frame = configuration.live.world_frame
        self.robot_base_frame = configuration.live.robot_base_frame
        self.odom = np.full((3), 0.0)
        self.laser_scan = np.full((2, 1080), configuration.projection.padding)
        self.point_cloud = np.full((2, 1000), configuration.projection.padding)
        # self.point_cloud = np.arange(2000).reshape(2, 1000)
        self.vel = [0, 0]
        self.acc = [0, 0]
        self.dynamic_obstacles = np.full(
            (
                configuration.live.previous_time_steps_for_dynamic,
                4,
                self.pipeline.max_projection_dynamic_obstacles,
            ),
            configuration.projection.padding,
        )
        self.dynamic_obstacles[:, 2:, :] = 0

        self.listener = tf.TransformListener()
        self.sub_scan = message_filters.Subscriber(configuration.live.laser_scan_topic, LaserScan)
        self.sub_pcd = message_filters.Subscriber(configuration.live.point_cloud_topic, PointCloud)
        self.sub_odom = message_filters.Subscriber(configuration.live.odometry_topic, Odometry)

        ts = message_filters.ApproximateTimeSynchronizer(
            [self.sub_odom, self.sub_pcd, self.sub_scan],
            queue_size=20,
            slop=1,
            allow_headerless=True,
        )
        ts.registerCallback(self.callback)

        self.create_dynamic_obstacle_subscriber(
            configuration.live.dynamic_obstacle_topic, configuration.live.dynamic_msg
        )

        self.received_goal = False
        self.goal = [0, 0]
        self.goal_in_current_frame = self.goal
        self.sub_goal = rospy.Subscriber(configuration.live.goal_topic, PoseStamped, self.update_goal)

        self.pub_vel = rospy.Publisher(configuration.live.velocity_command_topic, Twist, queue_size=10)
        self.pub_path = rospy.Publisher(configuration.live.path_topic, Path, queue_size=10)

        # rospy.Timer(rospy.Duration(10), self.timer_callback)

        self.num_sample = 6
        # self.i = -1

    def create_dynamic_obstacle_subscriber(self, topic_name: str, msg_type: Dynamic_Obstacles_Msg_Type):
        if msg_type == Dynamic_Obstacles_Msg_Type.MARKER_ARRY:
            self.sub_dynamic = rospy.Subscriber(
                topic_name,
                MarkerArray,
                self.update_dynamic_obstacles_from_marker_array,
            )
        elif msg_type == Dynamic_Obstacles_Msg_Type.AGENT_STATES:
            self.sub_dynamic = rospy.Subscriber(
                topic_name, AgentStates, self.update_dynamic_obstacles_from_agent_states
            )

        else:
            raise NotImplementedError(
                f"For dynamic obstacles, support for message type {msg_type.name} hasn't been implmented yet"
            )

    def update_goal(self, goal: PoseStamped):
        # _, _, yaw = euler_from_quaternion([goal.pose.orientation.x, goal.pose.orientation.y, goal.pose.orientation.z, goal.pose.orientation.w])
        # self.goal = [goal.pose.position.x, goal.pose.position.y, yaw]
        self.goal = [goal.pose.position.x, goal.pose.position.y]
        self.heading_to_goal = np.arctan2(self.goal[1], self.goal[0])
        self.received_goal = True
        # self.i = 0
        self.update_goal_and_check_distance()

    # def timer_callback(self, event):
    #     print("here")
    #     self.update_goal_and_check_distance()

    def update_goal_and_check_distance(self):
        self.update_goal_in_current_frame()

        dist = np.sqrt(self.goal_in_current_frame[0] ** 2 + self.goal_in_current_frame[1] ** 2)
        if dist <= self.pipeline.threshold_distance:
            cmd_vel = Twist()
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = 0.0

            if self.received_goal:
                if self.num_rollouts != 0:
                    print("Reached Goal")
                    print("Average Rollout Time: ", self.total_rollout_time / self.num_rollouts)
                self.total_rollout_time = 0
                self.num_rollouts = 0
                self.received_goal = False
                # self.i = -1

        # print("1", self.goal)
        # print("2", self.goal_in_current_frame)

    def update_goal_in_current_frame(self):
        try:
            (trans, rot) = self.listener.lookupTransform(self.world_frame, self.robot_base_frame, rospy.Time(0))
        except (
            tf.LookupException,
            tf.ConnectivityException,
            tf.ExtrapolationException,
        ):
            print(f"Lookup failed between {self.world_frame} and {self.robot_base_frame}")
            return

        translated_goal = np.array(self.goal) - np.array(trans[:2])

        transformation_matrix = quaternion_matrix(np.array(rot))
        # Transpose
        transformation_matrix = transformation_matrix[:3, :3].T
        goal = np.zeros((3, 1))
        goal[:2, 0] = translated_goal

        goal_in_current_frame = transformation_matrix @ goal
        self.goal_in_current_frame = goal_in_current_frame[:2, 0]
        # self.goal_in_current_frame = self.goal
        # print(self.goal_in_current_frame.shape)

    def update_laser_scan(self, laser_scan_message: LaserScan):
        max_useful_range = 50
        maximum_points = 1080
        ranges = np.array(laser_scan_message.ranges)

        # Check for zero or very small angle_increment
        if abs(laser_scan_message.angle_increment) < 1e-10:
            num_points = len(ranges)
            angles = np.linspace(laser_scan_message.angle_min, laser_scan_message.angle_max, num_points)
        else:
            angles = np.arange(
                laser_scan_message.angle_min,
                laser_scan_message.angle_max + laser_scan_message.angle_increment,
                laser_scan_message.angle_increment,
            )

        # Ensure angles and ranges have the same length. If not, set as same.
        if len(angles) != len(ranges):
            min_len = min(len(angles), len(ranges))
            angles = angles[:min_len]
            ranges = ranges[:min_len]

        valid_ranges = (
            (ranges >= laser_scan_message.range_min)
            & ((ranges <= laser_scan_message.range_max) | (np.isinf(laser_scan_message.range_max)))
            & (ranges <= max_useful_range)
        )
        valid_ranges_array = ranges[valid_ranges]
        valid_angles_array = angles[valid_ranges]

        # Pad remaining positions for consistent array length
        padded_ranges = np.pad(
            valid_ranges_array,
            (0, maximum_points - len(valid_ranges_array)),
            mode="constant",
            constant_values=np.nan,
        )[:maximum_points]
        padded_angles = np.pad(
            valid_angles_array,
            (0, maximum_points - len(valid_angles_array)),
            mode="constant",
            constant_values=np.nan,
        )[:maximum_points]

        self.laser_scan = np.stack((padded_ranges, padded_angles), axis=-1)

    def callback(
        self,
        odom: Odometry,
        point_cloud_message: PointCloud,
        laser_scan_message: LaserScan,
    ):
        x, y, theta = odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.orientation.z
        # print(odom.pose.pose.position.x, odom.pose.pose.position.y)
        _, _, theta = euler_from_quaternion(
            [
                odom.pose.pose.orientation.x,
                odom.pose.pose.orientation.y,
                odom.pose.pose.orientation.z,
                odom.pose.pose.orientation.w,
            ]
        )
        self.odom = np.array([x, y, theta])
        # TODO: split into vx, vy ?
        # self.vel = [v, w]

        self.update_laser_scan(laser_scan_message)

        self.update_pointcloud(point_cloud_message)

    def update_pointcloud(self, point_cloud_message: PointCloud):
        pcd_array = []
        for p in point_cloud_message.points:
            pcd_array.append([p.x, p.y, p.z])

        self.point_cloud = np.array(pcd_array)

    def update_dynamic_obstacles_from_marker_array(self, dynamic_obstacles: MarkerArray):
        if len(dynamic_obstacles.markers) > 0:
            obs_array = []
            for obs in dynamic_obstacles.markers:
                obs_array.append([obs.pose.position.x, obs.pose.position.y, 0, 0])

        self.update_dynamic_obstacles(obs_array)

    def update_dynamic_obstacles_from_agent_states(self, dynamic_obstacles: AgentStates):
        if len(dynamic_obstacles.agent_states) > 0:
            obs_array = []
            for obs in dynamic_obstacles.agent_states:
                obs_array.append(
                    [
                        obs.pose.position.x,
                        obs.pose.position.y,
                        obs.twist.linear.x,
                        obs.twist.linear.y,
                    ]
                )
        self.update_dynamic_obstacles(obs_array)

    def update_dynamic_obstacles(self, obs_array: list):
        self.dynamic_obstacles[0:-1] = self.dynamic_obstacles[1:]
        self.dynamic_obstacles[-1, 0:2, :] = self.pipeline.padding_obstacle
        self.dynamic_obstacles[-1, 2:, :] = 0

        # get closest dynamic obstacles
        obs_array = np.array(obs_array)
        obs_array = obs_array[np.argsort(np.linalg.norm(obs_array[:, :2], axis=-1), axis=-1)][
            : self.pipeline.max_projection_dynamic_obstacles
        ]
        obs_array = obs_array.T

        self.dynamic_obstacles[-1, :, 0 : obs_array.shape[1]] = obs_array

    def check_dynamic_obstacle_distance_for_stopping(self, threshold_distance: float = 1.0, num_obstacles: int = 1):
        current_obstacles = self.dynamic_obstacles[-1, :2, :]

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
        self.update_goal_and_check_distance()

        if self.received_goal and not self.check_dynamic_obstacle_distance_for_stopping():
            cmd_vel = Twist()

            start_time = time.perf_counter()

            coefficients, trajectories, scores = self.pipeline.run(
                np.array(self.goal_in_current_frame, dtype=np.float32),
                np.array(self.vel, dtype=np.float32),
                np.array(self.acc, dtype=np.float32),
                np.array(self.laser_scan, dtype=np.float32),
                np.array(self.point_cloud, dtype=np.float32),
                np.array(self.dynamic_obstacles, dtype=np.float32),
            )

            self.total_rollout_time = time.perf_counter() - start_time
            self.num_rollouts += 1
            best_index = np.argmax(scores)
            best_coeffs = coefficients[best_index]
            x_best, y_best = best_coeffs[:, 0], best_coeffs[:, 1]
            # print(x_best.shape)

            best_traj = trajectories[best_index]
            self.publish_path_message(best_traj)

            vx_control, vy_control, ax_control, ay_control, norm_v_t, angle_v_t = self.compute_controls(
                x_best * 0.8, y_best * 0.8
            )
            # vx_control, vy_control, norm_v_t, angle_v_t = self.compute_controls(x_best, y_best)

            self.vel = [vx_control, vy_control]
            self.acc = [ax_control, ay_control]

            # zeta = self.convert_angle(self.theta_init) - self.convert_angle(angle_v_t)
            zeta = self.convert_angle(0) - self.convert_angle(angle_v_t)
            v_t_control = norm_v_t * np.cos(zeta)
            # print("zeta", zeta)

            omega_control = -zeta / (self.num_sample * 5 * 0.01)

            # self.vx_init = vx_control
            # self.vy_init = vy_control

            # self.ax_init = ax_control
            # self.ay_init = ay_control

            cmd_vel.linear.x = v_t_control
            cmd_vel.angular.z = omega_control

            # print("Control", v_t_control, omega_control)
            self.pub_vel.publish(cmd_vel)

        else:
            self.publish_zero_vel()

        # # if self.i <= 3:
        # cmd_vel.linear.x = 0.0
        # cmd_vel.angular.z = 0.0
        # #     self.cmd_vel_pub.publish(cmd_vel)
        # #     self.i += 1

        # # else:

        # print(v_t_control, omega_control)
        # rospy.sleep(0.005)

    def convert_angle(self, angle):
        angle = np.unwrap(np.array([angle]), discont=np.pi, axis=0, period=6.283185307179586)

        return angle

    def compute_controls(self, x_best, y_best):
        xdot_best = np.dot(
            self.Pdot,
            x_best,
        )
        ydot_best = np.dot(
            self.Pdot,
            y_best,
        )

        xddot_best = np.dot(self.Pddot, x_best)
        yddot_best = np.dot(self.Pddot, y_best)

        vx_control = np.mean(xdot_best[0 : self.num_sample])
        vy_control = np.mean(ydot_best[0 : self.num_sample])

        ax_control = np.mean(xddot_best[0 : self.num_sample])
        ay_control = np.mean(yddot_best[0 : self.num_sample])

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
            # pose.pose.position.z = 0
            path.poses.append(pose)

        self.pub_path.publish(path)

    def publish_zero_vel(self):
        cmd_vel = Twist()
        cmd_vel.linear.x = 0.0
        cmd_vel.angular.z = 0.0

        for i in range(5):
            self.pub_vel.publish()


@initialize_configuration
def main(configuration: Configuration) -> None:
    check_configuration(configuration)

    if configuration.mode is Mode.LIVE:
        rospy.init_node("Planner")

        interface = ROSInterface(configuration)
        rate = rospy.Rate(10)
        rospy.on_shutdown(interface.publish_zero_vel)

        while not rospy.is_shutdown():
            interface.plan()
            rate.sleep()

    else:
        raise ValueError(f"Mode {configuration.mode} not supported for live inference.")


if __name__ == "__main__":
    main()
