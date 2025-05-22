#!/usr/bin/env python3

import colorsys
import os
import time

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
from pedsim_msgs.msg import AgentStates, TrackedPersons
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Empty, Header
from visualization_msgs.msg import MarkerArray

np.float = np.float64
import ros_numpy

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


class ROSInterface:
    def __init__(self, configuration: Configuration):
        self.pipeline = LivePipeline(configuration)

        self.Pdot = (
            self.pipeline.projection_guidance.BERNSTEIN_POLYNOMIALS_FIRST_DIFFERENTIAL.detach()
            .cpu()
            .numpy()
        )
        self.Pddot = (
            self.pipeline.projection_guidance.BERNSTEIN_POLYNOMIALS_SECOND_DIFFERENTIAL.detach()
            .cpu()
            .numpy()
        )

        # use if acceleration output is required
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
        self.point_cloud = np.full((1000, 3), configuration.projection.padding)
        # self.point_cloud = np.arange(2000).reshape(2, 1000)
        self.vel = [0, 0]
        self.acc = [0, 0]
        self.dynamic_obstacles = np.full(
            (
                5,
                4,
                self.pipeline.max_projection_dynamic_obstacles,
            ),
            configuration.projection.padding,
            dtype=float,
        )
        self.dynamic_obstacles[:, 2:, :] = 0

        self.listener = tf.TransformListener()
        self.sub_pcd = rospy.Subscriber(
            configuration.live.point_cloud_topic, PointCloud2, self.update_pointcloud
        )

        self.create_dynamic_obstacle_subscriber(
            configuration.live.dynamic_obstacle_topic,
            configuration.live.dynamic_obstacle_message_type,
        )

        self.received_goal = False
        self.goal = [0, 0]
        self.subgoal = self.goal
        self.sub_final_goal = rospy.Subscriber(
            configuration.live.goal_topic, PoseStamped, self.update_goal
        )
        self.sub_subgoal = rospy.Subscriber(
            configuration.live.sub_goal_topic, PoseStamped, self.update_subgoal
        )

        self.pub_vel = rospy.Publisher(
            configuration.live.velocity_command_topic, Twist, queue_size=10
        )
        self.pub_path = rospy.Publisher(
            configuration.live.path_topic, Path, queue_size=10
        )
        self.pub_trajectories = rospy.Publisher(
            "elite_traj", MarkerArray, queue_size=10
        )

        # rospy.Timer(rospy.Duration(10), self.timer_callback)

        self.num_sample = 6
        # self.i = -1
        self.max_angular_vel = configuration.live.max_angular_velocity

    def create_dynamic_obstacle_subscriber(
        self, topic_name: str, msg_type: DynamicObstaclesMessageType
    ):
        if msg_type == DynamicObstaclesMessageType.MARKER_ARRAY:
            self.sub_dynamic = rospy.Subscriber(
                topic_name,
                MarkerArray,
                self.update_dynamic_obstacles_from_marker_array,
            )

        elif msg_type == DynamicObstaclesMessageType.AGENT_STATES:
            self.sub_dynamic = rospy.Subscriber(
                topic_name, AgentStates, self.update_dynamic_obstacles_from_agent_states
            )

        elif msg_type == DynamicObstaclesMessageType.TRACKED_PERSONS:
            self.sub_dynamic = rospy.Subscriber(
                topic_name,
                TrackedPersons,
                self.update_dynamic_obstacles_from_tracked_persons,
            )

        else:
            raise NotImplementedError(
                f"For dynamic obstacles, support for message type {msg_type.name} hasn't been implmented yet"
            )

    def update_goal(self, goal: PoseStamped):
        # _, _, yaw = euler_from_quaternion([goal.pose.orientation.x, goal.pose.orientation.y, goal.pose.orientation.z, goal.pose.orientation.w])
        # self.goal = [goal.pose.position.x, goal.pose.position.y, yaw]
        self.goal = [goal.pose.position.x, goal.pose.position.y]
        # self.heading_to_goal = np.arctan2(self.goal[1], self.goal[0])
        self.received_goal = True
        # self.i = 0
        self.check_distance()

    def update_subgoal(self, subgoal: PoseStamped):
        # _, _, yaw = euler_from_quaternion([goal.pose.orientation.x, goal.pose.orientation.y, goal.pose.orientation.z, goal.pose.orientation.w])
        # self.goal = [goal.pose.position.x, goal.pose.position.y, yaw]
        self.subgoal = [subgoal.pose.position.x, subgoal.pose.position.y]
        self.heading_to_subgoal = np.arctan2(self.subgoal[1], self.subgoal[0])

    # def timer_callback(self, event):
    #     print("here")
    #     self.update_goal_and_check_distance()

    def check_distance(self):
        dist = np.sqrt(self.goal[0] ** 2 + self.goal[1] ** 2)
        if dist <= self.pipeline.threshold_distance:
            if self.received_goal:
                self.received_goal = False
                if self.num_rollouts != 0:
                    print("Reached Goal")
                    print(
                        "Average Rollout Time: ",
                        self.total_rollout_time / self.num_rollouts,
                    )
                self.total_rollout_time = 0
                self.num_rollouts = 0

    def update_pointcloud(self, point_cloud_message: PointCloud2):
        self.point_cloud = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(
            point_cloud_message
        )

    def update_dynamic_obstacles_from_marker_array(
        self, dynamic_obstacles: MarkerArray
    ):
        if len(dynamic_obstacles.markers) > 0:
            obs_array = []
            for obs in dynamic_obstacles.markers:
                obs_array.append([obs.pose.position.x, obs.pose.position.y, 0, 0])

        self.update_dynamic_obstacles(obs_array)

    def update_dynamic_obstacles_from_agent_states(
        self, dynamic_obstacles: AgentStates
    ):
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

    def update_dynamic_obstacles_from_tracked_persons(
        self, dynamic_obstacles: TrackedPersons
    ):
        if len(dynamic_obstacles.tracks) > 0:
            obs_array = []
            for obs in dynamic_obstacles.tracks:
                obs_array.append(
                    [
                        obs.pose.pose.position.x,
                        obs.pose.pose.position.y,
                        obs.twist.twist.linear.x,
                        obs.twist.twist.linear.y,
                    ]
                )
        self.update_dynamic_obstacles(obs_array)

    def update_dynamic_obstacles(self, obs_array: list):
        self.dynamic_obstacles[0:-1] = self.dynamic_obstacles[1:]
        self.dynamic_obstacles[-1, 0:2, :] = self.pipeline.padding_obstacle
        self.dynamic_obstacles[-1, 2:, :] = 0

        # get closest dynamic obstacles
        obs_array = np.array(obs_array, dtype=float)
        obs_array = obs_array[
            np.argsort(np.linalg.norm(obs_array[:, :2], axis=-1), axis=-1)
        ][: self.pipeline.max_projection_dynamic_obstacles]
        obs_array = obs_array.T

        self.dynamic_obstacles[-1, :, 0 : obs_array.shape[1]] = obs_array

    def check_dynamic_obstacle_distance_for_stopping(
        self, threshold_distance: float = 1.2, num_obstacles: int = 1
    ):
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
        self.check_distance()

        if self.received_goal:
            cmd_vel = Twist()

            start_time = time.perf_counter()
            coefficients, trajectories, scores = self.pipeline.run(
                np.array(self.subgoal, dtype=np.float32),
                np.array(self.vel, dtype=np.float32),
                np.array(self.acc, dtype=np.float32),
                np.array(self.point_cloud, dtype=np.float32),
                np.array(self.dynamic_obstacles, dtype=np.float32),
            )

            self.total_rollout_time += time.perf_counter() - start_time
            self.num_rollouts += 1

            best_index = np.argmax(scores)
            best_coeffs = coefficients[best_index]
            x_best, y_best = best_coeffs[:, 0], best_coeffs[:, 1]
            # print(x_best.shape)

            best_traj = trajectories[best_index]
            # Publish the selected trajectory and all the other trajectories
            # Check: Frame ID (ego or map)
            self.publish_trajectories_markers(
                trajectories, frame_id=self.robot_base_frame
            )
            self.publish_path_message(best_traj)

            vx_control, vy_control, ax_control, ay_control, norm_v_t, angle_v_t = (
                self.compute_controls(x_best * 0.8, y_best * 0.8)
            )
            # vx_control, vy_control, norm_v_t, angle_v_t = self.compute_controls(x_best, y_best)

            self.vel = [vx_control, vy_control]
            self.acc = [ax_control, ay_control]

            # zeta = self.convert_angle(self.theta_init) - self.convert_angle(angle_v_t)
            zeta = self.convert_angle(0) - self.convert_angle(angle_v_t)
            v_t_control = norm_v_t * np.cos(zeta)
            # print("zeta", zeta)

            omega_control = -zeta / (self.num_sample * 15 * 0.01)
            v_t_control = min(max(v_t_control, 0), self.pipeline.priest_planner.v_max)

            if np.abs(omega_control) > self.max_angular_vel:
                omega_control = np.sign(omega_control) * self.max_angular_vel

            cmd_vel.linear.x = v_t_control
            cmd_vel.angular.z = omega_control

            print("Control", v_t_control, omega_control)
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

    def publish_trajectories_markers(self, trajectories, frame_id="map"):
        # Assuming trajectories has a shape of [a, b, 2] where a is index of trajectory and b is length of trajectory
        marker_array = MarkerArray()
        num_trajectories = trajectories.shape[0]

        for traj_idx in range(num_trajectories):
            # Create a marker for each trajectory
            marker = Marker()
            marker.header.frame_id = frame_id
            marker.header.stamp = rospy.Time.now()
            marker.ns = "trajectory"
            marker.id = traj_idx
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD

            marker.scale.x = 0.008  # Line width

            # Generate unique color using HSV color space
            hue = traj_idx / float(num_trajectories)
            rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
            marker.color.r = rgb[0]
            marker.color.g = rgb[1]
            marker.color.b = rgb[2]
            marker.color.a = 1.0

            marker.pose.orientation.w = 1.0

            # Add trajectory points
            traj_points = trajectories[traj_idx]

            marker.points = [
                Point(x=point[0], y=point[1], z=0.0) for point in traj_points
            ]

            marker_array.markers.append(marker)

        # Publish the marker array
        self.pub_trajectories.publish(marker_array)

    def convert_angle(self, angle):
        angle = np.unwrap(
            np.array([angle]), discont=np.pi, axis=0, period=6.283185307179586
        )

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
