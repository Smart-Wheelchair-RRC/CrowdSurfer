#!/usr/bin/env python3

import rospy
import numpy as np
from pedsim_msgs.msg import AgentStates
from nav_msgs.msg import Odometry

# ! TODO add code to save the metrics in a file after testing


class CalculateMetrics:
    """
    Calculate the following metrics for a particular run
    1. Average Speed
    2. Average Distance Travelled (length of trajectory)
    3. Number of collisions (A collision is defined as a distance of less than 0.5m between the robot and a person)
    """

    def __init__(self, odom_topic="/odom", collision_padding=0.5) -> None:
        rospy.init_node("calculate_metrics")
        rospy.Subscriber(self.odom_topic, Odometry, self.odom_callback)
        rospy.Subscriber("/pedsim_simulator/simulated_agents", AgentStates, self.agent_states_callback)
        self.odom_topic = odom_topic
        self.odom: Odometry = None
        self.agent_states: AgentStates = None
        self.position_array: np.ndarray = np.array([])
        self.velocity_array: np.ndarray = np.array([])

        self.collision_padding = collision_padding
        self.num_collisions = 0

    def agent_states_callback(self, msg: AgentStates) -> None:
        self.agent_states = msg

    def odom_callback(self, msg: Odometry) -> None:
        self.odom = msg
        self.position_array = np.append(self.position_array, [msg.pose.pose.position.x, msg.pose.pose.position.y])
        self.velocity_array = np.append(self.velocity_array, [msg.twist.twist.linear.x, msg.twist.twist.linear.y])

        if self.check_collision():
            print("Collision Detected")
            self.num_collisions += 1
            # rospy.signal_shutdown("Collision Detected")

    def get_avg_velocity(self) -> np.ndarray:
        return np.mean(self.velocity_array, axis=0)

    def check_collision(self) -> bool:
        """
        Check if the robot is in collision with any person
        """
        if self.agent_states is None:
            return False

        robot_position = np.array([self.odom.pose.pose.position.x, self.odom.pose.pose.position.y])

        for agent in self.agent_states.agent_states:
            agent_position = np.array([agent.pose.position.x, agent.pose.position.y])

            if np.linalg.norm(robot_position - agent_position) < self.collision_padding:
                return True

        return False

    def get_path_length(self) -> float:
        path_length = 0.0

        for i in range(len(self.position_array)):
            if i == 0:
                continue
            path_length += np.linalg.norm(self.position_array[i] - self.position_array[i - 1])
        return path_length


def main():
    metrics = CalculateMetrics()
    rospy.spin()
    if rospy.is_shutdown():
        print("=====================================")
        print("Printing Metrics for the Current Run")
        print("1. Average Speed: ", metrics.get_avg_velocity(), "m/s")
        print("2. Average Distance Travelled: ", metrics.get_path_length(), "m")
        print(
            "3. Number of Collisions: ",
            metrics.num_collisions,
            f"collisions in {metrics.collision_padding}m around robot",
        )
        print("=====================================")
