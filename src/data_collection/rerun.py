#!/usr/bin/env python3

import os
import rospy
import subprocess
import numpy as np
from std_msgs.msg import Bool
from pedsim_msgs.msg import AgentStates
from geometry_msgs.msg import PoseStamped, Quaternion
from nav_msgs.msg import OccupancyGrid, Odometry


class ReRun:
    """
    Implemening the following functionality
    - Provide goalpoints that are a fixed distance away from the robot
    - If robot hits a person, kill all the simulation and restart
    """

    def __init__(
        self, odom_topic="/odom", min_goal_dist=5, collision_padding=0.7
    ) -> None:
        print("================Starting Rerun Script================")
        # Minimum distance from robot to a new goal point
        self.min_goal_dist = min_goal_dist
        # Minimum distance from robot to person
        self.collision_padding = collision_padding

        rospy.init_node("data_col")

        self.map_check = False
        self.free_points: np.ndarray = None

        rospy.Subscriber(odom_topic, Odometry, self.odom_callback)
        rospy.Subscriber(
            "/pedsim_simulator/simulated_agents",
            AgentStates,
            self.agent_states_callback,
        )
        rospy.Subscriber("/reached", Bool, self.reached_callback)
        self.next_goal_pub = rospy.Publisher(
            "/move_base_simple/goal", PoseStamped, queue_size=10
        )

        self.odom: Odometry = None
        self.agent_states: AgentStates = None
        self.map: OccupancyGrid = None

        self.reached_goal: bool = False
        self.published_new_goal: bool = False

        self.map = rospy.wait_for_message("/map", OccupancyGrid)
        self.free_points = self.get_free_points()
        print("Free Points:", len(self.free_points))

        rospy.spin()

    def reached_callback(self, msg):
        self.reached_goal = msg
        if self.reached_goal and not self.published_new_goal:
            # If the agent has reached a new goal and a new goal hasnt been published publish a new goal
            new_goal = self.get_goal_point()
            print("++++++++++++++++++++++++++++++++++++++++++++")
            print("Reached a Goal. Publishing next goal")
            print("++++++++++++++++++++++++++++++++++++++++++++")

            goal_msg: PoseStamped = PoseStamped()
            goal_msg.header.frame_id = "map"
            goal_msg.header.stamp = rospy.Time.now()
            print("New Goal:", new_goal)
            goal_msg.pose.position.x = new_goal[0]
            goal_msg.pose.position.x = new_goal[1]
            goal_msg.pose.orientation = Quaternion(0, 0, 0, 1)

            self.next_goal_pub.publish(goal_msg)

    def agent_states_callback(self, msg: AgentStates):
        self.agent_states = msg

    def odom_callback(self, msg):
        self.odom = msg

        if self.check_collisions():
            print("Collision Detected")
            self.perform_shutdown_tasks()

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        # print(self.map is None)
        return [
            points[0] * self.map.info.resolution + self.map.info.origin.position.x,
            points[1] * self.map.info.resolution + self.map.info.origin.position.y,
        ]

    def get_free_points(self) -> np.ndarray:
        free_points = []
        if self.map is not None:
            self.map_data = np.array(self.map.data).reshape(
                self.map.info.height, self.map.info.width
            )
            print(np.where(self.map_data == 0))
            for x, y in zip(*np.where(self.map_data == 0)):
                real_world_coords = self.transform_points(np.array([x, y]))

                free_points.append(real_world_coords)

            return np.array(free_points)
        else:
            return np.array([])

    def get_goal_point(self) -> np.ndarray:
        # Sample a point from self.free_points and return it if its euclidean distance from the robot is greater than 1m
        if not self.free_points.any():
            return None

        random_idx = np.random.randint(0, len(self.free_points))
        goal_point = self.free_points[random_idx]

        if self.odom is None:
            return None

        robot_x = self.odom.pose.pose.position.x
        robot_y = self.odom.pose.pose.position.y

        distance = np.linalg.norm(goal_point - np.array([robot_x, robot_y]))

        if distance > self.min_goal_dist:
            return goal_point  # Return the goal point if its far enough
        else:
            # Recursively call get_goal_point if point is too close
            return self.get_goal_point()

    def check_collisions(self) -> bool:
        """
        Check if the robot is in collision with any person
        """
        if self.agent_states is None:
            print("No agent in scene")
            return False

        robot_position = np.array(
            [self.odom.pose.pose.position.x, self.odom.pose.pose.position.y]
        )

        for agent in self.agent_states.agent_states:
            agent_position = np.array([agent.pose.position.x, agent.pose.position.y])
            # print("Distance:", np.linalg.norm(robot_position - agent_position))
            if np.linalg.norm(robot_position - agent_position) < self.collision_padding:
                print("Collision Detected")
                rospy.logerr("Collision Detected. Shutting down all nodes!")
                return True
        # print("++++++++++++++++++++++++++++++++++++++++++++")
        # print("No Collision Detected")
        return False

    def perform_shutdown_tasks(self) -> None:
        # Kills the roslaunch process effectively killing all nodes
        # rospy.signal_shutdown("Collision Detected")
        noetic_procs = os.popen("ps aux | grep 'noetic' | grep -v 'grep'").readlines()

        # Kill each ROS process using `kill` command
        for process in noetic_procs:
            pid = process.split()[1]
            os.system(f"kill {pid}")

        pedsim_procs = os.popen("ps aux | grep 'pedsim' | grep -v 'grep'").readlines()

        for process in pedsim_procs:
            pid = process.split()[1]
            os.system(f"kill {pid}")

if __name__ == "__main__":
    rr = ReRun(
        odom_topic="/pedsim_simulator/robot_position",
        min_goal_dist=10,
    )
