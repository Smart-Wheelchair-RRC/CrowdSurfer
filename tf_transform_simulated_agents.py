#!/usr/bin/env python3

import rospy
import tf
from geometry_msgs.msg import TransformStamped
from pedsim_msgs.msg import AgentStates
from std_msgs.msg import Header
from tf.transformations import quaternion_matrix
import numpy as np


class Transform_Simulated_Agents:
    def __init__(
        self,
        world_frame="map",
        robot_base_frame="base_link",
        subsriber_topic="/pedsim_simulator/simulated_agents",
        publisher_topic="/pedsim_simulator/agents_updated_in_base_frame",
    ):
        self.listener = tf.TransformListener()
        self.world_frame = world_frame
        self.robot_base_frame = robot_base_frame

        self.sub_agent_states = rospy.Subscriber(
            subsriber_topic,
            AgentStates,
            self.handle_simulated_agents,
        )

        self.pub_updated_states = rospy.Publisher(publisher_topic, AgentStates, queue_size=10)

    def handle_simulated_agents(self, msg: AgentStates):
        try:
            (trans, rot) = self.listener.lookupTransform(self.world_frame, self.robot_base_frame, rospy.Time(0))
        except (
            tf.LookupException,
            tf.ConnectivityException,
            tf.ExtrapolationException,
        ):
            print(f"Lookup failed between {self.world_frame} and {self.robot_base_frame}")
            return

        # translated_goal = np.array(self.goal) - np.array(trans[:2])

        rotatation_matrix = quaternion_matrix(np.array(rot))
        # Transpose
        rotatation_matrix = rotatation_matrix[:3, :3].T

        updated_agent_states = self.update_agent_states(msg.agent_states, rotatation_matrix, np.array(trans))

        out_msg = AgentStates()
        out_msg.agent_states = updated_agent_states

        self.pub_updated_states.publish(out_msg)

    # Orientation is not being updated
    def update_agent_states(self, agent_states, rotatation_matrix, translation):
        header = Header()
        header.frame_id = self.robot_base_frame

        updated_agents = []
        for agent in agent_states:
            agent.header = header
            pose = np.array([agent.pose.position.x, agent.pose.position.y, 0]).reshape((3, 1))
            pose[:2, 0] = pose[:2, 0] - translation[:2]
            pose = rotatation_matrix @ pose

            vel = np.array([agent.twist.linear.x, agent.twist.linear.y, 0]).reshape((3, 1))
            vel = rotatation_matrix @ vel

            print(
                "before",
                agent.pose.position.x,
                agent.pose.position.y,
                agent.twist.linear.x,
                agent.twist.linear.y,
            )

            agent.pose.position.x = pose[0, 0]
            agent.pose.position.y = pose[1, 0]
            agent.twist.linear.x = vel[0, 0]
            agent.twist.linear.y = vel[1, 0]

            updated_agents.append(agent)
            print(
                "after",
                agent.pose.position.x,
                agent.pose.position.y,
                agent.twist.linear.x,
                agent.twist.linear.y,
            )

        return updated_agents


if __name__ == "__main__":
    rospy.init_node("Agent_states_transformer")

    transformer = Transform_Simulated_Agents()

    while not rospy.is_shutdown():
        rospy.spin()
