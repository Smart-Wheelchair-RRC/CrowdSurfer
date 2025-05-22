#!/usr/bin/env python3
#
# revision history: xzt
#  20210604 (TE): first version
#
# usage:
#
# This script is to publish the robot pose.
# ------------------------------------------------------------------------------

import geometry_msgs.msg
import numpy as np
import rospy
import tf
from geometry_msgs.msg import Point, Pose, PoseStamped, Twist, TwistStamped


def robot_pose_pub():
    rospy.init_node("robot_pose", anonymous=True)
    tf_listener = tf.TransformListener()
    robot_pose_pub = rospy.Publisher("/robot_pose", PoseStamped, queue_size=1)
    rate = rospy.Rate(30)  # 10hz
    world_frame = rospy.get_param("~world_frame", "/map")
    robot_frame = rospy.get_param("~robot_frame", "/base_link")
    while not rospy.is_shutdown():
        trans = rot = None
        # look up the current pose of the base_footprint using the tf tree
        try:
            (trans, rot) = tf_listener.lookupTransform(
                world_frame, robot_frame, rospy.Time(0)
            )
        except (
            tf.LookupException,
            tf.ConnectivityException,
            tf.ExtrapolationException,
        ):
            rospy.logwarn("Could not get robot pose")
            trans = list([-1, -1, -1])
            rot = list([-1, -1, -1, -1])
        # publish robot pose:
        rob_pos = PoseStamped()
        rob_pos.header.stamp = rospy.Time.now()
        rob_pos.header.frame_id = "/map"
        rob_pos.pose.position.x = trans[0]
        rob_pos.pose.position.y = trans[1]
        rob_pos.pose.position.z = trans[2]
        rob_pos.pose.orientation.x = rot[0]
        rob_pos.pose.orientation.y = rot[1]
        rob_pos.pose.orientation.z = rot[2]
        rob_pos.pose.orientation.w = rot[3]
        robot_pose_pub.publish(rob_pos)

        rate.sleep()


if __name__ == "__main__":
    try:
        robot_pose_pub()
    except rospy.ROSInterruptException:
        pass
