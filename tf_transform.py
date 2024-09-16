#!/usr/bin/env python3
"""
@author: VAIL, IU
"""

import rospy
import tf
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry


def handle_jackal_pose(msg):
    br = tf.TransformBroadcaster()
    t = TransformStamped()
    # t.header.stamp = rospy.Time.now()
    t.header.stamp = msg.header.stamp
    t.header.frame_id = "world"
    t.child_frame_id = "base_link"
    t.transform.translation = msg.pose.pose.position
    t.transform.rotation = msg.pose.pose.orientation
    br.sendTransformMessage(t)


if __name__ == "__main__":
    rospy.init_node("tf_broadcaster")
    rospy.Subscriber("/robot1/odom", Odometry, handle_jackal_pose)
    rospy.spin()
