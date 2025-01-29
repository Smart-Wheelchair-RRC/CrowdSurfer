#!/bin/bash

echo $$
echo $$
echo $$
echo $$

source /opt/ros/noetic/setup.bash
source ../../../../devel/setup.bash
# roscore&
while true; do

	echo $$
	echo $$
	echo $$
	echo $$
	echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
	echo "Starting run"
	echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"

	roslaunch pedsim_simulator rerun.launch &
	# python3 rerun.py 
	echo "======++Waiting for the process to finish"
	wait $!

	echo "======================================================================================================================="
	echo "Completed run"
	echo "======================================================================================================================="
	sleep 5
done
wait