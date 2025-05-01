#!/bin/bash

tmux new-session -d -s crowd_surfer

tmux split-window -v

tmux select-pane -t 0
tmux send-keys "source devel/setup.bash" C-m
tmux send-keys "roslaunch crowdsurfer_ros global_nav.launch" C-m

tmux select-pane -t 1
tmux send-keys "source devel/setup.bash" C-m
tmux send-keys "conda activate crowdsurfer" C-m
tmux send-keys "rosrun crowdsurfer_ros ros_interface_with_global_plan.py" C-m

tmux attach-session -t crowd_surfer