#!/bin/bash

tmux new-session -d -s crowd_surfer

tmux split-window -v

tmux select-pane -t 0
tmux send-keys "roslaunch local_dynamic_nav global_nav.launch" C-m

tmux select-pane -t 1
tmux send-keys "conda activate PRIEST" C-m
tmux send-keys "rosrun local_dynamic_nav ros_interface_with_global_plan.py" C-m

tmux attach-session -t crowd_surfer
