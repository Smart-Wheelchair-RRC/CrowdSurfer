# CrowdSurfer: Sampling Optimization Augmented with Vector-Quantized Variational AutoEncoder for Dense Crowd Navigation

**Contributors:** Naman Kumar*<sup>1</sup>, Antareep Singha*<sup>1</sup>, Laksh Nanwani\*<sup>1</sup>, Dhruv Potdar<sup>1</sup>, Tarun R.<sup>1</sup>, Fatemeh Rastgar<sup>2</sup>, Simon Idoko<sup>2</sup>, Arun Kumar Singh<sup>2</sup>, K. Madhava Krishna<sup>1</sup>

<sup>1</sup>_Robotics Research Center, IIIT Hyderabad_; <sup>2</sup>_University of Tartu, Estonia_

\* denotes equal contribution

Accepted at IEEE International Conference on Robotics and Automation (ICRA) 2025

[Paper](https://arxiv.org/abs/2409.16011) | [Video](https://youtu.be/BMDCYdxfaXM) | [Website](https://smart-wheelchair-rrc.github.io/CrowdSurfer-webpage/)

![teaser](./crowdsurfer.png)

## Abstract

Navigation amongst densely packed crowds remains a challenge for mobile robots. The complexity increases further if the environment layout changes making the prior computed global plan infeasible. In this paper, we show that it is possible to dramatically enhance crowd navigation by just improving the local planner. Our approach combines generative modelling with inference time optimization to generate sophisticated long-horizon local plans at interactive rates. More specifically, we train a Vector Quantized Variational AutoEncoder to learn a prior over the expert trajectory distribution conditioned on the perception input. At run-time, this is used as an initialization for a sampling-based optimizer for further refinement. Our approach does not require any sophisticated prediction of dynamic obstacles and yet provides state-of-the- art performance. In particular, we compare against the recent DRL-VO approach and show a 40% improvement in success rate and a 6% improvement in travel time.

## Overview

This repository contains the code for the paper "CrowdSurfer: Sampling Optimization Augmented with Vector-Quantized Variational AutoEncoder for Dense Crowd Navigation".

All configuration is done via [this configuration file](./src/CrowdSurfer/configuration/configuration.yaml).
To run once setup, use this [script](./src/CrowdSurfer/run_CrowdSurfer.sh) to run the code in a tmux session.

## Installation

Setup your conda environment called "crowdsurfer" with the following packages:

-   CUDA 12.1
-   PyTorch
-   JAX
-   Hydra
-   HuggingFace Accelerate
-   Open3D
-   Scikit Learn

Running the simulation in gazebo requires pedsim_ros (to simulate the humans).
To install pedsim_ros and its other dependencies, proceed as follows:
The default version is ROS Noetic.

```bash
mkdir -p crowdsurfer_ws/src && cd crowdsurfer_ws/src
git clone https://github.com/TempleRAIL/robot_gazebo.git
git clone https://github.com/Smart-Wheelchair-RRC/pedsim_ros_with_gazebo.git
wget https://raw.githubusercontent.com/zzuxzt/turtlebot2_noetic_packages/master/turtlebot2_noetic_install.sh
sudo sh turtlebot2_noetic_install.sh
git clone https://github.com/Smart-Wheelchair-RRC/CrowdSurfer.git
cd ..
catkin build
```

## Run Demo

Download checkpoints from [this link](https://drive.google.com/drive/folders/1HSRrbuwwNk9_C1WKN9qnStjemFLukO8s)

1. Replace the checkpoint paths for VQVAE and Scoring Network in the [configuration file](./src/CrowdSurfer/configuration/configuration.yaml)
2. From within crowdsurfer_ws/ in tmux, run:

```bash
bash src/CrowdSurfer/src/CrowdSurfer/run_CrowdSurfer.sh
```
