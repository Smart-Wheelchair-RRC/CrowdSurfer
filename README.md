# CrowdSurfer: Sampling Optimization Augmented with Vector-Quantized Variational AutoEncoder for Dense Crowd Navigation

**Contributors:** Naman Kumar*<sup>1</sup>, Antareep Singha*<sup>1</sup>, Laksh Nanwani\*<sup>1</sup>, Dhruv Potdar<sup>1</sup>, Tarun R.<sup>1</sup>, Fatemeh Rastgar<sup>2</sup>, Simon Idoko<sup>2</sup>, Arun Kumar Singh<sup>2</sup>, K. Madhava Krishna<sup>1</sup>

<sup>1</sup>_Robotics Research Center, IIIT Hyderabad_; <sup>2</sup>_University of Tartu, Estonia_

\* denotes equal contribution

Accepted at **IEEE International Conference on Robotics and Automation (ICRA) 2025**

[Paper](https://arxiv.org/abs/2409.16011) | [Video](https://youtu.be/BMDCYdxfaXM) | [Website](https://smart-wheelchair-rrc.github.io/CrowdSurfer-webpage/)

![teaser](./crowdsurfer.png)

## Abstract

Navigation amongst densely packed crowds remains a challenge for mobile robots. The complexity increases further if the environment layout changes making the prior computed global plan infeasible. In this paper, we show that it is possible to dramatically enhance crowd navigation by just improving the local planner. Our approach combines generative modelling with inference time optimization to generate sophisticated long-horizon local plans at interactive rates. More specifically, we train a Vector Quantized Variational AutoEncoder to learn a prior over the expert trajectory distribution conditioned on the perception input. At run-time, this is used as an initialization for a sampling-based optimizer for further refinement. Our approach does not require any sophisticated prediction of dynamic obstacles and yet provides state-of-the- art performance. In particular, we compare against the recent DRL-VO approach and show a 40% improvement in success rate and a 6% improvement in travel time.

## Overview

This repository contains the code for the paper "CrowdSurfer: Sampling Optimization Augmented with Vector-Quantized Variational AutoEncoder for Dense Crowd Navigation".

All configuration is done via [this configuration file](./src/CrowdSurfer/configuration/configuration.yaml).

## Installation (Prerequisites - ROS1 Noetic, CUDA 11.8 / 12.1)

### Create a workspace
```bash
mkdir -p ~/crowdsurfer_ws/src
cd ~/crowdsurfer_ws/src
```

### Clone CrowdSurfer
```
git clone https://github.com/Smart-Wheelchair-RRC/CrowdSurfer.git
```

### Install Dependencies
#### Method 1: Conda (Manual)
1. Create Conda environment
```bash
conda create -n crowdsurfer python=3.10 -y
conda activate crowdsurfer
```
2. Install Dependencies
```bash
pip install "open3d>=0.19.0"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install "jax[cuda12]"
pip install "hydra-core>=1.3.2"
pip install "accelerate>=1.6.0"
pip install rosbag --extra-index-url https://rospypi.github.io/simple/
```

#### Method 2: UV (Automatic)
1. Install UV from [here](https://docs.astral.sh/uv/getting-started/installation/) or by running:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Setup environment and sync dependencies
```bash
uv venv --python 3.10 --system-site-packages src/CrowdSurfer/.venv
source src/CrowdSurfer/.venv/bin/activate
uv sync --active
```

Alternatively, using the same conda environment as above, run:
```bash
export VIRTUAL_ENV=$CONDA_PREFIX
uv sync --active
```

### Set up the simulation (ROS1 Noetic)

Running the simulation in Gazebo requires pedsim_ros (to simulate the humans).
To install pedsim_ros and other dependencies, proceed as follows:
The default version is ROS Noetic.

#### Install Simulation Dependencies
```bash
git clone https://github.com/TempleRAIL/robot_gazebo.git
git clone https://github.com/Smart-Wheelchair-RRC/pedsim_ros_with_gazebo.git
wget https://raw.githubusercontent.com/zzuxzt/turtlebot2_noetic_packages/master/turtlebot2_noetic_install.sh
sudo sh turtlebot2_noetic_install.sh
```

### Build the  workspace
```bash
cd ~/crowdsurfer_ws
rosdep install --from-paths src --ignore-src -r -y
catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3
source devel/setup.bash
```

### Download the checkpoints
#### To the Default Path
```
cd src/CrowdSurfer/src/CrowdSurfer && mkdir checkpoints
pip install gdown
gdown https://drive.google.com/drive/folders/1HSRrbuwwNk9_C1WKN9qnStjemFLukO8s -O checkpoints --folder
cd ~/crowdsurfer_ws
```

#### To a Custom Path
Download checkpoints from [the drive link](https://drive.google.com/drive/folders/1HSRrbuwwNk9_C1WKN9qnStjemFLukO8s)

Replace the checkpoint paths for VQVAE and PixelCNN in the [configuration file](./src/CrowdSurfer/configuration/configuration.yaml)

## Running the Demo

From within crowdsurfer_ws/ in tmux, run:

```bash
bash src/CrowdSurfer/src/CrowdSurfer/run_CrowdSurfer.sh
```



## If you liked our work, please consider citing us -
**Bibtex** -
```
@misc{kumar2025crowdsurfersamplingoptimizationaugmented,
      title={CrowdSurfer: Sampling Optimization Augmented with Vector-Quantized Variational AutoEncoder for Dense Crowd Navigation}, 
      author={Naman Kumar and Antareep Singha and Laksh Nanwani and Dhruv Potdar and Tarun R and Fatemeh Rastgar and Simon Idoko and Arun Kumar Singh and K. Madhava Krishna},
      year={2025},
      eprint={2409.16011},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2409.16011}, 
}
```
