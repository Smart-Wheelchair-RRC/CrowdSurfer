"""
Read bags from data/[dataset_name]/bags and process them
into folders of images and observations in data/[dataset_name]
"""

import os
from functools import partial
from typing import Callable, Dict, Generator, List, Tuple, cast

import numpy as np
import rosbag
from genpy import Message, Time
from tqdm import tqdm

from . import message_reading_functions, processing_functions
from .constants import ExtraDataKeys


class Dataset:
    def __init__(
        self,
        name: str,
        directory: str,
        topic_to_function_mapping: Dict[
            str, Callable[[Message], Dict[str, np.ndarray]]
        ],
        post_processing_functions: List[
            Callable[[Dict[str, np.ndarray]], Dict[str, np.ndarray]]
        ] = [],
        editing_post_processing_functions: List[
            Callable[[Dict[str, np.ndarray]], Dict[str, np.ndarray]]
        ] = [],
        nested: bool = False,
    ):
        self.name = name
        self.directory = directory
        self.topic_to_function_mapping = topic_to_function_mapping
        self.post_processing_functions = post_processing_functions
        self.editing_post_processing_functions = editing_post_processing_functions
        self.nested = nested

        self.topics = [
            topic
            for topic in self.topic_to_function_mapping.keys()
            if topic is not None
        ]

        self.processed_directory = os.path.join(self.directory, "processed")
        if not os.path.exists(self.processed_directory):
            os.mkdir(self.processed_directory)

        self.index_file = os.path.join(self.processed_directory, "index.txt")

    def get_data_arrays(
        self,
        bag: rosbag.Bag,
        sample_rate: float = 4.0,
    ) -> Tuple[Dict[str, np.ndarray], int]:
        if len(self.topics) == 0:
            raise ValueError("No topics to process")

        topics_in_bag = bag.get_type_and_topic_info().topics.keys()
        assert all(topic in topics_in_bag for topic in self.topics), (
            f"Topics not found in bag: {self.topics - topics_in_bag}"
        )

        messages: Generator[Tuple[str, Message, Time], None, None] = bag.read_messages(
            topics=self.topics
        )

        data_array_mapping: Dict[str, List[np.ndarray]] = {ExtraDataKeys.TIMESTAMP: []}

        # Sync messages
        current_data_mapping: Dict[str, Dict[str, np.ndarray]] = {
            topic: None for topic in self.topics
        }
        current_time = cast(float, bag.get_start_time())

        for topic, message, time in messages:
            assert topic in self.topics, f"Unexpected topic: {topic}"

            current_data_mapping[topic] = message

            if (time.to_sec() - current_time) >= (1.0 / sample_rate):
                if all(
                    current_data_mapping[topic] is not None for topic in self.topics
                ):
                    data_array_mapping[ExtraDataKeys.TIMESTAMP].append(current_time)
                    for topic in self.topics:
                        # Splitting the Dynamic Obstacle Position and Body Pose from obstacle_metadata_function(dynamic_obstacle_topic)
                        processed_data = self.topic_to_function_mapping[topic](
                            current_data_mapping[topic]
                        )

                        for data_type in processed_data.keys():
                            if data_type not in data_array_mapping:
                                data_array_mapping[data_type] = []
                            data_array_mapping[data_type].append(
                                processed_data[data_type]
                            )

                current_time = time.to_sec()

        output_array_mapping = {
            data_type: np.array(data_array_mapping[data_type])
            for data_type in data_array_mapping.keys()
        }

        data_length = len(data_array_mapping[list(output_array_mapping.keys())[0]])

        for post_processing_function in self.post_processing_functions:
            output_array_mapping = post_processing_function(output_array_mapping)

        return output_array_mapping, data_length

    def save_array(self, array: np.ndarray, filename: str):
        np.save(os.path.join(self.directory, filename + ".npy"), array)

    def save_arrays(self, arrays: Dict[str, np.ndarray], filename: str):
        # Save to processed directory
        array_directory = os.path.join(self.processed_directory, filename)
        if not os.path.exists(array_directory):
            os.mkdir(array_directory)

        for array_name, array in arrays.items():
            np.save(os.path.join(array_directory, f"{array_name}.npy"), array)

    def update_index(self, formatted_bag_index: str, bag_name: str, data_length: int):
        with open(self.index_file, "a") as f:
            f.write(f"{formatted_bag_index} {bag_name} {data_length}\n")

    def process_bags(self, resume=False):
        print(f"Processing dataset {self.name} with topics: {self.topics}")

        bag_directory = os.path.join(self.directory, "bags")

        for bagname in os.listdir(bag_directory):
            if bagname.endswith(".bag"):
                break
            elif os.path.isdir(os.path.join(bag_directory, bagname)):
                for nested_bagname in os.listdir(os.path.join(bag_directory, bagname)):
                    if nested_bagname.endswith(".bag"):
                        break
                else:
                    continue
                break
        else:
            raise FileNotFoundError(f"No bag files found in {bag_directory}")

        if self.nested:
            bag_directories = os.listdir(bag_directory)
            bag_names = [
                os.path.join(directory, bag_name)
                for directory in bag_directories
                if os.path.isdir(os.path.join(bag_directory, directory))
                for bag_name in os.listdir(os.path.join(bag_directory, directory))
            ]
        else:
            bag_names = os.listdir(bag_directory)

        if resume:
            assert os.path.exists(self.index_file), "Index file does not exist"
            with open(self.index_file, "r") as f:
                lines = f.readlines()
                processed_bag_names = [line.split()[1] for line in lines]
                add_to_index = int(lines[-1].split()[0]) + 1
        else:
            processed_bag_names = []
            add_to_index = 0

        bag_names = [
            bag_name
            for bag_name in bag_names
            if bag_name not in processed_bag_names and bag_name.endswith(".bag")
        ]

        progress_bar = tqdm(
            total=len(bag_names) + len(processed_bag_names),
            desc="Processing",
            unit="bag",
            initial=add_to_index,
        )

        subtract_from_index = 0
        for bag_index, bag_name in enumerate(bag_names):
            if not bag_name.endswith(".bag"):
                progress_bar.total -= 1
                subtract_from_index += 1
                continue

            progress_bar.set_description(f"Processing {bag_name}")

            bag_index = bag_index - subtract_from_index + add_to_index
            bag_index = f"{bag_index:08d}"
            bag = rosbag.Bag(os.path.join(bag_directory, bag_name))

            # Save odometry data
            data_map, data_length = self.get_data_arrays(bag)

            self.save_arrays(data_map, bag_index)

            self.update_index(bag_index, bag_name, data_length)

            bag.close()
            progress_bar.update(1)

        progress_bar.set_description("Done processing")

    def edit_index_data_length(self, formatted_bag_index: str, data_length: int):
        with open(self.index_file, "r") as f:
            lines = f.readlines()

        with open(self.index_file, "w") as f:
            for line in lines:
                if line.split()[0] == formatted_bag_index:
                    f.write(f"{formatted_bag_index} {line.split()[1]} {data_length}\n")
                else:
                    f.write(line)

    def edit_processed_files(self):
        assert os.path.exists(self.index_file), "Index file does not exist"

        # Make a copy of the index file
        with open(self.index_file, "r") as f:
            lines = f.readlines()

        with open(self.index_file + ".bak", "w") as f:
            for line in lines:
                f.write(line)

        processed_bag_names = os.listdir(self.processed_directory)

        progress_bar = tqdm(
            total=len(processed_bag_names),
            desc="Editing",
            unit="processed bag",
        )

        for processed_bag_name in processed_bag_names:
            processed_bag_path = os.path.join(
                self.processed_directory, processed_bag_name
            )
            if not os.path.isdir(processed_bag_path):
                progress_bar.total -= 1
                continue

            progress_bar.set_description(f"Editing {processed_bag_name}")

            array_names = os.listdir(processed_bag_path)
            data_map = {
                array_name[:-4]: np.load(os.path.join(processed_bag_path, array_name))
                for array_name in array_names
            }

            for (
                editing_post_processing_function
            ) in self.editing_post_processing_functions:
                data_map = editing_post_processing_function(data_map)

            self.save_arrays(data_map, processed_bag_name)

            self.edit_index_data_length(
                processed_bag_name, len(data_map[ExtraDataKeys.TIMESTAMP])
            )

            progress_bar.update(1)

        progress_bar.set_description("Done editing")


# def process_huron_bags(
#     directory: str,
#     resume: bool = False,
#     nested: bool = True,
#     trajectory_timesteps: int = 50,
#     trajectory_total_time: int = 5,
#     goal_increment: int = 49,
# ):
#     from navigation.priest_guidance import PriestPlanner

#     priest_planner = PriestPlanner(
#         num_dynamic_obstacles=10,
#         num_obstacles=1950,
#         t_fin=trajectory_timesteps,
#         num=trajectory_total_time,
#         num_waypoints=goal_increment + 1,
#     )

#     dataset = Dataset(
#         name="huron",
#         directory=directory,
#         topic_to_function_mapping={
#             "/odometry": message_reading_functions.process_odometry_message,
#             "/laserscan": partial(message_reading_functions.process_laser_scan_message, maximum_points=1950),
#             "/pedestrians_pose": partial(message_reading_functions.process_pedestrian_pose_message, max_obstacles=10),
#         },
#         post_processing_functions=[
#             processing_functions.replace_odometry_heading_with_calculated_heading,
#             processing_functions.create_ego_velocities_from_odometry,
#             partial(
#                 processing_functions.create_goal_positions_and_expert_trajectory_from_odometry,
#                 goal_increment=goal_increment,
#             ),
#             processing_functions.create_obstacle_velocities_from_positions,
#             partial(processing_functions.convert_laser_scan_to_point_cloud, max_points=1950),
#             processing_functions.create_occupancy_map_from_point_cloud,
#             partial(
#                 processing_functions.create_priest_trajectories_from_observations,
#                 planner=priest_planner,
#             ),
#             partial(processing_functions.delete_from_data, key="laser_scan"),
#         ],
#         editing_post_processing_functions=[processing_functions.replace_close_to_goal_with_linear_coefficients],
#         nested=nested,
#     )
#     dataset.process_bags(resume=resume)


# def process_scand_bags(directory: str, resume: bool = False, nested: bool = False):
#     dataset = Dataset(
#         name="scand",
#         directory=directory,
#         topic_to_function_mapping={
#             "/jackal_velocity_controller/odom": message_reading_functions.process_odometry_message
#         },
#         nested=nested,
#     )
#     dataset.process_bags(resume=resume)


def process_custom_bags(
    directory: str,
    resume: bool = False,
    nested: bool = False,
    trajectory_timesteps: int = 50,
    trajectory_total_time: int = 5,
    goal_increment: int = 49,
    weight_track: float = 0.1,
    weight_smoothness: float = 0.2,
):
    from navigation.priest_guidance import PriestPlanner

    priest_planner = PriestPlanner(
        num_dynamic_obstacles=10,
        num_static_obstacles=1950,
        time_horizon=trajectory_total_time,
        trajectory_length=trajectory_timesteps,
        num_waypoints=goal_increment + 1,
        tracking_weight=weight_track,
        smoothness_weight=weight_smoothness,
    )

    expert_priest_planner = PriestPlanner(
        num_dynamic_obstacles=10,
        num_static_obstacles=1950,
        time_horizon=trajectory_total_time,
        trajectory_length=trajectory_timesteps,
        num_waypoints=goal_increment + 1,
        tracking_weight=2.0,
        smoothness_weight=weight_smoothness,
        max_outer_iterations=6,
    )

    dataset = Dataset(
        name="custom",
        directory=directory,
        topic_to_function_mapping={
            "/odom": message_reading_functions.process_odometry_message,
            "/scan": partial(
                message_reading_functions.process_laser_scan_message,
                maximum_points=1950,
            ),
            "/marker": partial(
                message_reading_functions.process_pedestrian_pose_message,
                max_obstacles=50,
            ),
        },
        post_processing_functions=[
            processing_functions.replace_odometry_heading_with_calculated_heading,
            processing_functions.create_ego_velocities_from_odometry,
            partial(
                processing_functions.create_goal_positions_and_expert_trajectory_from_odometry,
                goal_increment=goal_increment,
            ),
            processing_functions.create_obstacle_velocities_from_positions,
            partial(
                processing_functions.convert_laser_scan_to_point_cloud,
                maximum_points=1950,
            ),
            processing_functions.create_occupancy_map_from_point_cloud,
            partial(
                processing_functions.create_priest_trajectories_from_observations,
                planner=priest_planner,
            ),
            partial(
                processing_functions.delete_from_data, key=ExtraDataKeys.LASER_SCAN
            ),
        ],
        editing_post_processing_functions=[
            partial(
                processing_functions.expert_trajectory_to_bernstein_priest,
                planner=expert_priest_planner,
            ),
            processing_functions.downsample_point_cloud,
            processing_functions.create_ray_traced_occupancy_map_from_point_cloud,
        ],
        nested=nested,
    )
    dataset.process_bags(resume=resume)
    dataset.edit_processed_files()
