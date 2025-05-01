import os

from processing import process_custom_bags

current_dir = os.path.dirname(os.path.abspath(__file__))

process_custom_bags(
    directory=os.path.join(current_dir, "data/custom/"),
    resume=False,
    nested=False,
    trajectory_timesteps=50,
    trajectory_total_time=5,
    goal_increment=99,
    weight_track=1.5,
)
