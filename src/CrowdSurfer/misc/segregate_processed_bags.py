import os
import shutil
from tqdm import tqdm

def segregate_npy_files(source_folder, destination_folder):
    categories = {
        "dynamic_obstacle_position": "_dynamic_obstacle_position",
        "dynamic_obstacle_velocity": "_dynamic_obstacle_velocity",
        "point_cloud": "_point_cloud",
        "odometry": "_odometry",
        "priest_trajectories": "_priest_trajectories",
        "timestamp": "_timestamp"
    }

    for category in categories.keys():
        os.makedirs(os.path.join(destination_folder, category), exist_ok=True)

    npy_files = [f for f in os.listdir(source_folder) if f.endswith('.npy')]
    
    for filename in tqdm(npy_files, desc="Processing files", unit="file"):
        source_path = os.path.join(source_folder, filename)
        
        for category, ending in categories.items():
            if filename.endswith(ending + '.npy'):
                destination_path = os.path.join(destination_folder, category, filename)
                
                shutil.move(source_path, destination_path)
                tqdm.write(f"Moved {filename} to {category} folder")
                break
        else:
            tqdm.write(f"No matching category found for {filename}")

if __name__ == "__main__":
    source_folder = "/home/wheelchair/huron"
    destination_folder = "/home/wheelchair/huron/processed_data"
    
    segregate_npy_files(source_folder, destination_folder)