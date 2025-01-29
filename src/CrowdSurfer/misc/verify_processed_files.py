import os

import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))

dataset_dir = os.path.join(current_dir, "data", "huron")

npz_file = [file for file in os.listdir(dataset_dir) if file.endswith(".npz")]


# Load all npz files
for file in range(len(npz_file)):
    data = np.load(os.path.join(dataset_dir, npz_file[file]))
    for key in data.keys():
        print(key, data[key].shape)
    break
