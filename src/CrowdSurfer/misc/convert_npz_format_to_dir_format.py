import os

import numpy as np
from numpy.lib.npyio import NpzFile


def extract_npz_files(directory: str):
    index_file = os.path.join(directory, "index.txt")
    extract_directory = os.path.join(directory, "processed")
    # Make directory if it doesnt exist
    if not os.path.exists(extract_directory):
        os.mkdir(extract_directory)

    with open(index_file, "r") as f:
        for line in f:
            split_line = line.split()
            npz_file: NpzFile = np.load(os.path.join(directory, f"{split_line[0]}.npz"))
            with npz_file.zip as zip_file:
                zip_file.extractall(os.path.join(extract_directory, split_line[0]))

    # Move index file into the processed directory
    os.rename(index_file, os.path.join(extract_directory, "index.txt"))

    return extract_directory


current_dir = os.path.dirname(os.path.abspath(__file__))
extract_npz_files(os.path.join(current_dir, "data/custom"))
