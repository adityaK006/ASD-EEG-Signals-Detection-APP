import numpy as np
import os

# List of .npy files to convert
npy_files = [
    "normal_sample_1.npy",
    "normal_sample_2.npy",
    "normal_sample_3.npy",
    "autism_sample_1.npy",
    "autism_sample_2.npy",
    "autism_sample_3.npy"
]

# Convert each .npy file to .csv by flattening the array
for npy_file in npy_files:
    # Load the data
    data = np.load(npy_file)

    # Flatten the array (preserving order) and save to CSV
    csv_file = npy_file.replace(".npy", ".csv")
    np.savetxt(csv_file, data.reshape(-1), delimiter=",")

    print(f"Converted {npy_file} -> {csv_file}")