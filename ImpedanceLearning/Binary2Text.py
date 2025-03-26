import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

# Directory containing .bin files
binary_folder = "Data/RealDataGT"  

# Define the correct number of categories (columns)
column_count = 21 + 24  # We confirmed earlier

# Define the column names based on C++ variables
column_names = [
    "time", 
    "f_x", "f_y", "f_z", 
    "m_x", "m_y", "m_z", 
    "x", "y", "z", 
    "x0", "y0", "z0",
    "u_x", "u_y", "u_z",
    "theta",
    "u0_x", "u0_y", "u0_z",
    "theta0",
    "dx", "dy", "dz",
    "lambda_11", "lambda_12", "lambda_13", "lambda_21", "lambda_22", "lambda_23", "lambda_31", "lambda_32", "lambda_33",
    "omega_x", "omega_y", "omega_z",
    "lambda_w_11", "lambda_w_12", "lambda_w_13", "lambda_w_21", "lambda_w_22", "lambda_w_23", "lambda_w_31", "lambda_w_32", "lambda_w_33",
]

# Find all .bin files in the directory
bin_files = glob.glob(os.path.join(binary_folder, "*.bin"))

# Process each .bin file
for binary_file in bin_files:
    # Construct output .txt file name
    text_file = binary_file.replace(".bin", ".txt")

    # Read the binary file as double-precision floats (float64)
    data = np.fromfile(binary_file, dtype=np.float64)

    # Reshape data into the correct structure
    if len(data) % column_count != 0:
        print(f"Error: Data length in {binary_file} is not a multiple of {column_count}. Skipping this file.")
        continue

    data = data.reshape(-1, column_count)  # Reshape into rows with 13 columns

    # Convert to Pandas DataFrame with column names
    df = pd.DataFrame(data, columns=column_names)

    # Save to a readable text file (CSV format, tab-separated)
    df.to_csv(text_file, sep="\t", index=False)

    print(f"Converted {binary_file} to {text_file} successfully.")

    # 3D Trajectory Plot
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')

    # Plot the trajectory using x, y, z
    #ax.plot(df["x"], df["y"], df["z"], label=f"Trajectory ({os.path.basename(binary_file)})", color="b")

    # Labels
    #ax.set_xlabel("X Position")
    #ax.set_ylabel("Y Position")
    #ax.set_zlabel("Z Position")
    #ax.set_title(f"3D Trajectory Plot: {os.path.basename(binary_file)}")

    # Show legend
    #ax.legend()

    # Show plot
    #plt.show()

print("Processing completed for all .bin files.")
