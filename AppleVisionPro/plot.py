import matplotlib.pyplot as plt
import numpy as np
import re
from mpl_toolkits.mplot3d import Axes3D

# Define file path
file_path = "streamed_data.txt"

# Lists to store extracted data
timestamps = []
x_translations = []
y_translations = []
z_translations = []

# Read and parse the file
with open(file_path, 'r') as file:
    for line in file:
        # Extract timestamps
        if "Timestamp:" in line:
            timestamps.append(float(line.split(":")[1].strip()))
        # Extract translation components from the transformation matrix
        match = re.search(r"\[\[\[.*?, .*?, .*?, (.*?)\], \[.*?, .*?, .*?, (.*?)\], \[.*?, .*?, .*?, (.*?)\]", line)
        if match:
            x_translations.append(float(match.group(1)))
            y_translations.append(float(match.group(2)))
            z_translations.append(float(match.group(3)))

# Convert to numpy arrays for better handling
timestamps = np.array(timestamps)
x_translations = np.array(x_translations)
y_translations = np.array(y_translations)
z_translations = np.array(z_translations)

# Create a 3D plot
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot the trajectory
ax.plot(x_translations, y_translations, z_translations, label="Translation Path", marker="o")

# Labels and title
ax.set_xlabel("X Translation (m)")
ax.set_ylabel("Y Translation (m)")
ax.set_zlabel("Z Translation (m)")
ax.set_title("3D Translation Path Over Time")

# Show the plot
plt.legend()
plt.show()
