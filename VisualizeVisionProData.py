import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist

def parse_matrices_from_file(file_path):
    """Parse translation vectors from the streamed data text file."""
    positions = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for i, line in enumerate(lines):
            if "Right Wrist:" in line:
                try:
                    # Extract the matrix string
                    matrix_str = line.split("Right Wrist: ")[1].strip()
                    matrix = np.array(eval(matrix_str))  # Convert string to numpy array

                    # Check if the matrix is nested (3D array containing a single 4x4 matrix)
                    if matrix.ndim == 3 and matrix.shape[0] == 1 and matrix.shape[1:] == (4, 4):
                        matrix = matrix[0]  # Extract the 4x4 matrix

                    if matrix.shape == (4, 4):  # Validate it's a 4x4 matrix
                        translation = matrix[:3, 3]  # Extract translation vector (T_x, T_y, T_z)
                        positions.append(translation)
                    else:
                        print(f"Unexpected matrix shape at line {i + 1}: {matrix.shape}")
                except Exception as e:
                    print(f"Error parsing matrix at line {i + 1}: {line}\n{e}")
    return np.array(positions)

def visualize_positions(positions):
    """Visualize the positions in 3D space."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c='r', label='Positions')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    ax.set_title('Robot Wrist Positions in 3D')
    ax.legend()
    plt.show()

def calculate_max_distance(positions):
    """Calculate the maximum Euclidean distance between positions."""
    return np.max(pdist(positions))

file_path = "streamed_data.txt"

# Parse positions from the file
positions = parse_matrices_from_file(file_path)

if len(positions) == 0:
    print("No valid positions found in the file.")
else:

    # Compute and print the maximum Euclidean distance
    max_distance = calculate_max_distance(positions)
    print(f"Maximum Euclidean Distance: {max_distance:.4f}")
    # Visualize the positions in 3D
    visualize_positions(positions)


