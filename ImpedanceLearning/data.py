import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



def compute_statistics_per_axis(data):
    """
    Computes min and max for each axis (x, y, z) for both clean and noisy trajectories in the dataset.

    Args:
        data (list): A list of dictionaries containing clean and noisy trajectories.

    Returns:
        dict: A dictionary containing min and max for each axis (x, y, z) for both clean and noisy trajectories.
    """
    # Concatenate positional data
    pos_0_data = np.concatenate([sample["pos_0"] for sample in data], axis=0).astype(np.float32)
    pos_data = np.concatenate([sample["pos"] for sample in data], axis=0).astype(np.float32)
    force_data = np.concatenate([sample["force"] for sample in data], axis=0).astype(np.float32)
    moment_data = np.concatenate([sample["moment"] for sample in data], axis=0).astype(np.float32)

    
    # Compute min/max for each component
    min_vals_pos_0 = torch.tensor(np.min(pos_0_data, axis=0), dtype=torch.float32)
    max_vals_pos_0 = torch.tensor(np.max(pos_0_data, axis=0), dtype=torch.float32)
    min_vals_pos = torch.tensor(np.min(pos_data, axis=0), dtype=torch.float32)
    max_vals_pos = torch.tensor(np.max(pos_data, axis=0), dtype=torch.float32)
    min_vals_force = torch.tensor(np.min(force_data, axis=0), dtype=torch.float32)
    max_vals_force = torch.tensor(np.max(force_data, axis=0), dtype=torch.float32)
    min_vals_moment = torch.tensor(np.min(moment_data, axis=0), dtype=torch.float32)
    max_vals_moment = torch.tensor(np.max(moment_data, axis=0), dtype=torch.float32)


    # Prevent division by zero for constant axes in both pos_0 and pos
    epsilon = 1e-8
    max_vals_pos_0 = torch.where(max_vals_pos_0 == min_vals_pos_0, max_vals_pos_0 + epsilon, max_vals_pos_0)
    max_vals_pos = torch.where(max_vals_pos == min_vals_pos, max_vals_pos + epsilon, max_vals_pos)
    max_vals_force = torch.where(max_vals_force == min_vals_force, max_vals_force + epsilon, max_vals_force)
    max_vals_moment = torch.where(max_vals_moment == min_vals_moment, max_vals_moment + epsilon, max_vals_moment)



    # Return the min and max for both pos_0 (clean) and pos (noisy), per axis (x, y, z)
    #Quaternions are already unit
    return {
        "min_pos_0": min_vals_pos_0, 
        "max_pos_0": max_vals_pos_0, 
        "min_pos": min_vals_pos, 
        "max_pos": max_vals_pos,
        "min_force": min_vals_force,
        "max_force": max_vals_force,
        "min_moment": min_vals_moment,
        "max_moment": max_vals_moment
    }


def normalize_data_per_axis(data, stats):
    """
    Normalizes each axis (x, y, z) in both clean (pos_0) and noisy (pos) trajectories, 
    as well as quaternions, forces, and moments.
    
    Args:
        data (list): A list of dictionaries containing clean and noisy trajectories, quaternions, and forces.
        stats (dict): Min and max values for normalization for both clean and noisy trajectories, quaternions, and forces.
    
    Returns:
        list: A list of dictionaries with normalized clean (pos_0), noisy (pos), quaternions, and forces.
    """
    normalized_data = []
    #Quaternions are already unit
    for sample in data:
        pos_0 = torch.tensor(sample["pos_0"], dtype=torch.float32)  # Clean trajectory [seq_length, 3]
        pos = torch.tensor(sample["pos"], dtype=torch.float32)  # Noisy trajectory [seq_length, 3]
        forces = torch.tensor(sample["force"], dtype=torch.float32)  # Forces [seq_length, 3]
        moments = torch.tensor(sample["moment"], dtype=torch.float32)  # Forces [seq_length, 3]

        # Retrieve min and max values for both clean and noisy trajectories and forces
        min_vals_pos_0, max_vals_pos_0 = stats["min_pos_0"], stats["max_pos_0"]
        min_vals_pos, max_vals_pos = stats["min_pos"], stats["max_pos"]
        min_vals_force, max_vals_force = stats["min_force"], stats["max_force"]
        min_vals_moment, max_vals_moment = stats["min_moment"], stats["max_moment"]
 

        # Normalize clean trajectory (pos_0)
        range_vals_pos_0 = max_vals_pos_0 - min_vals_pos_0
        is_constant_pos_0 = range_vals_pos_0 == 0  # Check for constant value per axis
        range_vals_pos_0 = torch.where(is_constant_pos_0, torch.ones_like(range_vals_pos_0), range_vals_pos_0)
        normalized_pos_0 = (pos_0 - min_vals_pos_0) / range_vals_pos_0

        # Normalize noisy trajectory (pos)
        range_vals_pos = max_vals_pos - min_vals_pos
        is_constant_pos = range_vals_pos == 0  # Check for constant value per axis
        range_vals_pos = torch.where(is_constant_pos, torch.ones_like(range_vals_pos), range_vals_pos)
        normalized_pos = (pos - min_vals_pos) / range_vals_pos

        # Normalize forces
        range_vals_force = max_vals_force - min_vals_force
        is_constant_force = range_vals_force == 0  # Check for constant value per axis
        range_vals_force = torch.where(is_constant_force, torch.ones_like(range_vals_force), range_vals_force)
        normalized_force = (forces - min_vals_force) / range_vals_force

        # Normalize moments
        range_vals_moment = max_vals_moment - min_vals_moment
        is_constant_moment = range_vals_moment == 0  # Check for constant value per axis
        range_vals_moment = torch.where(is_constant_moment, torch.ones_like(range_vals_moment), range_vals_moment)
        normalized_moment = (moments - min_vals_moment) / range_vals_moment

        # Assign fixed normalized value (e.g., 0.5) for constant axes
        for axis in range(pos_0.shape[-1]):  # Iterate over x, y, z
            if is_constant_pos_0[axis].item():
                normalized_pos_0[:, axis] = 0.5
            if is_constant_pos[axis].item():
                normalized_pos[:, axis] = 0.5
            if is_constant_force[axis].item():
                normalized_force[:, axis] = 0.5
            if is_constant_moment[axis].item():
                normalized_moment[:, axis] = 0.5

        # Debugging: Check for anomalies
        if torch.any(torch.isinf(normalized_pos_0)) or torch.any(torch.isnan(normalized_pos_0)):
            print("Error: Found inf/nan in normalized_pos_0:", normalized_pos_0)
        if torch.any(torch.isinf(normalized_pos)) or torch.any(torch.isnan(normalized_pos)):
            print("Error: Found inf/nan in normalized_pos:", normalized_pos)
        if torch.any(torch.isinf(normalized_force)) or torch.any(torch.isnan(normalized_force)):
            print("Error: Found inf/nan in normalized_force:", normalized_force)
        if torch.any(torch.isinf(normalized_moment)) or torch.any(torch.isnan(normalized_moment)):
            print("Error: Found inf/nan in normalized_moment:", normalized_moment)

        # Append normalized data (both pos_0, pos, and force)
        normalized_data.append({
            "pos_0": normalized_pos_0,
            "pos": normalized_pos,
            "force": normalized_force,
            "moment": normalized_moment
        })

    return normalized_data


def axis_angle_to_quaternion(axis, angle):
    """
    Converts an axis-angle representation to a quaternion.
    :param axis: [N, 3] array representing the rotation axis
    :param angle: [N, 1] array representing the rotation angle in radians
    :return: [N, 4] array representing the quaternion (w, x, y, z)
    """
    half_angle = angle / 2.0
    sin_half_angle = np.sin(half_angle)
    q_w = np.cos(half_angle)
    q_xyz = axis * sin_half_angle  # Element-wise multiplication
    return np.concatenate([q_w, q_xyz], axis=-1)  # Shape: [N, 4]

def is_unit_quaternion(q, tolerance=1e-2):
    """Checks if a quaternion is a unit quaternion within a given tolerance."""
    norms = np.linalg.norm(q, axis=1)
    return np.all(np.abs(norms - 1.0) < tolerance)

# Data Generation for synthetic data
def generate_data(num_samples=10000, seq_length=100):
    """
    Generates synthetic data consisting of clean trajectories based on sine and cosine functions.
    
    Args:
        num_samples (int): Number of trajectories to generate.
        seq_length (int): Length of each trajectory.

    Returns:
        list: A list of dictionaries, each containing the key "pos_0" for the clean trajectory.
    """
    data = []
    for _ in range(num_samples):
        t = np.linspace(0, 10, seq_length)
        clean_trajectory_x = np.sin(t) + np.cos(2 * t)
        clean_trajectory_y = np.sin(2 * t) + np.cos(t)
        clean_trajectory_z = np.sin(3 * t) + np.cos(3 * t)
        sample = {
            "pos_0": np.stack([clean_trajectory_x, clean_trajectory_y, clean_trajectory_z], axis=-1),  # Shape: [seq_length, 3]
        }
        data.append(sample)
    print(f"Generated {num_samples} samples, each with a sequence length of {seq_length} in 3D.")
    return data


def load_robot_data(folder_path, seq_length, use_overlap=True):
    """
    Loads real trajectory data from all text files in a folder and formats it like the generated data.

    Args:
        folder_path (str): Path to the folder containing the input text files.
        seq_length (int): Length of each trajectory segment.

    Returns:
        list: A combined list of dictionaries, each containing 'pos_0' with shape [seq_length, 3] and 'force' with shape [seq_length, 3].
    """
    all_data = []
     # Set stride based on flag
    stride = 10 if use_overlap else seq_length  # Overlapping = stride 10, Non-overlapping = full seq length

    # Iterate over all text files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):  # Process only text files
            filepath = os.path.join(folder_path, filename)
            try:
                # Load the data while skipping the header row
                df = pd.read_csv(filepath, sep="\t", skiprows=2, header=None, dtype=str)  # Read as strings first

                # Convert all columns to numeric, setting non-numeric values to NaN
                df = df.apply(pd.to_numeric, errors="coerce")

                # Drop any rows that contain NaN (i.e., rows with non-numeric data)
                df.dropna(inplace=True)

                # Rename columns after ensuring the data is clean
                df.columns = [    "time", 
                    "f_x", "f_y", "f_z", 
                    "m_x", "m_y", "m_z", 
                    "x", "y", "z", 
                    "x0", "y0", "z0",
                    "u_x", "u_y", "u_z",
                    "theta",
                    "u0_x", "u0_y", "u0_z",
                    "theta0",
                ]
                # Verify the total rows and sequence length
                if len(df) < seq_length:
                    print(f"Skipping file {filename} as it contains fewer rows ({len(df)}) than the required sequence length ({seq_length}).")
                    continue

                file_data = []  # Temporary storage for data from the current file

                # Extract clean trajectories for x, y, z and stack them
                #for i in range(0, len(df) - seq_length + 1, seq_length):
                for i in range(0, len(df) - seq_length + 1, stride):  #overlapping slicing with stride 10
                    clean_trajectory = np.stack([df["x0"].iloc[i:i + seq_length].values,
                                                 df["y0"].iloc[i:i + seq_length].values,
                                                 df["z0"].iloc[i:i + seq_length].values], axis=-1)
                    noisy_trajectory = np.stack([df["x"].iloc[i:i + seq_length].values,
                                                 df["y"].iloc[i:i + seq_length].values,
                                                 df["z"].iloc[i:i + seq_length].values], axis=-1)
                    clean_rotation_axis = np.stack([df["u0_x"].iloc[i:i + seq_length].values,
                                                    df["u0_y"].iloc[i:i + seq_length].values,
                                                    df["u0_z"].iloc[i:i + seq_length].values], axis=-1)
                    noisy_rotation_axis = np.stack([df["u_x"].iloc[i:i + seq_length].values,
                                                    df["u_y"].iloc[i:i + seq_length].values,
                                                    df["u_z"].iloc[i:i + seq_length].values], axis=-1)
                    clean_angle = np.stack([df["theta0"].iloc[i:i + seq_length].values], axis=-1)
                    noisy_angle = np.stack([df["theta"].iloc[i:i + seq_length].values], axis=-1)
                    clean_quaternion = axis_angle_to_quaternion(clean_rotation_axis, clean_angle)
                    noisy_quaternion = axis_angle_to_quaternion(noisy_rotation_axis, noisy_angle)

                    # Check if quaternions are unit quaternions
                    if not is_unit_quaternion(clean_quaternion):
                        print(f"Warning: Non-unit quaternion detected in {filename} (clean).")
                    if not is_unit_quaternion(noisy_quaternion):
                        print(f"Warning: Non-unit quaternion detected in {filename} (noisy).")


                    forces = np.stack([df["f_x"].iloc[i:i + seq_length].values,
                                       df["f_y"].iloc[i:i + seq_length].values,
                                       df["f_z"].iloc[i:i + seq_length].values], axis=-1)
                    moments = np.stack([df["m_x"].iloc[i:i + seq_length].values,
                                        df["m_y"].iloc[i:i + seq_length].values,
                                        df["m_z"].iloc[i:i + seq_length].values], axis=-1)
                    
                    sample = {
                        "pos_0": clean_trajectory,
                        "pos": noisy_trajectory,
                        "q_0": clean_quaternion,  # Replaces u_0 and theta_0
                        "q": noisy_quaternion,  # Replaces u and theta
                        "force": forces,
                        "moment": moments
                    }
                    file_data.append(sample)

                print(f"Loaded {len(file_data)} samples from {filename}, each with a sequence length of {seq_length}.")
                all_data.extend(file_data)

            except Exception as e:
                print(f"Error loading data from {filename}: {e}")

    print(f"Total loaded samples from all files: {len(all_data)}")
    return all_data





# Dataset Class for initial model (not used in main)
class ImpedanceDatasetInitial(Dataset):
    """
    A dataset class for timestamped trajectories with noisy inputs and clean outputs.
    Not used in the current script.
    """
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return (
            torch.tensor(sample["timestamp"], dtype=torch.float32),
            torch.tensor(sample["pos_0"], dtype=torch.float32),
            torch.tensor(sample["pos"], dtype=torch.float32),
            torch.tensor(sample["q_0"], dtype=torch.float32),
            torch.tensor(sample["q"], dtype=torch.float32),

            torch.tensor(sample["force"], dtype=torch.float32),
            torch.tensor(sample["moment"], dtype=torch.float32),
        )

class ImpedanceDatasetDiffusion(Dataset):
    """
    A dataset class for 3D diffusion-based training with per-axis normalization.

    Args:
        data (list): A list of dictionaries containing 3D clean trajectories.
        stats (dict): A dictionary containing min and max for each axis (x, y, z).

    Returns:    # For example, iterate over data
    for sample in data:
        clean_trajectory = sample["pos_0"]
        noisy_trajectory = sample["pos"]
        print("Clean trajectory shape:", clean_trajectory.shape)
        print("Noisy trajectory shape:", noisy_trajectory.shape)
        break
        torch.Tensor: Normalized clean 3D trajectory (pos_0).
    """
    def __init__(self, data, stats=None):
        self.data = data
        self.stats = stats

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        pos_0 = torch.tensor(sample["pos_0"], dtype=torch.float32) 
        pos = torch.tensor(sample["pos"], dtype=torch.float32) 
        q_0 = torch.tensor(sample["q_0"], dtype=torch.float32)
        q = torch.tensor(sample["q"], dtype=torch.float32)
        force = torch.tensor(sample["force"], dtype=torch.float32) 
        moment = torch.tensor(sample["moment"], dtype=torch.float32) 
        return pos_0, pos, q_0, q,force, moment

    def denormalize(self, normalized_data, trajectory_type="pos_0"):
        """
        Denormalizes the given data using stored statistics for each axis.

        Args:
            normalized_data (torch.Tensor): Normalized data to be denormalized.
            trajectory_type (str): Specifies whether to use 'pos_0' (clean) or 'pos' (noisy) statistics.

        Returns:
            torch.Tensor: Denormalized data.
        """

        if self.stats:
            if trajectory_type == "pos_0":  # Use clean trajectory statistics
                min_vals = self.stats["min_pos_0"]
                max_vals = self.stats["max_pos_0"]
            elif trajectory_type == "pos":  # Use noisy trajectory statistics
                min_vals = self.stats["min_pos"]
                max_vals = self.stats["max_pos"]
            elif trajectory_type == "q_0":
                if not np.all(is_unit_quaternion(normalized_data.numpy())):
                    print("Warning: Denormalized q_0 is not a unit quaternion.")
                return normalized_data  # Quaternions should remain unit-length
            elif trajectory_type == "q":
                if not np.all(is_unit_quaternion(normalized_data.numpy())):
                    print("Warning: Denormalized q is not a unit quaternion.")
                return normalized_data  # Quaternions should remain unit-length
            elif trajectory_type == "force":  # Use noisy trajectory statistics
                min_vals = self.stats["min_force"]
                max_vals = self.stats["max_force"]
            elif trajectory_type == "moment":  # Use noisy trajectory statistics
                min_vals = self.stats["min_moment"]
                max_vals = self.stats["max_moment"]
        
            else:
                raise ValueError("Invalid trajectory type. Must be 'pos_0' or 'pos'.")

            # Denormalize using the corresponding min and max
            denormalized_data = normalized_data * (max_vals - min_vals) + min_vals
            return denormalized_data
        else:
            return normalized_data