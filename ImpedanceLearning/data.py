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
    Computes per-axis min and max statistics for all array-like fields in the dataset.
    It dynamically detects all keys present across all samples (not just the first one).

    Args:
        data (list): A list of dictionaries, each sample containing various data arrays.

    Returns:
        dict: A dictionary with min and max values for each key in the data.
    """
    stats = {}
    epsilon = 1e-8

    if not data:
        return stats

    # Step 1: Collect all unique keys across all samples
    all_keys = set()
    for sample in data:
        all_keys.update(sample.keys())


    # Step 2: For each key, try to concatenate and compute min/max
    for key in all_keys:
        if key.startswith("q"):  # Skip quaternion stats
            continue

        try:
            # Collect valid arrays for this key
            stacked = np.concatenate(
                [sample[key] for sample in data if key in sample], axis=0
            ).astype(np.float32)

            min_vals = torch.tensor(np.min(stacked, axis=0), dtype=torch.float32)
            max_vals = torch.tensor(np.max(stacked, axis=0), dtype=torch.float32)

            # Avoid division by zero
            max_vals = torch.where(max_vals == min_vals, max_vals + epsilon, max_vals)

            stats[f"min_{key}"] = min_vals
            stats[f"max_{key}"] = max_vals

        except Exception as e:
            print(f"Skipping key '{key}' due to error during stats computation: {e}")
  
    return stats

import torch

def normalize_data_per_axis(data, stats):
    """
    Normalizes all numeric fields in the data using min-max stats.
    Handles vectors (e.g., pos, force) and matrices (e.g., lambda).
    Skips quaternion fields (q, q_0).
    
    Args:
        data (list): List of sample dictionaries with raw data.
        stats (dict): Dictionary of min/max per key from compute_statistics_per_axis().
    
    Returns:
        list: List of sample dictionaries with normalized data.
    """
    normalized_data = []

    for sample in data:
        norm_sample = {}

        for key, value in sample.items():
            if key.startswith("q"):  # Quaternions are already normalized
                norm_sample[key] = value
                continue

            min_key = f"min_{key}"
            max_key = f"max_{key}"

            if min_key not in stats or max_key not in stats:
                norm_sample[key] = value  # No stats available, keep original
                continue

            tensor_val = torch.tensor(value, dtype=torch.float32)
            min_val = stats[min_key]
            max_val = stats[max_key]

            # Reshape min/max and flatten value if it's a matrix (e.g., [seq_len, 3, 3])
            reshaped = False
            if tensor_val.ndim == 3:
                original_shape = tensor_val.shape
                tensor_val = tensor_val.view(tensor_val.shape[0], -1)
                min_val = min_val.view(-1)
                max_val = max_val.view(-1)
                reshaped = True

            # Compute range and handle constant dims
            range_val = max_val - min_val
            is_constant = range_val == 0
            range_val = torch.where(is_constant, torch.ones_like(range_val), range_val)

            # Normalize
            norm_val = (tensor_val - min_val) / range_val

            # Set constant dimensions to 0.5
            for axis in range(norm_val.shape[-1]):
                if is_constant[axis].item():
                    norm_val[:, axis] = 0.5

            # Reshape back if flattened earlier
            if reshaped:
                norm_val = norm_val.view(original_shape)

            # Sanity check
            if torch.any(torch.isnan(norm_val)) or torch.any(torch.isinf(norm_val)):
                print(f"Warning: NaN/Inf detected in normalized field '{key}'")

            norm_sample[key] = norm_val

        normalized_data.append(norm_sample)

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

def is_unit_quaternion(q, tolerance=5e-2):
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
    Handles both old and new formats with optional fields.

    Args:
        folder_path (str): Path to the folder containing the input text files.
        seq_length (int): Length of each trajectory segment.
        use_overlap (bool): Whether to use overlapping windows or not.

    Returns:
        list: A combined list of dictionaries, each containing trajectory and sensor data.
    """
    all_data = []
    stride = 10 if use_overlap else seq_length

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            filepath = os.path.join(folder_path, filename)
            try:
                df = pd.read_csv(filepath, sep="\t", skiprows=2, header=None, dtype=str)
                df = df.apply(pd.to_numeric, errors="coerce")
                df.dropna(inplace=True)

                # Assign expected base columns
                base_columns = [
                    "time", 
                    "f_x", "f_y", "f_z", 
                    "m_x", "m_y", "m_z", 
                    "x", "y", "z", 
                    "x0", "y0", "z0",
                    "u_x", "u_y", "u_z",
                    "theta",
                    "u0_x", "u0_y", "u0_z",
                    "theta0"
                ]

                # Optional columns (expected order if present)
                optional_columns = [
                    "dx", "dy", "dz",
                    "omega_x", "omega_y", "omega_z",
                    "lambda_11", "lambda_12", "lambda_13",
                    "lambda_21", "lambda_22", "lambda_23",
                    "lambda_31", "lambda_32", "lambda_33",
                    "lambda_w_11", "lambda_w_12", "lambda_w_13",
                    "lambda_w_21", "lambda_w_22", "lambda_w_23",
                    "lambda_w_31", "lambda_w_32", "lambda_w_33"
                ]

               # Total expected columns
                total_columns = base_columns + optional_columns

                # Assign column names for however many columns exist
                num_cols = df.shape[1]
                all_names = base_columns + optional_columns
                if num_cols > len(all_names):
                    raise ValueError(f"Too many columns in file: {filename}. Got {num_cols}, expected max {len(all_names)}")
                df.columns = all_names[:num_cols]


                if len(df) < seq_length:
                    print(f"Skipping file {filename} as it contains fewer rows ({len(df)}) than the required sequence length ({seq_length}).")
                    continue

                file_data = []

                for i in range(0, len(df) - seq_length + 1, stride):
                    clean_trajectory = np.stack([
                        df["x0"].iloc[i:i + seq_length].values,
                        df["y0"].iloc[i:i + seq_length].values,
                        df["z0"].iloc[i:i + seq_length].values
                    ], axis=-1)

                    noisy_trajectory = np.stack([
                        df["x"].iloc[i:i + seq_length].values,
                        df["y"].iloc[i:i + seq_length].values,
                        df["z"].iloc[i:i + seq_length].values
                    ], axis=-1)

                    clean_rotation_axis = np.stack([
                        df["u0_x"].iloc[i:i + seq_length].values,
                        df["u0_y"].iloc[i:i + seq_length].values,
                        df["u0_z"].iloc[i:i + seq_length].values
                    ], axis=-1)

                    noisy_rotation_axis = np.stack([
                        df["u_x"].iloc[i:i + seq_length].values,
                        df["u_y"].iloc[i:i + seq_length].values,
                        df["u_z"].iloc[i:i + seq_length].values
                    ], axis=-1)

                    clean_angle = np.stack([df["theta0"].iloc[i:i + seq_length].values], axis=-1)
                    noisy_angle = np.stack([df["theta"].iloc[i:i + seq_length].values], axis=-1)

                    clean_quaternion = axis_angle_to_quaternion(clean_rotation_axis, clean_angle)
                    noisy_quaternion = axis_angle_to_quaternion(noisy_rotation_axis, noisy_angle)

                    if not is_unit_quaternion(clean_quaternion):
                        print("non unit quaternion", clean_quaternion)
                        print(f"Warning: Non-unit quaternion detected in {filename} (clean).")
                    if not is_unit_quaternion(noisy_quaternion):
                        print("non unit quaternion", noisy_quaternion)
                        print(f"Warning: Non-unit quaternion detected in {filename} (noisy).")

                    forces = np.stack([
                        df["f_x"].iloc[i:i + seq_length].values,
                        df["f_y"].iloc[i:i + seq_length].values,
                        df["f_z"].iloc[i:i + seq_length].values
                    ], axis=-1)

                    moments = np.stack([
                        df["m_x"].iloc[i:i + seq_length].values,
                        df["m_y"].iloc[i:i + seq_length].values,
                        df["m_z"].iloc[i:i + seq_length].values
                    ], axis=-1)

                    # Optional fields
                    dx = None
                    angular_velocity = None
                    lambda_matrix = None
                    lambda_w_matrix = None
                   
                    if {"dx", "dy", "dz"}.issubset(df.columns):
                        
                        dx = np.stack([
                            df["dx"].iloc[i:i + seq_length].values,
                            df["dy"].iloc[i:i + seq_length].values,
                            df["dz"].iloc[i:i + seq_length].values
                        ], axis=-1)

                    if {"omega_x", "omega_y", "omega_z"}.issubset(df.columns):
                        angular_velocity = np.stack([
                            df["omega_x"].iloc[i:i + seq_length].values,
                            df["omega_y"].iloc[i:i + seq_length].values,
                            df["omega_z"].iloc[i:i + seq_length].values
                        ], axis=-1)

                    lambda_cols = [f"lambda_{r}{c}" for r in range(1, 4) for c in range(1, 4)]
                    if set(lambda_cols).issubset(df.columns):
                        lambda_matrix = np.stack([df[col].iloc[i:i + seq_length].values for col in lambda_cols], axis=-1)
                        lambda_matrix = lambda_matrix.reshape(seq_length, 3, 3)

                    lambda_w_cols = [f"lambda_w_{r}{c}" for r in range(1, 4) for c in range(1, 4)]
                    if set(lambda_w_cols).issubset(df.columns):
                        lambda_w_matrix = np.stack([df[col].iloc[i:i + seq_length].values for col in lambda_w_cols], axis=-1)
                        lambda_w_matrix = lambda_w_matrix.reshape(seq_length, 3, 3)

                    sample = {
                        "pos_0": clean_trajectory,
                        "pos": noisy_trajectory,
                        "q_0": clean_quaternion,
                        "q": noisy_quaternion,
                        "force": forces,
                        "moment": moments
                    }

                    if dx is not None:
                        sample["dx"] = dx
                    if angular_velocity is not None:
                        sample["omega"] = angular_velocity
                    if lambda_matrix is not None:
                        sample["lambda"] = lambda_matrix
                    if lambda_w_matrix is not None:
                        sample["lambda_w"] = lambda_w_matrix

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
        return pos_0, pos, q_0, q, force, moment

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
                return normalized_data  # Quaternions should remain unit-length
            elif trajectory_type == "q":
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