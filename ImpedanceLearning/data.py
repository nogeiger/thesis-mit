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
    Computes min and max values per axis for each key in the dataset.
    Ignores sentinel values (-9999.0) for optional fields.

    Args:
        data (list): A list of sample dictionaries with trajectory and sensor data.

    Returns:
        dict: A dictionary with min and max tensors per key.
    """
    stats = {}
    epsilon = 1e-8
    sentinel = -9999.0

    # Collect all unique keys across all samples
    all_keys = set()
    for sample in data:
        all_keys.update(sample.keys())

    for key in all_keys:
        if key.startswith("q"):  # Quaternions don't need stats
            continue

        try:
            # Stack all data for this key, ignoring samples that are missing it
            values = [sample[key] for sample in data if key in sample]
            stacked = np.concatenate(values, axis=0).astype(np.float32)

            # Flatten if matrix-shaped (e.g., lambda)
            if stacked.ndim > 2:
                stacked = stacked.reshape(stacked.shape[0], -1)

            # Mask out sentinel values (-9999)
            mask = ~(stacked == sentinel).any(axis=-1)
            valid_data = stacked[mask]

            if valid_data.size == 0:
                print(f" No valid data found for '{key}' (all entries were -9999). Skipping.")
                continue

            min_val = torch.tensor(np.min(valid_data, axis=0), dtype=torch.float32)
            max_val = torch.tensor(np.max(valid_data, axis=0), dtype=torch.float32)

            # Avoid divide-by-zero
            max_val = torch.where(max_val == min_val, max_val + epsilon, max_val)

            stats[f"min_{key}"] = min_val
            stats[f"max_{key}"] = max_val

        except Exception as e:
            print(f"Error computing stats for '{key}': {e}")

    return stats

def normalize_data_per_axis(data, stats):
    """
    Normalizes all valid fields in the dataset using per-axis min-max normalization.
    Skips quaternion fields. Ignores sentinel values (-9999.0) during normalization.

    Args:
        data (list): List of dictionaries with raw data.
        stats (dict): Dictionary of min/max values per key.

    Returns:
        list: List of dictionaries with normalized values.
    """
    normalized_data = []
    sentinel = -9999.0

    for sample in data:
        norm_sample = {}

        for key, value in sample.items():
            if key.startswith("q"):  # Skip quaternions
                norm_sample[key] = value
                continue

            min_key = f"min_{key}"
            max_key = f"max_{key}"

            if min_key not in stats or max_key not in stats:
                norm_sample[key] = value  # No stats → return as-is
                continue

            tensor_val = torch.tensor(value, dtype=torch.float32)

            # Create mask for valid (non-sentinel) entries
            if tensor_val.ndim == 2:
                mask = ~torch.any(tensor_val == sentinel, dim=1)
            elif tensor_val.ndim == 3:
                mask = ~torch.any((tensor_val == sentinel).view(tensor_val.shape[0], -1), dim=1)
            else:
                norm_sample[key] = value  # Unsupported shape
                continue

            # Flatten if needed (e.g., [T, 3, 3] → [T, 9])
            reshaped = False
            if tensor_val.ndim == 3:
                original_shape = tensor_val.shape
                tensor_val = tensor_val.view(tensor_val.shape[0], -1)
                reshaped = True

            min_val = stats[min_key].view(-1)
            max_val = stats[max_key].view(-1)

            range_val = max_val - min_val
            is_constant = range_val == 0
            range_val = torch.where(is_constant, torch.ones_like(range_val), range_val)

            # Normalize valid entries only
            norm_val = (tensor_val - min_val) / range_val
            for axis in range(norm_val.shape[-1]):
                if is_constant[axis].item():
                    norm_val[:, axis] = 0.5

            # Re-apply sentinel values to rows that were originally invalid
            norm_val[~mask] = sentinel

            # Reshape back if needed
            if reshaped:
                norm_val = norm_val.view(original_shape)

            # Safety check
            if torch.any(torch.isnan(norm_val)) or torch.any(torch.isinf(norm_val)):
                print(f" NaN/Inf found in normalized field '{key}'")

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
    Loads trajectory and sensor data from text files in a folder.

    Args:
        folder_path (str): Path to the folder containing input text files.
        seq_length (int): Length of each sequence segment.
        use_overlap (bool): Use overlapping windows or not.

    Returns:
        list: List of sample dictionaries with consistent keys and -9999.0 for missing fields.
    """
    all_data = []
    stride = 10 if use_overlap else seq_length  # Overlap control

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            filepath = os.path.join(folder_path, filename)
            try:
                df = pd.read_csv(filepath, sep="\t", skiprows=2, header=None, dtype=str)
                df = df.apply(pd.to_numeric, errors="coerce")
                df.dropna(inplace=True)

                base_columns = [
                    "time", "f_x", "f_y", "f_z",
                    "m_x", "m_y", "m_z",
                    "x", "y", "z",
                    "x0", "y0", "z0",
                    "u_x", "u_y", "u_z", "theta",
                    "u0_x", "u0_y", "u0_z", "theta0"
                ]

                # Assign column names
                num_cols = df.shape[1]
                optional_columns = [
                    "dx", "dy", "dz",
                    "omega_x", "omega_y", "omega_z"
                ] + [f"lambda_{r}{c}" for r in range(1, 4) for c in range(1, 4)] \
                  + [f"lambda_w_{r}{c}" for r in range(1, 4) for c in range(1, 4)]
                
                all_columns = base_columns + optional_columns
                if num_cols > len(all_columns):
                    raise ValueError(f"Too many columns in {filename}. Expected up to {len(all_columns)}, got {num_cols}")
                df.columns = all_columns[:num_cols]

                if len(df) < seq_length:
                    print(f"Skipping {filename}: not enough rows ({len(df)}). Needed: {seq_length}")
                    continue

                file_data = []

                for i in range(0, len(df) - seq_length + 1, stride):
                    # Clean & noisy position
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

                    # Axis-angle to quaternion
                    clean_axis = np.stack([
                        df["u0_x"].iloc[i:i + seq_length].values,
                        df["u0_y"].iloc[i:i + seq_length].values,
                        df["u0_z"].iloc[i:i + seq_length].values
                    ], axis=-1)

                    noisy_axis = np.stack([
                        df["u_x"].iloc[i:i + seq_length].values,
                        df["u_y"].iloc[i:i + seq_length].values,
                        df["u_z"].iloc[i:i + seq_length].values
                    ], axis=-1)

                    clean_angle = np.stack([df["theta0"].iloc[i:i + seq_length].values], axis=-1)
                    noisy_angle = np.stack([df["theta"].iloc[i:i + seq_length].values], axis=-1)

                    clean_quaternion = axis_angle_to_quaternion(clean_axis, clean_angle)
                    noisy_quaternion = axis_angle_to_quaternion(noisy_axis, noisy_angle)

                    if not is_unit_quaternion(clean_quaternion):
                        print(f"Warning: non-unit clean quaternion in {filename}")
                    if not is_unit_quaternion(noisy_quaternion):
                        print(f"Warning: non-unit noisy quaternion in {filename}")

                    # Force & moment
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

                    # Handle optional fields with -9999.0 defaults

                    # delta_pos
                    if {"dx", "dy", "dz"}.issubset(df.columns):
                        delta_pos = np.stack([
                            df["dx"].iloc[i:i + seq_length].values,
                            df["dy"].iloc[i:i + seq_length].values,
                            df["dz"].iloc[i:i + seq_length].values
                        ], axis=-1)
                    else:
                        delta_pos = np.full((seq_length, 3), -9999.0, dtype=np.float32)

                    # omega
                    if {"omega_x", "omega_y", "omega_z"}.issubset(df.columns):
                        omega = np.stack([
                            df["omega_x"].iloc[i:i + seq_length].values,
                            df["omega_y"].iloc[i:i + seq_length].values,
                            df["omega_z"].iloc[i:i + seq_length].values
                        ], axis=-1)
                    else:
                        omega = np.full((seq_length, 3), -9999.0, dtype=np.float32)

                    # lambda
                    lambda_cols = [f"lambda_{r}{c}" for r in range(1, 4) for c in range(1, 4)]
                    if set(lambda_cols).issubset(df.columns):
                        lambda_matrix = np.stack([df[col].iloc[i:i + seq_length].values for col in lambda_cols], axis=-1)
                        lambda_matrix = lambda_matrix.reshape(seq_length, 3, 3)
                    else:
                        lambda_matrix = np.full((seq_length, 3, 3), -9999.0, dtype=np.float32)

                    # lambda_w
                    lambda_w_cols = [f"lambda_w_{r}{c}" for r in range(1, 4) for c in range(1, 4)]
                    if set(lambda_w_cols).issubset(df.columns):
                        lambda_w_matrix = np.stack([df[col].iloc[i:i + seq_length].values for col in lambda_w_cols], axis=-1)
                        lambda_w_matrix = lambda_w_matrix.reshape(seq_length, 3, 3)
                    else:
                        lambda_w_matrix = np.full((seq_length, 3, 3), -9999.0, dtype=np.float32)

                    sample = {
                        "pos_0": clean_trajectory,
                        "pos": noisy_trajectory,
                        "q_0": clean_quaternion,
                        "q": noisy_quaternion,
                        "force": forces,
                        "moment": moments,
                        "delta_pos": delta_pos,
                        "omega": omega,
                        "lambda": lambda_matrix,
                        "lambda_w": lambda_w_matrix
                    }

                    file_data.append(sample)

                print(f"Loaded {len(file_data)} samples from {filename}, each with sequence length {seq_length}")
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
        self.sentinel = -9999.0
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
        delta_pos = torch.tensor(sample["delta_pos"], dtype=torch.float32)
        omega = torch.tensor(sample["omega"], dtype=torch.float32)
        lambda_matrix = torch.tensor(sample["lambda"], dtype=torch.float32)
        lambda_w_matrix = torch.tensor(sample["lambda_w"], dtype=torch.float32)

        return pos_0, pos, q_0, q, force, moment, delta_pos, omega, lambda_matrix, lambda_w_matrix

    def denormalize(self, normalized_data, trajectory_type="pos_0"):
        """
        Denormalizes the given data using stored statistics for each axis,
        skipping values where sentinel (-9999.0) is present.

        Args:
            normalized_data (torch.Tensor): Normalized data to be denormalized.
            trajectory_type (str): Field name to use for stats.

        Returns:
            torch.Tensor: Denormalized data.
        """
    
        if not self.stats or trajectory_type.startswith("q"):
            return normalized_data

        min_key = f"min_{trajectory_type}"
        max_key = f"max_{trajectory_type}"
        
        if min_key not in self.stats or max_key not in self.stats:
            return normalized_data

        min_vals = self.stats[min_key]
        max_vals = self.stats[max_key]
        range_vals = max_vals - min_vals
        range_vals = torch.where(range_vals == 0, torch.ones_like(range_vals), range_vals)
        
        denorm = normalized_data.clone()
        

        reshaped = False
        if denorm.ndim == 4 and denorm.shape[-2:] == (3, 3):  # Only reshape if it's a matrix like lambda
            original_shape = denorm.shape
            denorm = denorm.view(denorm.shape[0], denorm.shape[1], -1)  # [B, T, 9]
            reshaped = True

        # Get original shape
        original_shape = denorm.shape  # [B, T, D]
        B, T, D = denorm.shape

        # Flatten for safe row-wise masking
        denorm = denorm.view(-1, D)  # [B*T, D]

        # Mask where no -9999 is present in a row - dont want to normalize that
        mask = ~(denorm == self.sentinel).any(dim=1)  # [B*T]

        # Apply only where valid
        denorm[mask] = denorm[mask] * range_vals + min_vals

        # Restore shape
        denorm = denorm.view(original_shape)

        if reshaped:
            denorm = denorm.view(original_shape)

        return denorm