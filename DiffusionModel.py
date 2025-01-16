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
    Computes min and max for each axis (x, y, z) in the dataset for normalization.

    Args:
        data (list): A list of dictionaries containing clean trajectories.

    Returns:
        dict: A dictionary containing min and max for each axis (x, y, z).
    """
    data_concat = np.concatenate([sample["pos_0"] for sample in data], axis=0)  # Shape: [total_points, 3]
    min_vals = torch.tensor(data_concat.min(axis=0), dtype=torch.float32)  # Shape: [3]
    max_vals = torch.tensor(data_concat.max(axis=0), dtype=torch.float32)  # Shape: [3]

    # Prevent division by zero
    epsilon = 1e-8
    max_vals = torch.where(max_vals == min_vals, max_vals + epsilon, max_vals)
    return {"min": min_vals, "max": max_vals}


def normalize_data_per_axis(data, stats):
    """
    Normalizes each axis (x, y, z) in the data.
    Constant axes are assigned a fixed normalized value (e.g., 0.5).

    Args:
        data (list): A list of dictionaries containing clean trajectories.
        stats (dict): Min and max values for normalization.

    Returns:
        list: A list of dictionaries with normalized trajectories.
    """
    normalized_data = []
    for sample in data:
        pos_0 = torch.tensor(sample["pos_0"], dtype=torch.float32)  # Shape: [seq_length, 3]
        min_vals, max_vals = stats["min"], stats["max"]

        # Handle constant axes
        range_vals = max_vals - min_vals
        is_constant = range_vals == 0

        # Avoid division by zero for non-constant axes
        range_vals = torch.where(is_constant, torch.ones_like(range_vals), range_vals)
        normalized_pos = (pos_0 - min_vals) / range_vals

        # Assign fixed normalized value (e.g., 0.5) for constant axes
        for axis in range(pos_0.shape[-1]):  # Iterate over x, y, z
            if is_constant[axis]:
                normalized_pos[:, axis] = 0.5  # Fixed normalized value for constant axes

        # Debugging: Check for anomalies
        if torch.any(torch.isinf(normalized_pos)) or torch.any(torch.isnan(normalized_pos)):
            print("Error: Found inf/nan in normalized_pos:", normalized_pos)

        normalized_data.append({"pos_0": normalized_pos})

    return normalized_data



def denormalize_data_with_constant_axes(normalized_data, stats):
    """
    Denormalizes each axis (x, y, z) in the data.
    Constant axes are restored to their original constant values.

    Args:
        normalized_data (list): Normalized trajectory data.
        stats (dict): Min and max values for denormalization.

    Returns:
        list: A list of dictionaries with denormalized trajectories.
    """
    denormalized_data = []
    for sample in normalized_data:
        pos_0 = sample["pos_0"]
        min_vals, max_vals = stats["min"], stats["max"]

        # Handle constant axes
        range_vals = max_vals - min_vals
        is_constant = range_vals == 0

        range_vals = torch.where(is_constant, torch.ones_like(range_vals), range_vals)
        denormalized_pos = pos_0 * range_vals + min_vals

        # Restore original constant values
        for axis in range(pos_0.shape[-1]):
            if is_constant[axis]:
                denormalized_pos[:, axis] = min_vals[axis]  # Restore original constant value

        denormalized_data.append({"pos_0": denormalized_pos})

    return denormalized_data


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


def load_robot_data(folder_path, seq_length):
    """
    Loads real trajectory data from all text files in a folder and formats it like the generated data.

    Args:
        folder_path (str): Path to the folder containing the input text files.
        seq_length (int): Length of each trajectory segment.

    Returns:
        list: A combined list of dictionaries, each containing 'pos_0' with shape [seq_length, 3].
    """
    all_data = []

    # Iterate over all text files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):  # Process only text files
            filepath = os.path.join(folder_path, filename)
            try:
                # Load the data while skipping the header row
                df = pd.read_csv(filepath, sep="\t", skiprows=1, header=None)
                df.columns = ["Time", "Pos_0_x", "Pos_0_y", "Pos_0_z", "Pos_x", "Pos_y", "Pos_z", "Force_x", "Force_y", "Force_z"]

                # Verify the total rows and sequence length
                if len(df) < seq_length:
                    print(f"Skipping file {filename} as it contains fewer rows ({len(df)}) than the required sequence length ({seq_length}).")
                    continue

                file_data = []  # Temporary storage for data from the current file

                # Extract clean trajectories for x, y, z and stack them
                for i in range(0, len(df) - seq_length + 1, seq_length):
                    clean_trajectory = np.stack([
                        df["Pos_0_x"].iloc[i:i + seq_length].values,
                        df["Pos_0_y"].iloc[i:i + seq_length].values,
                        df["Pos_0_z"].iloc[i:i + seq_length].values,
                    ], axis=-1)  # Shape: [seq_length, 3]

                    sample = {"pos_0": clean_trajectory}
                    file_data.append(sample)

                print(f"Loaded {len(file_data)} samples from {filename}, each with a sequence length of {seq_length} in 3D.")
                all_data.extend(file_data)

            except Exception as e:
                print(f"Error loading data from {filename}: {e}")

    print(f"Total loaded samples from all files: {len(all_data)}")
    return all_data


# Loss Function
def loss_function(predicted_noise, actual_noise):
    """
    Computes the mean squared error between the predicted and actual noise.

    Args:
        predicted_noise (torch.Tensor): Predicted noise added to the trajectory.
        actual_noise (torch.Tensor): Actual noise added to the trajectory.

    Returns:
        torch.Tensor: Mean squared error loss.
    """
    return nn.MSELoss()(predicted_noise, actual_noise)

# Add Noise Function for 3D Trajectories
def add_noise(clean_trajectory, noiseadding_steps):
    """
    Dynamically adds Gaussian noise to a clean 3D trajectory.

    Args:
        clean_trajectory (torch.Tensor): The clean trajectory with shape [seq_length, 3].
        noiseadding_steps (int): Number of steps to iteratively add noise.

    Returns:
        torch.Tensor: Noisy trajectory with shape [seq_length, 3].
    """
    noisy_trajectory = clean_trajectory.clone()
    for _ in range(noiseadding_steps):
        noise = torch.randn_like(clean_trajectory) * 0.1  # Scale the noise
        noisy_trajectory += noise
    return noisy_trajectory

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
            torch.tensor(sample["pos"], dtype=torch.float32),
            torch.tensor(sample["factual_noiseorce"], dtype=torch.float32),
            torch.tensor(sample["pos_0"], dtype=torch.float32),
        )

class ImpedanceDatasetDiffusion(Dataset):
    """
    A dataset class for 3D diffusion-based training with per-axis normalization.

    Args:
        data (list): A list of dictionaries containing 3D clean trajectories.
        stats (dict): A dictionary containing min and max for each axis (x, y, z).

    Returns:
        torch.Tensor: Normalized clean 3D trajectory (pos_0).
    """
    def __init__(self, data, stats=None):
        self.data = data
        self.stats = stats

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        pos_0 = torch.tensor(sample["pos_0"], dtype=torch.float32)  # Already normalized
        return pos_0

    def denormalize(self, normalized_data):
        """
        Denormalizes the given data using stored statistics for each axis.

        Args:
            normalized_data (torch.Tensor): Normalized data to be denormalized.

        Returns:
            torch.Tensor: Denormalized data.
        """
        if self.stats:
            return normalized_data * (self.stats["max"] - self.stats["min"]) + self.stats["min"]
        else:
            return normalized_data

class NoisePredictor(nn.Module):
    """
    A feedforward neural network to predict clean 3D trajectories from noisy inputs.
    """
    def __init__(self, seq_length, hidden_dim):
        super(NoisePredictor, self).__init__()
        input_dim = seq_length * 3  # Flatten 3D data into 1D vector
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layer_1 = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, input_dim)
        self.relu = nn.ReLU()

    def forward(self, noisy_trajectory):
        """
        Forward pass to predict clean 3D trajectory.

        Args:
            noisy_trajectory (torch.Tensor): Input noisy trajectory of shape [batch_size, seq_length, 3].

        Returns:
            torch.Tensor: Predicted clean trajectory of shape [batch_size, seq_length, 3].
        """
        batch_size, seq_length, _ = noisy_trajectory.shape
        x = noisy_trajectory.view(batch_size, -1)  # Flatten to [batch_size, seq_length * 3]
        x = self.input_layer(x)
        x = self.relu(x)
        x = self.hidden_layer_1(x)
        x = self.relu(x)
        x = self.hidden_layer_2(x)
        x = self.relu(x)
        x = self.output_layer(x)
        return x.view(batch_size, seq_length, 3)  # Reshape back to [batch_size, seq_length, 3]

def train_model_diffusion(model, dataloader, optimizer, criterion, device, num_epochs, noiseadding_steps):
    """
    Trains the NoisePredictor model using diffusion-based noisy trajectories.

    Args:
        model (nn.Module): The NoisePredictor model.
        dataloader (DataLoader): DataLoader for training data.
        optimizer (optim.Optimizer): Optimizer for parameter updates.
        criterion (nn.Module): Loss function.
        device (torch.device): Device for training (CPU or GPU).
        num_epochs (int): Number of training epochs.
        noiseadding_steps (int): Number of steps to add noise.

    Returns:
        list: List of average losses for each epoch.
    """
    model.train()
    epoch_losses = []

    for epoch in range(num_epochs):
        total_loss = 0
        batch_count = 0  # To count the number of batches used in the epoch
        print(f"Starting epoch {epoch + 1}/{num_epochs}...")

        for batch_idx, clean_trajectory in enumerate(dataloader):
            batch_count += 1
            #print(f"Processing batch {batch_idx + 1}/{len(dataloader)}...")
            
            clean_trajectory = clean_trajectory.to(device)
            
            # Add noise to the clean trajectory
            noisy_trajectory = add_noise(clean_trajectory, noiseadding_steps)
            
            optimizer.zero_grad()
            
            # Predict the clean trajectory
            predicted_trajectory = model(noisy_trajectory)
            
            # Calculate loss
            loss = criterion(predicted_trajectory, clean_trajectory)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")
        #print(f"Total batches processed in epoch {epoch + 1}: {batch_count}")
    
    return epoch_losses


# Validation Loop for Diffusion
def validate_model_diffusion(model, dataloader, criterion, device, noiseadding_steps):
    """
    Validates the NoisePredictor model on unseen data.

    Args:
        model (nn.Module): The NoisePredictor model.
        dataloader (DataLoader): DataLoader for validation data.
        criterion (nn.Module): Loss function.
        device (torch.device): Device for validation (CPU or GPU).
        noiseadding_steps (int): Number of steps to add noise.

    Returns:
        float: Average validation loss.
    """
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for clean_trajectory in dataloader:
            clean_trajectory = clean_trajectory.to(device)

            # Add noise to the clean trajectory
            noisy_trajectory = add_noise(clean_trajectory, noiseadding_steps)

            # Predict the clean trajectory
            predicted_trajectory = model(noisy_trajectory)

            # Calculate loss
            loss = criterion(predicted_trajectory, clean_trajectory)
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Validation Loss: {avg_loss:.4f}")
    return avg_loss

def main():
    """
    Main function to execute the training and validation of the NoisePredictor model.
    """
    # Hyperparametersprint
    seq_length = 100
    input_dim = seq_length * 3  # Flattened input dimension
    hidden_dim = 64
    batch_size = 4
    num_epochs = 1000
    learning_rate = 1e-3
    noiseadding_steps = 10

    # File path to the real data
    file_path = "Data/1D_diffusion/SimData/sin"

    # Load real data
    data = load_robot_data(file_path, seq_length)
    #print(f"Total loaded data shape: {len(data)} samples, each with shape {data[0]['pos_0'].shape}")

    # Compute per-axis normalization statistics
    stats = compute_statistics_per_axis(data)

    # Normalize data per axis
    normalized_data = normalize_data_per_axis(data, stats)
    
    # Split into training and validation sets
    split = int(len(normalized_data) * 0.8)
    train_data = normalized_data[:split]
    val_data = normalized_data[split:]

    
    # Create datasets with per-axis normalization
    train_dataset = ImpedanceDatasetDiffusion(train_data, stats)
    val_dataset = ImpedanceDatasetDiffusion(val_data, stats)

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    
    # Model, optimizer, and loss function
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NoisePredictor(seq_length, hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # Train and validate
    train_losses = train_model_diffusion(model, train_loader, optimizer, criterion, device, num_epochs, noiseadding_steps)
    val_loss = validate_model_diffusion(model, val_loader, criterion, device, noiseadding_steps)
    
    # Plot training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
    plt.axhline(val_loss, color='red', linestyle='--', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

    # Visualize predictions
    model.eval()
    clean_trajectory = next(iter(val_loader)).to(device)  # Shape: [batch_size, seq_length, 3]

    with torch.no_grad():
        noisy_trajectory = add_noise(clean_trajectory, noiseadding_steps)
        predicted_trajectory = model(noisy_trajectory)

    # Denormalize for visualization
    clean_trajectory = val_dataset.denormalize(clean_trajectory.cpu())
    noisy_trajectory = val_dataset.denormalize(noisy_trajectory.cpu())
    predicted_trajectory = val_dataset.denormalize(predicted_trajectory.cpu())

    # Plot predictions for all 3 dimensions
    plt.figure(figsize=(10, 5))
    for i, label in enumerate(['x', 'y', 'z']):
        plt.plot(clean_trajectory[0, :, i], label=f'Clean {label}')
        plt.plot(noisy_trajectory[0, :, i], linestyle='--', label=f'Noisy {label}')
        plt.plot(predicted_trajectory[0, :, i], linestyle='-.', label=f'Predicted {label}')
    plt.xlabel('Time Step')
    plt.ylabel('Position')
    plt.title('Trajectory Prediction (3D)')
    plt.legend()
    plt.show()
    

if __name__ == "__main__":
    main()