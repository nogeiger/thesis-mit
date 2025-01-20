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
    # Concatenate both clean (pos_0) and noisy (pos) trajectories for each sample
    pos_0_data = np.concatenate([sample["pos_0"] for sample in data], axis=0)  # Shape: [total_points, 3]
    pos_data = np.concatenate([sample["pos"] for sample in data], axis=0)  # Shape: [total_points, 3]

    # Calculate min and max values for each axis (x, y, z) separately
    min_vals_pos_0 = torch.tensor(np.min(pos_0_data, axis=0), dtype=torch.float32)  # Min for clean (pos_0)
    max_vals_pos_0 = torch.tensor(np.max(pos_0_data, axis=0), dtype=torch.float32)  # Max for clean (pos_0)
    
    min_vals_pos = torch.tensor(np.min(pos_data, axis=0), dtype=torch.float32)  # Min for noisy (pos)
    max_vals_pos = torch.tensor(np.max(pos_data, axis=0), dtype=torch.float32)  # Max for noisy (pos)

    # Prevent division by zero for constant axes in both pos_0 and pos
    epsilon = 1e-8
    max_vals_pos_0 = torch.where(max_vals_pos_0 == min_vals_pos_0, max_vals_pos_0 + epsilon, max_vals_pos_0)
    max_vals_pos = torch.where(max_vals_pos == min_vals_pos, max_vals_pos + epsilon, max_vals_pos)

    # Return the min and max for both pos_0 (clean) and pos (noisy), per axis (x, y, z)
    return {
        "min_pos_0": min_vals_pos_0, 
        "max_pos_0": max_vals_pos_0, 
        "min_pos": min_vals_pos, 
        "max_pos": max_vals_pos
    }


def normalize_data_per_axis(data, stats):
    """
    Normalizes each axis (x, y, z) in both clean (pos_0) and noisy (pos) trajectories.
    Constant axes are assigned a fixed normalized value (e.g., 0.5).

    Args:
        data (list): A list of dictionaries containing clean and noisy trajectories.
        stats (dict): Min and max values for normalization for both clean and noisy trajectories.

    Returns:
        list: A list of dictionaries with normalized clean (pos_0) and noisy (pos) trajectories.
    """
    normalized_data = []
    
    for sample in data:
        pos_0 = torch.tensor(sample["pos_0"], dtype=torch.float32)  # Clean trajectory [seq_length, 3]
        pos = torch.tensor(sample["pos"], dtype=torch.float32)  # Noisy trajectory [seq_length, 3]

        # Retrieve min and max values for both clean and noisy trajectories
        min_vals_pos_0, max_vals_pos_0 = stats["min_pos_0"], stats["max_pos_0"]
        min_vals_pos, max_vals_pos = stats["min_pos"], stats["max_pos"]

        # Normalize clean trajectory (pos_0)
        range_vals_pos_0 = max_vals_pos_0 - min_vals_pos_0
        is_constant_pos_0 = range_vals_pos_0 == 0  # Check for constant value per axis
        range_vals_pos_0 = torch.where(is_constant_pos_0, torch.ones_like(range_vals_pos_0), range_vals_pos_0)
        normalized_pos_0 = (pos_0 - min_vals_pos_0) / range_vals_pos_0

        # Assign fixed normalized value (e.g., 0.5) for constant axes in pos_0
        for axis in range(pos_0.shape[-1]):  # Iterate over x, y, z
            if is_constant_pos_0[axis].item():  # Use .item() to access the value correctly
                normalized_pos_0[:, axis] = 0.5  # Fixed normalized value for constant axes in pos_0

        # Normalize noisy trajectory (pos)
        range_vals_pos = max_vals_pos - min_vals_pos
        is_constant_pos = range_vals_pos == 0  # Check for constant value per axis
        range_vals_pos = torch.where(is_constant_pos, torch.ones_like(range_vals_pos), range_vals_pos)
        normalized_pos = (pos - min_vals_pos) / range_vals_pos

        # Assign fixed normalized value (e.g., 0.5) for constant axes in pos
        for axis in range(pos.shape[-1]):  # Iterate over x, y, z
            if is_constant_pos[axis].item():  # Use .item() to access the value correctly
                normalized_pos[:, axis] = 0.5  # Fixed normalized value for constant axes in pos

        # Debugging: Check for anomalies
        if torch.any(torch.isinf(normalized_pos_0)) or torch.any(torch.isnan(normalized_pos_0)):
            print("Error: Found inf/nan in normalized_pos_0:", normalized_pos_0)
        if torch.any(torch.isinf(normalized_pos)) or torch.any(torch.isnan(normalized_pos)):
            print("Error: Found inf/nan in normalized_pos:", normalized_pos)

        # Append normalized data (both pos_0 and pos)
        normalized_data.append({"pos_0": normalized_pos_0, "pos": normalized_pos})

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

                    noisy_trajectory = np.stack([
                        df["Pos_x"].iloc[i:i + seq_length].values,
                        df["Pos_y"].iloc[i:i + seq_length].values,
                        df["Pos_z"].iloc[i:i + seq_length].values,
                    ], axis=-1)  # Shape: [seq_length, 3]

                    sample = {"pos_0": clean_trajectory,
                              "pos": noisy_trajectory,
                              }
                    file_data.append(sample)

                print(f"Loaded {len(file_data)} samples from {filename}, each with a sequence length of {seq_length} in 3D.")
                all_data.extend(file_data)

            except Exception as e:
                print(f"Error loading data from {filename}: {e}")

    print(f"Total loaded samples from all files: {len(all_data)}")
    return all_data


def loss_function(predicted_noise, actual_noise):
    """
    Computes the mean squared error between the predicted and actual noise.
    """
    return nn.MSELoss()(predicted_noise, actual_noise)

def add_noise(clean_trajectory, max_noiseadding_steps, beta_start=0.0001, beta_end=0.02):
    """
    Dynamically adds Gaussian noise to a clean 3D trajectory based on a diffusion model schedule.
    Noise is progressively added over several steps according to a linear schedule.

    Args:
        clean_trajectory (torch.Tensor): The clean trajectory with shape [seq_length, 3].
        noiseadding_steps (int): Number of steps to iteratively add noise.
        beta_start (float): Initial value of noise scale.
        beta_end (float): Final value of noise scale.

    Returns:
        torch.Tensor: Noisy trajectory with shape [seq_length, 3].
    """
    # Randomly choose the number of noise adding steps between 1 and max_noiseadding_steps
    noiseadding_steps = torch.randint(1, max_noiseadding_steps + 1, (1,)).item()
    
    noisy_trajectory = clean_trajectory.clone()
    
    # Linear schedule for noise scale (beta values)
    beta_values = torch.linspace(beta_start, beta_end, noiseadding_steps)  # Linearly spaced values between beta_start and beta_end
    
    for step in range(noiseadding_steps):
        # Get current noise scale based on the diffusion schedule
        beta = beta_values[step]  # Beta increases over time
        
        # Add Gaussian noise scaled by the current beta
        noise = torch.randn_like(clean_trajectory) * torch.sqrt(beta)  # Apply sqrt(beta) for scaling the noise
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
        pos_0 = torch.tensor(sample["pos_0"], dtype=torch.float32) 
        pos = torch.tensor(sample["pos"], dtype=torch.float32) # Already normalized
        return pos_0, pos

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
            else:
                raise ValueError("Invalid trajectory type. Must be 'pos_0' or 'pos'.")

            # Denormalize using the corresponding min and max
            denormalized_data = normalized_data * (max_vals - min_vals) + min_vals
            return denormalized_data
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
        predicted_noise = self.output_layer(x)
        return predicted_noise.view(batch_size, seq_length, 3)  # Reshape back to [batch_size, seq_length, 3]
    
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

        for batch_idx, (pos_0,pos) in enumerate(dataloader):
            batch_count += 1
            #print(f"Processing batch {batch_idx + 1}/{len(dataloader)}...")
            #print("Shape of clean trajectory (pos_0):", pos_0.shape)
            #print("Shape of noisy trajectory (pos):", pos.shape)


            clean_trajectory = pos_0.to(device)
            noisy_trajectory = pos.to(device)
            
            # Add noise to the clean trajectory
            #noisy_trajectory = add_noise(clean_trajectory, noiseadding_steps)
            
            # Compute the actual noise added
            actual_noise = noisy_trajectory - clean_trajectory  # The difference between noisy and clean trajectory

            optimizer.zero_grad()

            # Predict the noise from the noisy trajectory
            predicted_noise = model(noisy_trajectory)

            # Calculate loss between predicted noise and actual noise
            loss = criterion(predicted_noise, actual_noise)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
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
        for batch_idx, (pos_0,pos) in enumerate(dataloader):
            clean_trajectory = pos_0.to(device)
            noisy_trajectory = pos.to(device)
            
            # Add noise to the clean trajectory
            #noisy_trajectory = add_noise(clean_trajectory, noiseadding_steps)
            noisy_trajectory = pos

            # Calculate the actus_constant_pos_0[axis]al noise added
            actual_noise = noisy_trajectory - clean_trajectory

            # Predict the noise from the noisy trajectory
            predicted_noise = model(noisy_trajectory)

            # Calculate loss between predicted noise and actual noise
            loss = criterion(predicted_noise, actual_noise)
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Validation Loss: {avg_loss:.4f}")
    return avg_loss




def main():
    #To do: 
    #noise durch max step size aufteilen
    #add noise anpassen
    #force hinzufuegen



    """
    Main function to execute the training and validation of the NoisePredictor model.
    """

    #TRAINING AND TESTING
    ####################
    # Hyperparameters
    seq_length = 100
    input_dim = seq_length * 3  # Flattened input dimension
    hidden_dim = 128
    batch_size = 4
    num_epochs = 1
    learning_rate = 1e-3
    noiseadding_steps = 1000

    # File path to the real data
    file_path = "Data/1D_diffusion/SimData/sin"

    # Load real data
    data = load_robot_data(file_path, seq_length)
    
    # Compute per-axis normalization statistics
    stats = compute_statistics_per_axis(data)

    # Normalize data per axis
    normalized_data = normalize_data_per_axis(data, stats)
    #print("Normalized data shape:", normalized_data.shape)
    
    # Split into training and validation sets
    split = int(len(normalized_data) * 0.8)
    train_data = normalized_data[:split]
    val_data = normalized_data[split:]

    # Create datasets with per-axis normalization
    train_dataset = ImpedanceDatasetDiffusion(train_data, stats)
    val_dataset = ImpedanceDatasetDiffusion(train_data, stats)

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
    
    # Visualize predictions (Noise and Clean Trajectories)
    model.eval()
    clean_trajectory,noisy_trajectory = next(iter(val_loader))#.to(device)  # Shape: [batch_size, seq_length, 3]
    clean_trajectory = clean_trajectory.to(device)
    noisy_trajectory = noisy_trajectory.to(device)
    
    with torch.no_grad():        
        # Calculate the actual noise added to the clean trajectory
        actual_noise = noisy_trajectory - clean_trajectory
        
        # Predict the noise from the noisy trajectory
        predicted_noise = model(noisy_trajectory)
        
        # Recover the predicted clean trajectory by subtracting predicted noise from noisy trajectory
        predicted_clean_trajectory = noisy_trajectory - predicted_noise

    
    # Denormalize for visualization
    clean_trajectory = val_dataset.denormalize(clean_trajectory.cpu(), "pos_0")
    noisy_trajectory = val_dataset.denormalize(noisy_trajectory.cpu(),"pos")
    predicted_clean_trajectory = val_dataset.denormalize(predicted_clean_trajectory.cpu())  # Denormalize predicted clean trajectory

    # Move to CPU for plotting
    actual_noise = actual_noise.cpu()
    predicted_noise = predicted_noise.cpu()
    
    # Plot predictions for the noise (x, y, z)
    plt.figure(figsize=(10, 5))
    for i, label in enumerate(['x', 'y', 'z']):
        if i != 1:
            continue
        plt.plot(actual_noise[0, :, i], label=f'Actual Noise {label}')
        plt.plot(predicted_noise[0, :, i], linestyle='-.', label=f'Predicted Noise {label}')
    plt.xlabel('Time Step')
    plt.ylabel('Noise')
    plt.title('Noise Prediction')
    plt.legend()
    plt.show()
    
    # Inference: Iterative denoising
    ###########################
    # Start with a noisy trajectory and progressively denoise
    denoised_trajectory = noisy_trajectory.clone()

    # Number of denoising steps
    num_denoising_steps = noiseadding_steps  # Typically, this should be the same as the number of noise steps used in training

    for step in range(num_denoising_steps):
        # Predict the noise at the current step
        predicted_noise = model(denoised_trajectory)

        # Subtract the predicted noise to denoise the trajectory
        denoised_trajectory = denoised_trajectory - predicted_noise

    # Detach and move the clean and denoised trajectories to CPU before plotting
    clean_trajectory = clean_trajectory.detach().cpu().numpy()
    denoised_trajectory = denoised_trajectory.detach().cpu().numpy()

    # Plot the clean trajectory and denoised trajectory
    plt.figure(figsize=(10, 5))
    for i, label in enumerate(['x', 'y', 'z']):
        if i != 1:  # Adjust this condition based on which axis you want to plot
            continue

        # Plot the ground truth clean trajectory
        plt.plot(clean_trajectory[0, :, i], label=f'Ground Truth Clean {label}')
        
        # Plot the denoised trajectory
        plt.plot(denoised_trajectory[0, :, i], linestyle='-.', label=f'Denoised {label}')
        
    plt.xlabel('Time Step')
    plt.ylabel('Position')
    plt.title('Clean and Denoised Trajectory Comparison')
    plt.legend()
    plt.show()
    
if __name__ == "__main__":
    main()