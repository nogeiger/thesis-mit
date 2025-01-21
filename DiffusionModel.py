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
    force_data = np.concatenate([sample["force"] for sample in data], axis=0)  # Shape: [total_points, 3]

    # Calculate min and max values for each axis (x, y, z) separately
    min_vals_pos_0 = torch.tensor(np.min(pos_0_data, axis=0), dtype=torch.float32)  # Min for clean (pos_0)
    max_vals_pos_0 = torch.tensor(np.max(pos_0_data, axis=0), dtype=torch.float32)  # Max for clean (pos_0)

    min_vals_pos = torch.tensor(np.min(pos_data, axis=0), dtype=torch.float32)  # Min for noisy (pos)
    max_vals_pos = torch.tensor(np.max(pos_data, axis=0), dtype=torch.float32)  # Max for noisy (pos)

    min_vals_force = torch.tensor(np.min(force_data, axis=0), dtype=torch.float32)  # Min for noisy (pos)
    max_vals_force = torch.tensor(np.max(force_data, axis=0), dtype=torch.float32)  # Max for noisy (pos)

    # Prevent division by zero for constant axes in both pos_0 and pos
    epsilon = 1e-8
    max_vals_pos_0 = torch.where(max_vals_pos_0 == min_vals_pos_0, max_vals_pos_0 + epsilon, max_vals_pos_0)
    max_vals_pos = torch.where(max_vals_pos == min_vals_pos, max_vals_pos + epsilon, max_vals_pos)
    max_vals_force = torch.where(max_vals_force == min_vals_force, max_vals_force + epsilon, max_vals_force)



    # Return the min and max for both pos_0 (clean) and pos (noisy), per axis (x, y, z)
    return {
        "min_pos_0": min_vals_pos_0, 
        "max_pos_0": max_vals_pos_0, 
        "min_pos": min_vals_pos, 
        "max_pos": max_vals_pos,
        "min_force": min_vals_force,
        "max_force": max_vals_force,
    }


def normalize_data_per_axis(data, stats):
    """
    Normalizes each axis (x, y, z) in both clean (pos_0) and noisy (pos) trajectories, and also normalizes the forces.
    
    Args:
        data (list): A list of dictionaries containing clean and noisy trajectories and forces.
        stats (dict): Min and max values for normalization for both clean and noisy trajectories and forces.
    
    Returns:
        list: A list of dictionaries with normalized clean (pos_0), noisy (pos), and forces.
    """
    normalized_data = []
    
    for sample in data:
        pos_0 = torch.tensor(sample["pos_0"], dtype=torch.float32)  # Clean trajectory [seq_length, 3]
        pos = torch.tensor(sample["pos"], dtype=torch.float32)  # Noisy trajectory [seq_length, 3]
        forces = torch.tensor(sample["force"], dtype=torch.float32)  # Forces [seq_length, 3]

        # Retrieve min and max values for both clean and noisy trajectories and forces
        min_vals_pos_0, max_vals_pos_0 = stats["min_pos_0"], stats["max_pos_0"]
        min_vals_pos, max_vals_pos = stats["min_pos"], stats["max_pos"]
        min_vals_force, max_vals_force = stats["min_force"], stats["max_force"]

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

        # Assign fixed normalized value (e.g., 0.5) for constant axes
        for axis in range(pos_0.shape[-1]):  # Iterate over x, y, z
            if is_constant_pos_0[axis].item():
                normalized_pos_0[:, axis] = 0.5
            if is_constant_pos[axis].item():
                normalized_pos[:, axis] = 0.5
            if is_constant_force[axis].item():
                normalized_force[:, axis] = 0.5

        # Debugging: Check for anomalies
        if torch.any(torch.isinf(normalized_pos_0)) or torch.any(torch.isnan(normalized_pos_0)):
            print("Error: Found inf/nan in normalized_pos_0:", normalized_pos_0)
        if torch.any(torch.isinf(normalized_pos)) or torch.any(torch.isnan(normalized_pos)):
            print("Error: Found inf/nan in normalized_pos:", normalized_pos)
        if torch.any(torch.isinf(normalized_force)) or torch.any(torch.isnan(normalized_force)):
            print("Error: Found inf/nan in normalized_force:", normalized_force)

        # Append normalized data (both pos_0, pos, and force)
        normalized_data.append({
            "pos_0": normalized_pos_0,
            "pos": normalized_pos,
            "force": normalized_force
        })

    return normalized_data



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
        list: A combined list of dictionaries, each containing 'pos_0' with shape [seq_length, 3] and 'force' with shape [seq_length, 3].
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

                    # Extract forces
                    forces = np.stack([
                        df["Force_x"].iloc[i:i + seq_length].values,
                        df["Force_y"].iloc[i:i + seq_length].values,
                        df["Force_z"].iloc[i:i + seq_length].values,
                    ], axis=-1)  # Shape: [seq_length, 3]

                    sample = {
                        "pos_0": clean_trajectory,
                        "pos": noisy_trajectory,
                        "force": forces  # Add forces to the sample
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

def add_noise(clean_trajectory, noisy_trajectory, max_noiseadding_steps, beta_start=0.8, beta_end=0.1):
    """
    Dynamically adds noise to a clean 3D trajectory based on the actual noise between the clean and noisy trajectories,
    following a diffusion model schedule.

    Args:
        clean_trajectory (torch.Tensor): The clean trajectory with shape [seq_length, 3].
        noisy_trajectory (torch.Tensor): The noisy trajectory with shape [seq_length, 3].
        max_noiseadding_steps (int): Maximum number of steps to iteratively add noise.
        beta_start (float): Initial value of noise scale.
        beta_end (float): Final value of noise scale.

    Returns:
        torch.Tensor: Noisy trajectory with shape [seq_length, 3].
    """
    # Calculate the actual noise (difference between clean and noisy)
    actual_noise = noisy_trajectory - clean_trajectory

    # Randomly choose the number of noise adding steps between 1 and max_noiseadding_steps
    noiseadding_steps = torch.randint(1, max_noiseadding_steps + 1, (1,)).item()

    # Initialize the noisy trajectory as the clean trajectory
    noisy_trajectory_output = clean_trajectory.clone()

    # Linear schedule for noise scale (beta values)
    beta_values = torch.linspace(beta_start, beta_end, noiseadding_steps)  # Linearly spaced values between beta_start and beta_end
    
    for step in range(noiseadding_steps):
        # Get current noise scale based on the diffusion schedule
        beta = beta_values[step]  # Beta increases over time

        # Scale the actual noise by sqrt(beta) and add it to the clean trajectory
        noise_to_add = actual_noise * torch.sqrt(beta)
        noisy_trajectory_output += noise_to_add
    
    return noisy_trajectory_output


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
        pos = torch.tensor(sample["pos"], dtype=torch.float32) # Already normalized
        force = torch.tensor(sample["force"], dtype=torch.float32) # Already normalized
        return pos_0, pos, force

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
    A feedforward neural network to predict clean 3D trajectories from noisy inputs, 
    including forces as extra features if the flag is set.
    """
    def __init__(self, seq_length, hidden_dim, use_forces=False):
        super(NoisePredictor, self).__init__()
        self.use_forces = use_forces
        input_dim = seq_length * 3  # Clean and noisy trajectories (pos_0 and pos)
        
        if self.use_forces:
            input_dim += seq_length * 3  # Add forces (force_x, force_y, force_z)

        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layer_1 = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, seq_length * 3)  # Output clean trajectory (pos_0)
        self.relu = nn.ReLU()

    def forward(self, noisy_trajectory, forces=None):
        """
        Forward pass to predict clean 3D trajectory.

        Args:
            noisy_trajectory (torch.Tensor): Input noisy trajectory of shape [batch_size, seq_length, 3].
            forces (torch.Tensor, optional): Input forces of shape [batch_size, seq_length, 3].

        Returns:
            torch.Tensor: Predicted clean trajectory of shape [batch_size, seq_length, 3].
        """
        batch_size, seq_length, _ = noisy_trajectory.shape

        # If using forces, concatenate them with noisy trajectory
        if self.use_forces:
            x = torch.cat((noisy_trajectory, forces), dim=-1)  # Concatenate noisy trajectory and forces
        else:
            x = noisy_trajectory

        x = x.view(batch_size, -1)  # Flatten to [batch_size, seq_length * 6] if forces are included
        x = self.input_layer(x)
        x = self.relu(x)
        x = self.hidden_layer_1(x)
        x = self.relu(x)
        x = self.hidden_layer_2(x)
        x = self.relu(x)
        predicted_noise = self.output_layer(x)
        return predicted_noise.view(batch_size, seq_length, 3)  # Reshape back to [batch_size, seq_length, 3]

class NoisePredictorLSTM(nn.Module):
    def __init__(self, seq_length, hidden_dim, use_forces=False):
        super(NoisePredictorLSTM, self).__init__()
        self.use_forces = use_forces
        input_dim = 3  # Each timestep has (x, y, z)

        if self.use_forces:
            input_dim += 3  # Add forces (force_x, force_y, force_z)

        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=2)  # Two LSTM layers
        self.fc = nn.Linear(hidden_dim, 3)  # Predict (x, y, z) noise for each timestep

    def forward(self, noisy_trajectory, forces=None):
        batch_size, seq_length, _ = noisy_trajectory.shape

        if self.use_forces:
            x = torch.cat((noisy_trajectory, forces), dim=-1)  # Concatenate noisy trajectory and forces
        else:
            x = noisy_trajectory

        # Pass through LSTM
        lstm_out, _ = self.lstm(x)  # Shape: [batch_size, seq_length, hidden_dim]

        # Predict noise for each timestep
        predicted_noise = self.fc(lstm_out)  # Shape: [batch_size, seq_length, 3]

        return predicted_noise

def train_model_diffusion(model, dataloader, optimizer, criterion, device, num_epochs, noiseadding_steps, use_forces=False):
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
        use_forces (bool): Whether to use forces as additional input to the model.

    Returns:
        list: List of average losses for each epoch.
    """
    model.train()
    epoch_losses = []

    for epoch in range(num_epochs):
        total_loss = 0
        batch_count = 0  # To count the number of batches used in the epoch
        print(f"Starting epoch {epoch + 1}/{num_epochs}...")

        for batch_idx, (pos_0, pos, force) in enumerate(dataloader):
            batch_count += 1

            # Move data to device
            clean_trajectory = pos_0.to(device)
            complete_noisy_trajectory = pos.to(device)
            force = force.to(device)

            # Dynamically add noise based on the actual difference (using the diffusion schedule) 
            # between the clean trajectory and the complete noisy trajectory
            noisy_trajectory = add_noise(clean_trajectory, complete_noisy_trajectory, noiseadding_steps)
            
            # Compute the actual noise added
            actual_noise = noisy_trajectory - clean_trajectory  # The difference between noisy and clean trajectory

            optimizer.zero_grad()

            # Predict the noise from the noisy trajectory (and forces if use_forces is True)
            if use_forces:
                predicted_noise = model(noisy_trajectory, force)
            else:
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



def validate_model_diffusion(model, dataloader, criterion, device, max_noiseadding_steps, use_forces=False):
    """
    Validates the NoisePredictor model on unseen data using diffusion-based noisy trajectories.

    Args:
        model (nn.Module): The NoisePredictor model.
        dataloader (DataLoader): DataLoader for validation data.
        criterion (nn.Module): Loss function.
        device (torch.device): Device for validation (CPU or GPU).
        max_noiseadding_steps (int): Maximum number of steps to add noise.
        use_forces (bool): Whether to use forces as additional input to the model.

    Returns:
        float: Average validation loss.
    """
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_idx, (pos_0, pos, force) in enumerate(dataloader):
            clean_trajectory = pos_0.to(device)
            noisy_trajectory = pos.to(device)
            force = force.to(device)

            # Dynamically add noise based on the actual difference (using the diffusion schedule)
            noisy_trajectory = add_noise(clean_trajectory, noisy_trajectory, max_noiseadding_steps)

            # Calculate the actual noise added (difference between noisy and clean)
            actual_noise = noisy_trajectory - clean_trajectory

            # Predict the noise from the noisy trajectory
            if use_forces:
                predicted_noise = model(noisy_trajectory, force)
            else:
                predicted_noise = model(noisy_trajectory)

            # Calculate loss between predicted noise and actual noise
            loss = criterion(predicted_noise, actual_noise)
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Validation Loss: {avg_loss:.4f}")
    return avg_loss




def main():
    #To do: 
    #force hinzufuegen

    """
    Main function to execute the training and validation of the NoisePredictor model.
    """

    # Hyperparameters
    seq_length = 100
    input_dim = seq_length * 3  # Flattened input dimension
    hidden_dim = 128
    batch_size = 32
    num_epochs = 5000 #2000
    learning_rate = 1e-3
    noiseadding_steps = 2 #5
    use_forces = True  # Set this to True if you want to use forces as input to the model

    # File path to the real data
    file_path = "Data/1D_diffusion/SimData"
    #file_path = "Data/1D_diffusion/SimData/sin"

    # Load real data
    data = load_robot_data(file_path, seq_length)
    
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
    model = NoisePredictor(seq_length, hidden_dim, use_forces=use_forces).to(device)
    #model = NoisePredictorLSTM(seq_length, hidden_dim, use_forces=use_forces).to(device)
    
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    # Train and validate
    train_losses = train_model_diffusion(model, train_loader, optimizer, criterion, device, num_epochs, noiseadding_steps, use_forces)
    val_loss = validate_model_diffusion(model, val_loader, criterion, device, noiseadding_steps, use_forces)
    
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
    clean_trajectory, noisy_trajectory, force = next(iter(val_loader))  # Get data from dataloader
    clean_trajectory = clean_trajectory.to(device)
    noisy_trajectory = noisy_trajectory.to(device)
    force = force.to(device)
    
    with torch.no_grad():        
        # Calculate the actual noise added to the clean trajectory
        actual_noise = noisy_trajectory - clean_trajectory
        
        # Predict the noise from the noisy trajectory
        predicted_noise = model(noisy_trajectory, force) if use_forces else model(noisy_trajectory)
        
        # Recover the predicted clean trajectory by subtracting predicted noise from noisy trajectory
        predicted_clean_trajectory = noisy_trajectory - predicted_noise

    # Denormalize for visualization
    clean_trajectory = val_dataset.denormalize(clean_trajectory.cpu(), "pos_0")
    noisy_trajectory = val_dataset.denormalize(noisy_trajectory.cpu(), "pos")
    predicted_clean_trajectory = val_dataset.denormalize(predicted_clean_trajectory.cpu())  # Denormalize predicted clean trajectory

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
        predicted_noise = model(denoised_trajectory, force) if use_forces else model(denoised_trajectory)

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
