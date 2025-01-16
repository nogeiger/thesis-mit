import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

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
        t = np.linspace(0, 10, seq_length)  # Timestamps
        clean_trajectory = np.sin(t) + np.cos(2 * t)  # Example clean trajectory
        sample = {
            "pos_0": clean_trajectory,
        }
        data.append(sample)
    print(f"Generated {num_samples} samples, each with a sequence length of {seq_length}.")
    return data

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

# Add Noise Function
def add_noise(clean_trajectory, noiseadding_steps):
    """
    Dynamically adds Gaussian noise to a clean trajectory.

    Args:
        clean_trajectory (torch.Tensor): The clean trajectory.
        noiseadding_steps (int): Number of steps to iteratively add noise.

    Returns:
        torch.Tensor: Noisy trajectory.
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

# Dataset Class for diffusion-based training
class ImpedanceDatasetDiffusion(Dataset):
    """
    A dataset class for diffusion-based training.

    Args:
        data (list): A list of dictionaries containing clean trajectories.

    Returns:
        torch.Tensor: Clean trajectory (pos_0).
    """
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return torch.tensor(sample["pos_0"], dtype=torch.float32)

# Noise Predictor Model
class NoisePredictor(nn.Module):
    """
    A feedforward neural network to predict clean trajectories from noisy inputs.
    """
    def __init__(self, input_dim, hidden_dim):
        super(NoisePredictor, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layer_1 = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, input_dim)
        self.relu = nn.ReLU()

    def forward(self, noisy_trajectory):
        """
        Forward pass to predict clean trajectory.

        Args:
            noisy_trajectory (torch.Tensor): Input noisy trajectory.

        Returns:
            torch.Tensor: Predicted clean trajectory.
        """
        x = self.input_layer(noisy_trajectory)
        x = self.relu(x)
        x = self.hidden_layer_1(x)
        x = self.relu(x)
        x = self.hidden_layer_2(x)
        x = self.relu(x)
        x = self.output_layer(x)
        return x

# Training Loop for Diffusion
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
        for clean_trajectory in dataloader:
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

# Main Execution
def main():
    """
    Main function to execute the training and validation of the NoisePredictor model.
    """
    # Hyperparameters
    input_dim = 100  # Sequence length
    hidden_dim = 64
    batch_size = 32
    num_epochs = 20
    learning_rate = 1e-3
    noiseadding_steps = 5

    # Generate data
    data = generate_data()
    split = int(len(data) * 0.8)
    train_data = data[:split]
    val_data = data[split:]

    # Datasets and Dataloaders
    train_dataset = ImpedanceDatasetDiffusion(train_data)
    val_dataset = ImpedanceDatasetDiffusion(val_data)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model, optimizer, and loss function
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NoisePredictor(input_dim, hidden_dim).to(device)
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
    clean_trajectory = next(iter(val_loader)).to(device)

    with torch.no_grad():
        noisy_trajectory = add_noise(clean_trajectory, noiseadding_steps)
        predicted_trajectory = model(noisy_trajectory)

    # Plot a single sequence
    plt.figure(figsize=(10, 5))
    plt.plot(clean_trajectory[0].cpu().numpy(), label='Clean Trajectory')
    plt.plot(noisy_trajectory[0].cpu().numpy(), label='Noisy Trajectory', linestyle='--')
    plt.plot(predicted_trajectory[0].cpu().numpy(), label='Predicted Trajectory', linestyle='-.')
    plt.xlabel('Time Step')
    plt.ylabel('Position')
    plt.title('Trajectory Prediction')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
