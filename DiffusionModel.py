import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# Set seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Data generation
def generate_data(num_samples=1000, timesteps=100):
    data = []
    for _ in range(num_samples):
        t = np.linspace(0, 10, timesteps)
        pos_0_x = np.sin(t)
        pos_0_y = np.cos(t)
        pos_0_z = np.sin(2 * t)
        
        # Add noise to positions
        pos_x = pos_0_x + np.random.normal(0, 0.1, timesteps)
        pos_y = pos_0_y + np.random.normal(0, 0.1, timesteps)
        pos_z = pos_0_z + np.random.normal(0, 0.1, timesteps)
        
        # Generate random forces
        force_x = np.random.normal(0, 0.5, timesteps)
        force_y = np.random.normal(0, 0.5, timesteps)
        force_z = np.random.normal(0, 0.5, timesteps)
        
        sample = np.stack([t, pos_0_x, pos_0_y, pos_0_z, pos_x, pos_y, pos_z, force_x, force_y, force_z], axis=1)
        data.append(sample)
    return np.array(data)


# Dataset class
class TrajectoryDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        t = sample[:, 0]
        pos_0 = sample[:, 1:4]
        pos = sample[:, 4:7]
        forces = sample[:, 7:10]
        return torch.tensor(np.concatenate([pos, forces], axis=1), dtype=torch.float32), torch.tensor(pos_0, dtype=torch.float32)


# Define U-Net model
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(6, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.bottleneck = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 3, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)
        return x
    

# Training function
def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for inputs, targets in train_loader:
            inputs = inputs.permute(0, 2, 1)  # Change to [batch, channels, timesteps]
            targets = targets.permute(0, 2, 1)  # Change to [batch, channels, timesteps]
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_losses.append(train_loss / len(train_loader))

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.permute(0, 2, 1)
                targets = targets.permute(0, 2, 1)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        val_losses.append(val_loss / len(val_loader))
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")

    return train_losses, val_losses


# Plot predictions
def plot_predictions(model, val_loader):
    model.eval()
    inputs, targets = next(iter(val_loader))
    inputs = inputs.permute(0, 2, 1)
    targets = targets.permute(0, 2, 1)
    with torch.no_grad():
        outputs = model(inputs)

    # Select one trajectory for visualization
    idx = 0
    t = np.linspace(0, 10, inputs.size(-1))
    plt.figure(figsize=(12, 6))
    for i, label in enumerate(['x', 'y', 'z']):
        plt.plot(t, targets[idx, i, :].numpy(), label=f"True {label}")
        plt.plot(t, outputs[idx, i, :].numpy(), '--', label=f"Predicted {label}")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Position")
    plt.title("True vs Predicted Trajectories")
    plt.show()

# Main script
if __name__ == "__main__":
    # Generate synthetic data
    data = generate_data(num_samples=200, timesteps=100)
    train_data = data[:150]
    val_data = data[150:]

    # Create datasets and dataloaders
    train_dataset = TrajectoryDataset(train_data)
    val_dataset = TrajectoryDataset(val_data)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Initialize model
    model = UNet()

    # Train model
    train_losses, val_losses = train_model(model, train_loader, val_loader, num_epochs=50)

    # Plot validation predictions
    plot_predictions(model, val_loader)