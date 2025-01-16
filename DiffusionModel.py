import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

# Data Generation for synthetic data
def generate_data(num_samples=10000, seq_length=100):
    #pos_0 is clean data on sin curve
    data = []
    for _ in range(num_samples):
        t = np.linspace(0, 10, seq_length)  # Timestamps
        clean_trajectory = np.sin(t) + np.cos(2 * t)  # Example clean trajectory
        sample = {
            "pos_0": clean_trajectory,
        }
        data.append(sample)
    return data


def loss_function(predicted_noise, actual_noise):

    return nn.MSELoss()(predicted_noise, actual_noise)

# Add Noise Function
#change or adapt this function
def add_noise(clean_trajectory, noiseadding_steps):
    noisy_trajectory = clean_trajectory.clone()
    for _ in range(noiseadding_steps):
        noise = torch.randn_like(clean_trajectory) * 0.1  # Scale the noise
        noisy_trajectory += noise
    return noisy_trajectory

# Dataset Class
class ImpedanceDatasetInitial(Dataset):
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
    
# Dataset Class
class ImpedanceDatasetDiffusion(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return torch.tensor(sample["pos_0"], dtype=torch.float32)
    

# Diffusion Model Architecture
class InitialModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(InitialModel, self).__init__()
        
        # Define layean_trajectory, noiseadding_steps)

        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layer_1 = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()


    def forward(self, noisy_pos):#, forces, timestamps):

        
        # Forward pass through each layer
        x = self.input_layer(noisy_pos)
        x = self.relu(x)
        x = self.hidden_layer_1(x)
        x = self.relu(x)
        x = self.hidden_layer_2(x)
        x = self.relu(x)
        x = self.output_layer(x)

        return x
    
# Noise Predictor Model
class NoisePredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(NoisePredictor, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layer_1 = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, input_dim)
        self.relu = nn.ReLU()

    def forward(self, noisy_trajectory):
        x = self.input_layer(noisy_trajectory)
        x = self.relu(x)
        x = self.hidden_layer_1(x)
        x = self.relu(x)
        x = self.hidden_layer_2(x)
        x = self.relu(x)
        x = self.output_layer(x)
        return x

# Training Loop
def train_model(model, dataloader, optimizer, criterion, device, num_epochs=10):
    model.train()
    epoch_losses = []
    for epoch in range(num_epochs):
        total_loss = 0
        for timestamps, noisy_pos, forces, clean_pos in dataloader:
            timestamps, noisy_pos, forces, clean_pos = (
                timestamps.to(device),
                noisy_pos.to(device),
                forces.to(device),
                clean_pos.to(device),
            )

            optimizer.zero_grad()

            predicted_pos = model(noisy_pos)
            loss = criterion(predicted_pos, clean_pos)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")
    return epoch_losses

# Training Loop
def train_model_diffusion(model, dataloader, optimizer, criterion, device, num_epochs, noiseadding_steps):
    model.train()
    epoch_losses = []
    for epoch in range(num_epochs):
        total_loss = 0
        for clean_trajectory in dataloader:
            clean_trajectory = clean_trajectory.to(device)

            # Add noise to the clean trajenoisyctory
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


# Validation Loop
def validate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for timestamps, noisy_pos, forces, clean_pos in dataloader:
            timestamps, noisy_pos, forces, clean_pos = (
                timestamps.to(device),
                noisy_pos.to(device),
                forces.to(device),
                clean_pos.to(device),
            )

            predicted_pos = model(noisy_pos)
            loss = criterion(predicted_pos, clean_pos)
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Validation Loss: {avg_loss:.4f}")
    return avg_loss

# Validation Loop
def validate_model_diffusion(model, dataloader, criterion, device, noiseadding_steps):
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
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
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
