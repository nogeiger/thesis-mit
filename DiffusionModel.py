import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

# Data Generation
def generate_data(num_samples=10000, seq_length=100):
    data = []
    for _ in range(num_samples):
        t = np.linspace(0, 10, seq_length)  # Timestamps
        clean_trajectory = np.sin(t) + np.cos(2 * t)  # Example clean trajectory
        noisy_trajectory = clean_trajectory + np.random.normal(0, 0.1, size=t.shape)  # Add noise
        forces = -np.gradient(clean_trajectory, t)  # Simulated forces
        sample = {
            "timestamp": t,
            "pos_0": clean_trajectory,
            "pos": noisy_trajectory,
            "force": forces,
        }
        data.append(sample)
    return data


# Dataset Class
class ImpedanceDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return (
            torch.tensor(sample["timestamp"], dtype=torch.float32),
            torch.tensor(sample["pos"], dtype=torch.float32),
            torch.tensor(sample["force"], dtype=torch.float32),
            torch.tensor(sample["pos_0"], dtype=torch.float32),
        )
    

# Diffusion Model Architecture
class DiffusionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DiffusionModel, self).__init__()
        
        # Define layers explicitly
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layer_1 = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()


    def forward(self, noisy_pos):#, forces, timestamps):
        #print("shape noisy inputs: ", noisy_pos.shape)
        #print("shape forces: ", forces.shape)
        #print("shape timestamps: ", timestamps.shape)
        #x = torch.cat([noisy_pos, forces, timestamps], dim=-1)

        #print("shape noisy inputs: ", noisy_pos.shape)
        
        # Forward pass through each layer
        x = self.input_layer(noisy_pos)
        x = self.relu(x)
        x = self.hidden_layer_1(x)
        x = self.relu(x)
        x = self.hidden_layer_2(x)
        x = self.relu(x)
        x = self.output_layer(x)
        
        #print("shape x after final layer: ", x.shape)
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

# Main Execution
def main():
    # Hyperparameters
    input_dim = 100  # noisy_pos, forces, timestamp
    hidden_dim = 64
    output_dim = 100  # Clean trajectory output
    batch_size = 32
    num_epochs = 20
    learning_rate = 1e-3

    # Generate data
    data = generate_data()
    split = int(len(data) * 0.8)
    train_data = data[:split]
    val_data = data[split:]


    # Print data shapes for testing
    print(f"Total data size: {len(data)}")
    print(f"Training data size: {len(train_data)}")
    print(f"Validation data size: {len(val_data)}")

    # Inspect one sample from training data
    sample = train_data[0]
    print("---------------------------------")
    print("Sample keys:", sample.keys())
    print("Timestamps shape:", sample['timestamp'].shape)
    print("Noisy trajectory shape (pos):", sample['pos'].shape)
    print("Forces shape:", sample['force'].shape)
    print("Clean trajectory shape (pos_0):", sample['pos_0'].shape)

    # Datasets and Dataloaders
    train_dataset = ImpedanceDataset(train_data)
    val_dataset = ImpedanceDataset(val_data)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Check the training data loader
    print("---------------------------------")
    print("Inspecting training data loader:")
    for batch_idx, (timestamps, noisy_pos, forces, clean_pos) in enumerate(train_loader):
        print(f"Batch {batch_idx + 1}:")
        print(f"  Timestamps shape: {timestamps.shape}")
        print(f"  Noisy trajectory shape: {noisy_pos.shape}")
        print(f"  Forces shape: {forces.shape}")
        print(f"  Clean trajectory shape: {clean_pos.shape}")
        break  # Only check the first batch

    # Check the validation data loader
    print("---------------------------------")
    print("Inspecting validation data loader:")
    for batch_idx, (timestamps, noisy_pos, forces, clean_pos) in enumerate(val_loader):
        print(f"Batch {batch_idx + 1}:")
        print(f"  Timestamps shape: {timestamps.shape}")
        print(f"  Noisy trajectory shape: {noisy_pos.shape}")
        print(f"  Forces shape: {forces.shape}")
        print(f"  Clean trajectory shape: {clean_pos.shape}")
        break  # Only check the first batch

    # Model, optimizer, and loss function

    #activate gpu later on
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")


    model = DiffusionModel(input_dim, hidden_dim, output_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

     # Train and validate
    train_losses = train_model(model, train_loader, optimizer, criterion, device, num_epochs)
    val_loss = validate_model(model, val_loader, criterion, device)

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
    timestamps, noisy_pos, forces, clean_pos = next(iter(val_loader))
    timestamps, noisy_pos, forces, clean_pos = (
        timestamps.to(device),
        noisy_pos.to(device),
        forces.to(device),
        clean_pos.to(device),
    )

    with torch.no_grad():
        predicted_pos = model(noisy_pos)

    # Plot a single sequence (e.g., the first one in the batch)
    plt.figure(figsize=(10, 5))
    plt.plot(clean_pos[0].cpu().numpy(), label='Real Trajectory')
    plt.plot(predicted_pos[0].cpu().numpy(), label='Predicted Trajectory', linestyle='--')
    plt.xlabel('Time Step')
    plt.ylabel('Position')
    plt.title('Real vs Predicted Trajectory')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
