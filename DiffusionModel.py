import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from models import NoisePredictor, NoisePredictorLSTM
from data import ImpedanceDatasetDiffusion, load_robot_data, compute_statistics_per_axis, normalize_data_per_axis
from train_val_test import train_model_diffusion, validate_model_diffusion
from utils import loss_function, add_noise

    
def main():
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
