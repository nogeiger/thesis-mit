import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from models import NoisePredictorInitial, NoisePredictorLSTM, NoisePredictorLSTMWithAttention, NoisePredictorTransformer, NoisePredictorGRU, NoisePredictorConvLSTM, NoisePredictorConv1D, NoisePredictorHybrid, NoisePredictorTCN
from data import ImpedanceDatasetDiffusion, load_robot_data, compute_statistics_per_axis, normalize_data_per_axis
from train_val_test import train_model_diffusion, validate_model_diffusion
from utils import loss_function, loss_function_start_point, add_noise, calculate_max_noise_factor

    
def main():
    """
    Main function to execute the training and validation of the NoisePredictor model.
    """ 
    
    # Hyperparameters
    seq_length = 100 #seq len of data
    input_dim = seq_length * 3  # Flattened input dimension
    hidden_dim = 128 #hidden dim of the model
    batch_size = 32 #batch size
    num_epochs = 100 #number of epochs
    learning_rate = 1e-3 #learning rate
    noiseadding_steps = 20 # Number of steps to add noise
    use_forces = True  # Set this to True if you want to use forces as input to the model
    noise_with_force = False#True # Set this to True if you want to use forces as the noise
    beta_start = 0.00001 #for the noise diffusion model
    beta_end = 0.00025 #for the noise diffusion model
    max_grad_norm=7.0 #max grad norm for gradient clipping 
    add_gaussian_noise = False#True # to add additional guassian noise

    # File path to the real data
    file_path = "Data/1D_diffusion_large_data"

    #if force is used as noise, then force should not be used as input
    if noise_with_force:
            use_forces = False

    print("max noise factor per batch and step: ",calculate_max_noise_factor(beta_start,beta_end,noiseadding_steps))

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

    #model = NoisePredictorInitial(seq_length, hidden_dim, use_forces=use_forces).to(device)
    #model = NoisePredictorTransformer(seq_length, hidden_dim, use_forces=use_forces).to(device)
    #model = NoisePredictorLSTMWithAttention(seq_length, hidden_dim, use_forces=use_forces).to(device)
    #model = NoisePredictorLSTM(seq_length, hidden_dim, use_forces=use_forces).to(device)
    #model = NoisePredictorGRU(seq_length, hidden_dim, use_forces=use_forces).to(device)
    #model = NoisePredictorConvLSTM(seq_length, hidden_dim, use_forces=use_forces).to(device)
    #model = NoisePredictorConv1D(seq_length, hidden_dim, use_forces=use_forces).to(device)
    #model = NoisePredictorHybrid(seq_length, hidden_dim, use_forces=use_forces).to(device)
    model = NoisePredictorTCN(seq_length, hidden_dim, use_forces=use_forces).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    #criterion = nn.MSELoss()
    criterion = loss_function_start_point

    # Train and validate
    train_losses = train_model_diffusion(
        model,
        train_loader, 
        optimizer, 
        criterion, 
        device, 
        num_epochs, 
        noiseadding_steps, 
        beta_start, 
        beta_end, 
        use_forces,
        noise_with_force, 
        max_grad_norm,
        add_gaussian_noise)
    
    val_loss = validate_model_diffusion(
        model, 
        val_loader,
        criterion, 
        device, 
        noiseadding_steps, 
        beta_start, 
        beta_end, 
        use_forces, 
        noise_with_force,
        add_gaussian_noise)
    
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
        

    # Plot predictions for the noise (x, y, z)
    plt.figure(figsize=(10, 5))
    for i, label in enumerate(['x', 'y', 'z']):
        if i != 1:
            continue
        plt.plot(actual_noise.detach().cpu()[0, :, i], label=f'Actual Noise {label}')
        plt.plot(predicted_noise.detach().cpu()[0, :, i], linestyle='-.', label=f'Predicted Noise {label}')
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
    num_denoising_steps = 1#noiseadding_steps  # this should be the same as the number of noise steps used in training


    for step in range (num_denoising_steps):
        # Predict the noise at the current step
        predicted_noise = model(denoised_trajectory, force) if use_forces else model(denoised_trajectory)

        # Subtract the predicted noise to denoise the trajectory
        denoised_trajectory = denoised_trajectory - predicted_noise


    # Detach and move the clean and denoised trajectories to CPU before plotting
    noisy_trajectory = val_dataset.denormalize(noisy_trajectory.detach().cpu(), "pos").numpy()
    clean_trajectory = val_dataset.denormalize(clean_trajectory.detach().cpu(), "pos_0").numpy()
    denoised_trajectory = val_dataset.denormalize(denoised_trajectory.detach().cpu(), "pos_0").numpy()

    # Compute mean absolute difference per axis
    mean_diff_x = np.mean(np.abs(clean_trajectory[:, :, 0] - denoised_trajectory[:, :, 0]))
    mean_diff_y = np.mean(np.abs(clean_trajectory[:, :, 1] - denoised_trajectory[:, :, 1]))
    mean_diff_z = np.mean(np.abs(clean_trajectory[:, :, 2] - denoised_trajectory[:, :, 2]))

    # Overall mean difference across all axes
    overall_mean_diff = np.mean(np.abs(clean_trajectory - denoised_trajectory))

    # Print results
    print(f"Mean Absolute Difference (x-axis): {mean_diff_x:.6f}")
    print(f"Mean Absolute Difference (y-axis): {mean_diff_y:.6f}")
    print(f"Mean Absolute Difference (z-axis): {mean_diff_z:.6f}")
    print(f"Overall Mean Absolute Difference: {overall_mean_diff:.6f}")



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
