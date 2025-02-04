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
from train_val_test import train_model_diffusion, validate_model_diffusion, test_model
from utils import loss_function, loss_function_start_point, add_noise, calculate_max_noise_factor




def main():
    """
    Main function to execute the training and validation of the NoisePredictor model.
    """ 
    
    # Hyperparameters
    seq_length = 128 #seq len of data
    input_dim = seq_length * 3  # Flattened input dimension
    hidden_dim = 768 #hidden dim of the model
    batch_size = 64 #batch size
    num_epochs = 21#300 #number of epochs
    learning_rate = 3e-5 #learning rate
    noiseadding_steps = 50 # Number of steps to add noise
    use_forces = True  # Set this to True if you want to use forces as input to the model
    noise_with_force = False#True # Set this to True if you want to use forces as the noise
    #if force is used as noise, then force should not be used as input
    if noise_with_force:
            use_forces = False
    beta_start = 0.0001 #for the noise diffusion model
    beta_end = 0.0025 #for the noise diffusion model
    max_grad_norm=7.0 #max grad norm for gradient clipping 
    add_gaussian_noise = False#True # to add additional guassian noise

    # File path to the real data
    file_path = "Data/1D_diffusion_large_data"

    print("max noise factor per batch and step: ",calculate_max_noise_factor(beta_start,beta_end,noiseadding_steps))

    # Load real data
    data = load_robot_data(file_path, seq_length)
    
    # Compute per-axis normalization statistics
    stats = compute_statistics_per_axis(data)

    # Normalize data per axis
    normalized_data = normalize_data_per_axis(data, stats)

    # Define split ratios
    train_ratio = 0.7  
    val_ratio = 0.2  

    # Compute split indices
    total_size = len(normalized_data)
    train_split = int(total_size * train_ratio)
    val_split = train_split + int(total_size * val_ratio)

    # Split data
    train_data = normalized_data[:train_split]
    val_data = normalized_data[train_split:val_split]
    test_data = normalized_data[val_split:]

    # Create datasets with per-axis normalization
    train_dataset = ImpedanceDatasetDiffusion(train_data, stats)
    val_dataset = ImpedanceDatasetDiffusion(val_data, stats)
    test_dataset = ImpedanceDatasetDiffusion(test_data, stats)


    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print(f"Total sequences loaded: {len(test_dataset)} for training, {len(test_dataset)} for validation.")
    print(f"Total batches per epoch: {len(test_loader)} (Expected: {len(test_dataset) // 64})")


    # Model, optimizer, and loss function
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = NoisePredictorInitial(seq_length, hidden_dim, use_forces=use_forces).to(device)
    #model = NoisePredictorTransformer(seq_length, hidden_dim, use_forces=use_forces).to(device)
    #model = NoisePredictorLSTMWithAttention(seq_length, hidden_dim, use_forces=use_forces).to(device)
    #model = NoisePredictorLSTM(seq_length, hidden_dim, use_forces=use_forces).to(device)
    #model = NoisePredictorGRU(seq_length, hidden_dim, use_forces=use_forces).to(device)
    #model = NoisePredictorConvLSTM(seq_length, hidden_dim, use_forces=use_forces).to(device)
    #model = NoisePredictorConv1D(seq_length, hidden_dim, use_forces=use_forces).to(device)
    #model = NoisePredictorHybrid(seq_length, hidden_dim, use_forces=use_forces).to(device)
    #model = NoisePredictorTCN(seq_length, hidden_dim, use_forces=use_forces).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Model Parameters: {num_params}")
    #optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    #criterion = nn.MSELoss()
    #criterion = loss_function_start_point
    criterion=nn.SmoothL1Loss()

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
    
    num_denoising_steps=noiseadding_steps
    test_model(model, val_loader, val_dataset, device, use_forces, num_denoising_steps=num_denoising_steps, num_samples=5)


if __name__ == "__main__":
    main()
