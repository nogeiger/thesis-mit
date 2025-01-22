import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from utils import loss_function, add_noise


def train_model_diffusion(model, dataloader, optimizer, criterion, device, num_epochs, noiseadding_steps, beta_start, beta_end, use_forces=False):
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
            noisy_trajectory = add_noise(clean_trajectory, complete_noisy_trajectory, noiseadding_steps, beta_start, beta_end)
            
            # Compute the actual noise added
            actual_noise = noisy_trajectory - clean_trajectory  # The difference between noisy and clean trajectory
            #print(f"Actual noise at start: {actual_noise[:, 0, :]}")
            optimizer.zero_grad()

            # Predict the noise from the noisy trajectory (and forces if use_forces is True)
            if use_forces:
                predicted_noise = model(noisy_trajectory, force)
            else:
                predicted_noise = model(noisy_trajectory)

            #print(f"Predicted noise at start: {predicted_noise[:, 0, :]}")
            #print(f"Actual noise at start: {actual_noise[:, 0, :]}")

            # Calculate loss between predicted noise and actual noise
            loss = criterion(predicted_noise, actual_noise)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    return epoch_losses



def validate_model_diffusion(model, dataloader, criterion, device, max_noiseadding_steps, beta_start, beta_end, use_forces=False):
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
            noisy_trajectory = add_noise(clean_trajectory, noisy_trajectory, max_noiseadding_steps,beta_start, beta_end)

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