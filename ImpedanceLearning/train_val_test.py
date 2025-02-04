import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import random
from utils import loss_function, add_noise
from torch.optim.lr_scheduler import ReduceLROnPlateau



def train_model_diffusion(model, traindataloader, valdataloader,optimizer, criterion, device, num_epochs, noiseadding_steps, beta_start, 
                          beta_end, use_forces=False, noise_with_force=False, max_grad_norm=7.0, add_gaussian_noise=False, save_interval = 20, 
                          save_path = "save_checkpoints",patience = 10):
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
    train_epoch_losses = []
    val_epoch_losses = []

    os.makedirs(save_path, exist_ok=True)  # Ensure save directory exists
    best_val_loss = float('inf')  # Track best validation loss
    early_stopping_counter = 0  # Count epochs since last improvement


    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        # Use tqdm to create a progress bar
        for batch_idx, (pos_0, pos, force) in enumerate(tqdm(traindataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=True)):

            # Move data to device
            clean_trajectory = pos_0.to(device)
            complete_noisy_trajectory = pos.to(device)
            force = force.to(device)

            # Dynamically add noise
            noisy_trajectory = add_noise(clean_trajectory, complete_noisy_trajectory, force, noiseadding_steps, beta_start, 
                                         beta_end, noise_with_force, add_gaussian_noise)
            
            # Compute the max noise (actual noise) based on the flag
            if noise_with_force:
                actual_noise = force  # Use force as the max noise
                use_forces = False #then force should not be used as input
            else:
                actual_noise = noisy_trajectory - clean_trajectory  # Default noise

            optimizer.zero_grad()

            # Predict the noise from the noisy trajectory
            if use_forces:
                predicted_noise = model(noisy_trajectory, force)
            else:
                predicted_noise = model(noisy_trajectory)

            # Calculate loss and perform backward pass
            loss = criterion(predicted_noise, actual_noise)
            loss.backward()

            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(traindataloader)
        train_epoch_losses.append(avg_train_loss)


        #Validation after each epoch
        model.eval()  # Switch to evaluation mode
        with torch.no_grad():
            val_loss = validate_model_diffusion(
                model, valdataloader, criterion, device, noiseadding_steps, beta_start, beta_end, 
                use_forces, noise_with_force, add_gaussian_noise
            )
        val_epoch_losses.append(val_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {val_loss:.4f}")


        # Early Stopping Check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0  # Reset counter
            best_model_path = os.path.join(save_path, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved at {best_model_path} after epoch {epoch+1}")
        else:
            early_stopping_counter += 1
            print(f"Early stopping patience: {early_stopping_counter}/{patience}")

        # If no improvement for `patience` epochs, stop training
        if early_stopping_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs. Restoring best model.")
            model.load_state_dict(torch.load(best_model_path))  # Restore best model
            break  # Exit training loop

        # Save model every 'save_interval' epochs
        if (epoch + 1) % save_interval == 0:
            checkpoint_path = os.path.join(save_path, f"model_epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model checkpoint saved at {checkpoint_path}")

    

    return train_epoch_losses, val_epoch_losses


def validate_model_diffusion(model, dataloader, criterion, device, max_noiseadding_steps, 
                             beta_start, beta_end, use_forces=False, noise_with_force=False, add_gaussian_noise=False):
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

    # Use tqdm to create a progress bar
    with torch.no_grad():
        for batch_idx, (pos_0, pos, force) in enumerate(tqdm(dataloader, desc="Validating", leave=True)):
            clean_trajectory = pos_0.to(device)
            noisy_trajectory = pos.to(device)
            force = force.to(device)

            # Dynamically add noise
            noisy_trajectory = add_noise(clean_trajectory, noisy_trajectory, force, max_noiseadding_steps, 
                                         beta_start, beta_end, noise_with_force, add_gaussian_noise)

            # Compute the max noise (actual noise) based on the flag
            if noise_with_force:
                actual_noise = force  # Use force as the max noise
                use_forces = False #then force should not be used as input
            else:
                actual_noise = noisy_trajectory - clean_trajectory  # Default: noise is the diff


            # Predict the noise from the noisy trajectory
            if use_forces:
                predicted_noise = model(noisy_trajectory, force)
            else:
                predicted_noise = model(noisy_trajectory)

            # Calculate loss
            loss = criterion(predicted_noise, actual_noise)
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Validation Loss: {avg_loss:.4f}")
    return avg_loss


def test_model(model, val_loader, val_dataset, device, use_forces, num_denoising_steps=1, num_samples=5):
    """
    Function to evaluate the model by predicting noise, performing iterative denoising,
    and visualizing the results.

    Args:
        model (torch.nn.Module): The trained noise predictor model.
        val_loader (DataLoader): DataLoader for the validation set.
        val_dataset (Dataset): Validation dataset (for denormalization).
        device (torch.device): Device (CPU/GPU).
        use_forces (bool): Whether forces are used as an input to the model.
        num_denoising_steps (int): Number of denoising steps.
        num_samples (int): Number of samples to visualize.
    """

    model.eval()  # Set model to evaluation mode

    # Convert validation dataset into a list (ensures correct indexing)
    val_data = list(val_loader.dataset)

    # Randomly select `num_samples` different indices from the validation dataset
    sample_indices = random.sample(range(len(val_data)), num_samples)

    # Initialize lists for mean absolute differences
    mean_diffs_x, mean_diffs_y, mean_diffs_z, overall_mean_diffs = [], [], [], []
    stiffness_values = []

    for sample_idx, idx in enumerate(sample_indices):
        # Fetch a **random sample** instead of always using the first batch
        clean_trajectory, noisy_trajectory, force = val_data[idx]

        # Move data to the correct device
        clean_trajectory = clean_trajectory.unsqueeze(0).to(device)  # Add batch dimension
        noisy_trajectory = noisy_trajectory.unsqueeze(0).to(device)
        force = force.unsqueeze(0).to(device)

        # Start iterative denoising
        denoised_trajectory = noisy_trajectory.clone()
        for _ in range(num_denoising_steps):
            predicted_noise = model(denoised_trajectory, force) if use_forces else model(denoised_trajectory)
            denoised_trajectory = denoised_trajectory - predicted_noise  # Remove noise iteratively

        # Denormalize trajectories
        noisy_trajectory_np = val_dataset.denormalize(noisy_trajectory.detach().cpu(), "pos").numpy()
        clean_trajectory_np = val_dataset.denormalize(clean_trajectory.detach().cpu(), "pos_0").numpy()
        denoised_trajectory_np = val_dataset.denormalize(denoised_trajectory.detach().cpu(), "pos_0").numpy()
        force_np = val_dataset.denormalize(force.detach().cpu(), "force").numpy()


        # Compute stiffness K = F / (x - x_0)
        displacement = noisy_trajectory_np - clean_trajectory_np  # (x - x_0)
        stiffness = np.divide(force_np, displacement, out=np.zeros_like(force_np), where=displacement != 0)

        # Compute mean stiffness for the sample
        mean_stiffness = np.mean(stiffness)
        stiffness_values.append(mean_stiffness)

        # Compute mean absolute differences
        mean_diff_x = np.mean(np.abs(clean_trajectory_np[:, :, 0] - noisy_trajectory_np[:, :, 0]))
        mean_diff_y = np.mean(np.abs(clean_trajectory_np[:, :, 1] - noisy_trajectory_np[:, :, 1]))
        mean_diff_z = np.mean(np.abs(clean_trajectory_np[:, :, 2] - noisy_trajectory_np[:, :, 2]))
        overall_mean_diff = np.mean(np.abs(clean_trajectory_np - noisy_trajectory_np))

        # Compute mean absolute differences
        mean_diff_x = np.mean(np.abs(clean_trajectory_np[:, :, 0] - denoised_trajectory_np[:, :, 0]))
        mean_diff_y = np.mean(np.abs(clean_trajectory_np[:, :, 1] - denoised_trajectory_np[:, :, 1]))
        mean_diff_z = np.mean(np.abs(clean_trajectory_np[:, :, 2] - denoised_trajectory_np[:, :, 2]))
        overall_mean_diff = np.mean(np.abs(clean_trajectory_np - denoised_trajectory_np))

        mean_diffs_x.append(mean_diff_x)
        mean_diffs_y.append(mean_diff_y)
        mean_diffs_z.append(mean_diff_z)
        overall_mean_diffs.append(overall_mean_diff)

        # Create a separate figure for each sample
        fig, ax_traj = plt.subplots(1, 1, figsize=(10, 5))

        # Plot clean vs denoised trajectory (Y-axis only)
        ax_traj.plot(clean_trajectory_np[0, :, 1], label='Clean', alpha=0.7)
        ax_traj.plot(denoised_trajectory_np[0, :, 1], linestyle='--', label='Denoised', alpha=0.7)
        ax_traj.set_xlabel('Time Step')
        ax_traj.set_ylabel('Position')
        ax_traj.set_title(f'Clean vs Denoised Trajectory - Sample {sample_idx+1}')
        ax_traj.legend()

        # Show plots without blocking execution
        plt.show(block=False)

    # Print Mean Absolute Differences
    print(f"\nMean Absolute Differences Across {num_samples} Samples:")
    print(f"X-axis: {np.mean(mean_diffs_x):.6f}")
    print(f"Y-axis: {np.mean(mean_diffs_y):.6f}")
    print(f"Z-axis: {np.mean(mean_diffs_z):.6f}")
    print(f"Overall: {np.mean(overall_mean_diffs):.6f}")

    # Print Mean Stiffness Values
    print(f"\nMean Stiffness Across {num_samples} Samples: {np.mean(stiffness_values):.6f}")


    # Keep plots open until the user closes them
    plt.pause(0.1)
    input("Press Enter to close all plots and continue...")
    plt.close('all')