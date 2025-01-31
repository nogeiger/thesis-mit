import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def loss_function(predicted_noise, actual_noise):
    """
    Computes the mean squared error between the predicted and actual noise.
    """
    return nn.MSELoss()(predicted_noise, actual_noise)

def loss_function_start_point(predicted_noise, actual_noise, weight_start_point=0):
    """
    Modified loss function to enforce alignment at the first timestep.
    """
    mse_loss = nn.MSELoss()(predicted_noise, actual_noise)
    
    # Add a penalty term for misalignment at the start point
    start_point_loss = nn.MSELoss()(predicted_noise[:, 0, :], actual_noise[:, 0, :])
    
    # Combine losses (adjust weight of the penalty as needed, e.g., 0.1)
    total_loss = mse_loss + weight_start_point * start_point_loss
    return total_loss

def add_noise(clean_trajectory, noisy_trajectory, force, max_noiseadding_steps, 
              beta_start=0.8, beta_end=0.1, noise_with_force=False, add_gaussian_noise=False):

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
    # Calculate the actual noise (difference between clean and noisy) - if scale_noise_with_force is True, use force as the noise
    if noise_with_force:
        actual_noise = force
    # otherwise use difference between clean and noisy trajectory
    else:
        actual_noise = noisy_trajectory - clean_trajectory

    # Optionally add Gaussian noise to the actual noise
    if add_gaussian_noise:
        # Generate Gaussian noise
        gaussian_noise = torch.randn_like(actual_noise)
        
        # Normalize the Gaussian noise to have unit norm
        norm = torch.norm(gaussian_noise)
        if norm > 0:  # Avoid division by zero
            gaussian_noise = gaussian_noise / norm
        
        #scale the normalized noise (to match force/trajectorz noise scale / to not have noise from gaussian which is way higher then the actual noise)
        gaussian_noise = gaussian_noise * torch.norm(actual_noise)  # Scale to match actual_noise norm
        
        # Add the normalized Gaussian noise to the actual noise
        actual_noise += gaussian_noise

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
        #print("wurzel beta value: ", torch.sqrt(beta))
        noise_to_add = actual_noise * torch.sqrt(beta)
        noisy_trajectory_output += noise_to_add

    
    return noisy_trajectory_output


def calculate_max_noise_factor(beta_start, beta_end, max_noiseadding_steps):
    """
    Calculate the maximum factor of noise that can be added based on the noise schedule.

    Args:
        beta_start (float): Initial value of noise scale.
        beta_end (float): Final value of noise scale.
        max_noiseadding_steps (int): Maximum number of steps to iteratively add noise.

    Returns:
        float: The maximum noise factor that can be added.
    """
    import torch

    # Linear schedule for beta values
    beta_values = torch.linspace(beta_start, beta_end, max_noiseadding_steps)

    # Compute the sum of sqrt(beta) for all steps
    max_noise_factor = torch.sum(torch.sqrt(beta_values)).item()

    return max_noise_factor

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

    # Keep plots open until the user closes them
    plt.pause(0.1)
    input("Press Enter to close all plots and continue...")
    plt.close('all')


