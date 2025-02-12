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
              beta_start, beta_end, noise_with_force=False, add_gaussian_noise=False):

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
    beta_values = torch.linspace(beta_start, beta_end, noiseadding_steps)  # Linearly spaced beta values
    alpha_values = 1 - beta_values  # Compute α from β
    alpha_bar = torch.cumprod(alpha_values, dim=0)  # Compute cumulative product α̅

    # Sample a random timestep t
    t = torch.randint(0, noiseadding_steps, (1,)).item()  # Select a random diffusion step

    # Compute the noisy trajectory at timestep t
    noisy_trajectory_output = torch.sqrt(alpha_bar[t]) * clean_trajectory + torch.sqrt(1 - alpha_bar[t]) * actual_noise
    #print(torch.sqrt(1 - alpha_bar[t]))
    noise_scale = 1 / torch.sqrt(alpha_bar[t])
    
    return noisy_trajectory_output, noise_scale, t


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




