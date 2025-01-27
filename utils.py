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

def add_noise(clean_trajectory, noisy_trajectory, force, max_noiseadding_steps, beta_start=0.8, beta_end=0.1, noise_with_force=False):

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


