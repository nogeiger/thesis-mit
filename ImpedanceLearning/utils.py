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

def add_noise(clean_pos, noisy_pos, clean_u, noisy_u, clean_theta, noisy_theta, force, moment,
              max_noiseadding_steps, beta_start, beta_end, noise_with_force=False, add_gaussian_noise=False):
    
    """
    Dynamically adds noise to a clean 3D trajectory based on the actual noise between the clean and noisy trajectories,
    following a diffusion model schedule.

    Args:
        clean_trajectory (torch.Tensor): The clean trajectory with shape [seq_length, 3].
        noisy_trajectory (torch.Tensor): The noisy trajectory with shape [seq_length, 3].
        clean_u (torch.Tensor): The clean input force with shape [seq_length, 3].
        noisy_u (torch.Tensor): The noisy input force with shape [seq_length, 3].
        clean_theta (torch.Tensor): The clean input angle with shape [seq_length, 1].
        noisy_theta (torch.Tensor): The noisy input angle with shape [seq_length, 1].
        force (torch.Tensor): The force with shape [seq_length, 3].
        moment (torch.Tensor): The moment with shape [seq_length, 3].
        max_noiseadding_steps (int): Maximum number of steps to iteratively add noise.
        beta_start (float): Initial value of noise scale.
        beta_end (float): Final value of noise scale.

    Returns:
        torch.Tensor: Noisy trajectory with shape [seq_length, 3].
    """
    # Calculate the actual noise (difference between clean and noisy) - if scale_noise_with_force is True, use force as the noise
    if noise_with_force:
        actual_noise_pos = force
    # otherwise use difference between clean and noisy trajectory
    else:
        actual_noise_pos = noisy_pos - clean_pos

    actual_noise_theta = noisy_theta - clean_theta

    #use coisne similiarity to calc alpha which is seen as the noise here for u (displacement vector)
    # Compute cosine similarity directly (dot product since length is 1)
    cos_alpha = torch.sum(noisy_u * clean_u, dim=-1)  # Shape: [64, 32]
    # Ensure values are clipped to [-1, 1] to avoid numerical errors in arccos
    cos_alpha = torch.clamp(cos_alpha, -1.0, 1.0)
    # Compute angle alpha in radians
    actual_noise_alpha = torch.acos(cos_alpha)  #full possible noise in angle between noisy and clean u


    # Optionally add Gaussian noise to the actual noise
    if add_gaussian_noise:
        # Generate Gaussian noise
        gaussian_noise_pos = torch.randn_like(actual_noise_pos)
        gaussian_noise_alpha = torch.randn_like(actual_noise_alpha)
        gaussian_noise_theta = torch.randn_like(actual_noise_theta)
        
        # Normalize the Gaussian noise to have unit norm
        norm_pos = torch.norm(gaussian_noise_pos)
        norm_u_alpha = torch.norm(gaussian_noise_u_alpha)
        norm_theta = torch.norm(gaussian_noise_theta)

        if norm_pos > 0:  # Avoid division by zero
            gaussian_noise_pos = gaussian_noise_pos / norm_pos
        if norm_u_alpha > 0:  # Avoid division by zero
            gaussian_noise_u_alpha = gaussian_noise_u_alpha / norm_u_alpha
        if norm_theta > 0:  # Avoid division by zero
            gaussian_noise_theta = gaussian_noise_theta / norm_theta
        
        #scale the normalized noise (to match force/trajectorz noise scale / to not have noise from gaussian which is way higher then the actual noise)
        gaussian_noise_pos = gaussian_noise_pos * torch.norm(actual_noise_pos)  # Scale to match actual_noise norm
        gaussian_noise_u_alpha = gaussian_noise_alpha * torch.norm(actual_noise_alpha)  # Scale to match actual_noise norm
        gaussian_noise_theta = gaussian_noise_theta * torch.norm(actual_noise_theta)  # Scale to match actual_noise norm
        
        # Add the normalized Gaussian noise to the actual noise
        actual_noise_pos += gaussian_noise_pos
        actual_noise_alpha += gaussian_noise_alpha
        actual_noise_theta += gaussian_noise_theta

    # Randomly choose the number of noise adding steps between 1 and max_noiseadding_steps
    noiseadding_steps = torch.randint(1, max_noiseadding_steps + 1, (1,)).item()

    # Initialize the noisy trajectory as the clean trajectory
    noisy_pos_output = clean_pos.clone()
    noisy_u_output = clean_u.clone()
    noisy_theta_output = clean_theta.clone()

    # Linear schedule for noise scale (beta values)
    beta_values = torch.linspace(beta_start, beta_end, noiseadding_steps)  # Linearly spaced beta values
    alpha_values = 1 - beta_values  # Compute α from β
    alpha_bar = torch.cumprod(alpha_values, dim=0)  # Compute cumulative product α̅

    # Sample a random timestep t
    t = torch.randint(0, noiseadding_steps, (1,)).item()  # Select a random diffusion step

    # Compute the noisy trajectory at timestep t
    noisy_pos_output = torch.sqrt(alpha_bar[t]) * clean_pos + torch.sqrt(1 - alpha_bar[t]) * actual_noise_pos
    noisy_theta_output = torch.sqrt(alpha_bar[t]) * clean_theta + torch.sqrt(1 - alpha_bar[t]) * actual_noise_theta

    # Compute noisy rotation angle at timestep t
    noisy_alpha =  torch.sqrt(1 - alpha_bar[t]) * actual_noise_alpha# + torch.sqrt(alpha_bar[t]) * 0 -  cause for clean alpha the angle is 0
    
    #calculate the noisy_u based on the clean_u and the add noisy_alpha and expand to [64,32,-1]
    #calculate the prependicular component v to reconstruct the noisy_u
    v = noisy_u - (torch.sum(clean_u * noisy_u, dim=-1, keepdim=True) * clean_u)
    v = v / (torch.norm(v, dim=-1, keepdim=True) + 1e-8)  # Normalize v, avoid division by zero
    # Compute noisy_u_output using rotation formula
    noisy_u_output = torch.cos(noisy_alpha).unsqueeze(-1) * clean_u + torch.sin(noisy_alpha).unsqueeze(-1) * v

    # Noise scale
    noise_scale = 1 / torch.sqrt(alpha_bar[t])

    return noisy_pos_output, noisy_u_output, noisy_alpha, noisy_theta_output, noise_scale


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




