import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial.transform import Rotation as R

def quaternion_inverse(q):
    """Computes the inverse of a unit quaternion q = (w, x, y, z)."""
    q_inv = q.clone()
    q_inv[..., 1:] *= -1  # Negate x, y, z components
    return q_inv

def quaternion_multiply(q1, q2):
    """Computes the Hamilton product of two quaternions q1 and q2."""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack((w, x, y, z), dim=-1)

def slerp(q0, q1, t):
    """Performs Spherical Linear Interpolation (SLERP) between two quaternions."""
    # Compute dot product (cosine of the angle)
    dot_product = torch.sum(q0 * q1, dim=-1, keepdim=True)

    # Clamp to prevent numerical errors
    dot_product = torch.clamp(dot_product, -1.0, 1.0)

    # Compute theta (angle between q0 and q1)
    theta = torch.acos(dot_product)

    # Compute sin(theta) to avoid division by zero
    sin_theta = torch.sin(theta)

    # Handle small angles by using linear interpolation
    small_angle = sin_theta < 1e-6
    s1 = torch.sin((1 - t) * theta) / (sin_theta + small_angle.float())  # Avoid division by zero
    s2 = torch.sin(t * theta) / (sin_theta + small_angle.float())

    # SLERP interpolation
    q_interp = s1 * q0 + s2 * q1

    # Normalize the result to keep it a valid unit quaternion
    q_interp = q_interp / torch.norm(q_interp, dim=-1, keepdim=True)

    return q_interp



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

def add_noise(clean_pos, noisy_pos, clean_q, noisy_q, force, moment,
              max_noiseadding_steps, beta_start, beta_end, noise_with_force=False, add_gaussian_noise=False):
    
    """
    Dynamically adds noise to a clean 3D trajectory based on the actual noise between the clean and noisy trajectories,
    following a diffusion model schedule.

    Args:
        clean_trajectory (torch.Tensor): The clean trajectory with shape [seq_length, 3].
        noisy_trajectory (torch.Tensor): The noisy trajectory with shape [seq_length, 3].
        clean_q (torch.Tensor): The clean input quaternion with shape [seq_length, 3, 4].
        noisy_q (torch.Tensor): The noisy input quaternion with shape [seq_length, 3, 4].
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

    # Optionally add Gaussian noise to the actual noise
    if add_gaussian_noise:
        #add gaussian noise to pos
        # Generate Gaussian noise
        gaussian_noise_pos = torch.randn_like(actual_noise_pos)
        # Normalize the Gaussian noise to have unit norm
        norm_pos = torch.norm(gaussian_noise_pos)
        if norm_pos > 0:  # Avoid division by zero
            gaussian_noise_pos = gaussian_noise_pos / norm_pos
        #scale the normalized noise (to match force/trajectorz noise scale / to not have noise from gaussian which is way higher then the actual noise)
        gaussian_noise_pos = gaussian_noise_pos * torch.norm(actual_noise_pos)  # Scale to match actual_noise norm
        # Add the normalized Gaussian noise to the actual noise
        actual_noise_pos += gaussian_noise_pos

        #add gaussian noise to q_noisy which is the max noise to which we "slerp"/interpolate later with factor alpha
        # Generate a small random axis-angle perturbation (Gaussian noise)
        random_axis = torch.randn_like(noisy_q[..., 1:])  # Exclude w-component
        random_axis = random_axis / torch.norm(random_axis, dim=-1, keepdim=True)  # Normalize to unit vector
        random_angle = torch.randn(noisy_q.shape[:-1], device=noisy_q.device) * 0.1  # Small perturbation angle

        # Compute the magnitude of noise (like norm in pos)
        dot_product = torch.sum(clean_q * noisy_q, dim=-1, keepdim=True)
        dot_product = torch.clamp(dot_product, -1.0, 1.0)  # Avoid numerical errors
        theta = 2 * torch.acos(torch.abs(dot_product))  # Angular difference

        # Scale noise by quaternion difference (like norm in pos)
        scaling_factor = (theta / torch.pi).unsqueeze(-1)  # Ensure it has shape [batch_size, 1]
        scaled_angle = random_angle * scaling_factor  # Scale noise by rotation difference

        # Convert axis-angle perturbation to a noise quaternion
        sin_half_angle = torch.sin(scaled_angle / 2).unsqueeze(-1)
        cos_half_angle = torch.cos(scaled_angle / 2).unsqueeze(-1)
        gaussian_noise_q = torch.cat([cos_half_angle, sin_half_angle * random_axis], dim=-1)

        # Apply the Gaussian noise to noisy_q
        noisy_q = quaternion_multiply(gaussian_noise_q, noisy_q)



    # Randomly choose the number of noise adding steps between 1 and max_noiseadding_steps
    noiseadding_steps = torch.randint(1, max_noiseadding_steps + 1, (1,)).item()

    # Initialize the noisy trajectory as the clean trajectory
    noisy_pos_output = clean_pos.clone()
    noisy_q_output = clean_q.clone()

    # Linear schedule for noise scale (beta values)
    beta_values = torch.linspace(beta_start, beta_end, noiseadding_steps)  # Linearly spaced beta values
    alpha_values = 1 - beta_values  # Compute α from β
    alpha_bar = torch.cumprod(alpha_values, dim=0)  # Compute cumulative product α̅

    # Sample a random timestep t
    t = torch.randint(0, noiseadding_steps, (1,)).item()  # Select a random diffusion step

    # Compute the noisy trajectory at timestep t
    noisy_pos_output = torch.sqrt(alpha_bar[t]) * clean_pos + torch.sqrt(1 - alpha_bar[t]) * actual_noise_pos
    
    #Use Slerp for q and q_0 (both unit quaternions) to calc the output quaternion which is in between with the noise scheduler as t
    # SLERP(q0, q1, t) = (sin((1-t) * omega) / sin(omega)) * q0 + (sin(t * omega) / sin(omega)) * q1
    #use alpha_bar as t to interpolate the noisy quaternion output
    noisy_q_output = slerp(clean_q, noisy_q, torch.sqrt(alpha_bar[t]))

    # Noise scale
    noise_scale = 1 / torch.sqrt(alpha_bar[t])

    return noisy_pos_output, noisy_q_output, noise_scale


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




