import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial.transform import Rotation as R
import torch.nn.functional as F

def quaternion_inverse(q):
    """Computes the inverse of a quaternion q = (w, x, y, z)."""
    norm_sq = torch.sum(q**2, dim=-1, keepdim=True)  # Compute |q|^2
    q_conjugate = torch.cat([q[..., :1], -q[..., 1:]], dim=-1)  # Efficient conjugate calculation
    return q_conjugate / norm_sq  # Normalize by |q|^2

def quaternion_multiply(q1, q2):
    """Computes the Hamilton product of two quaternions q1 and q2."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    return torch.stack((
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    ), dim=-1)

def smooth_quaternions_slerp(q_series, window_size=5, smoothing_factor=0.5):
    """
    Smooths a quaternion time series using SLERP interpolation over fixed windows.

    Args:
        q_series (torch.Tensor): Quaternion time series of shape [T, 4] (T = timesteps).
        window_size (int): Number of timesteps in each smoothing window.
        smoothing_factor (float): Weight factor (0.0 = no smoothing, 1.0 = full smoothing).

    Returns:
        torch.Tensor: Smoothed quaternion time series of shape [T, 4].
    """
    T = q_series.shape[0]

    for start in range(0, T - window_size + 1, window_size):  # Process in chunks
        end = min(start + window_size, T)

        # Compute SLERP between previous and next quaternion for the whole window at once
        q_prev = q_series[start:end-2]  # All except last two
        q_next = q_series[start+2:end]  # All except first two

        q_series[start+1:end-1] = slerp(q_prev, q_next, smoothing_factor)  # Apply SLERP

    return q_series

def slerp(q0, q1, t):
    """Performs Spherical Linear Interpolation (SLERP) between two quaternions."""
    # Compute dot product (cosine of the angle)
    dot_product = torch.sum(q0 * q1, dim=-1, keepdim=True).clamp(-1.0, 1.0)

    # Compute theta (angle between q0 and q1)
    theta = torch.acos(dot_product)
    sin_theta = torch.sin(theta)

    # Handle small angles with linear interpolation
    near_zero = sin_theta < 1e-6
    s1 = torch.where(near_zero, 1.0 - t, torch.sin((1 - t) * theta) / sin_theta)
    s2 = torch.where(near_zero, t, torch.sin(t * theta) / sin_theta)

    # Compute interpolated quaternion
    q_interp = s1 * q0 + s2 * q1

    # Normalize the quaternion to ensure it's a valid unit quaternion
    return q_interp / torch.norm(q_interp, dim=-1, keepdim=True)


# Function to extract rotation axis (u) from a quaternion
def quaternion_to_axis(q):
    """
    Extracts the rotation axis from a unit quaternion.

    Args:
        q (numpy array): Quaternion array of shape (sequence_length, 4),
                         where each quaternion is (w, x, y, z).

    Returns:
        numpy array: Rotation axis (unit vector), shape (sequence_length, 3).
    """
    w = q[:, 0]  # Extract w component
    xyz = q[:, 1:]  # Extract (x, y, z) components

    # Compute sin(theta/2) with numerical stability
    sin_half_theta = np.sqrt(np.clip(1 - w**2, 0.0, None))  # Use clip for stability

    # Avoid division by zero using np.where
    u = np.where(sin_half_theta[:, np.newaxis] > 1e-6, xyz / sin_half_theta[:, np.newaxis], np.zeros_like(xyz))

    return u

def quaternion_loss(pred_q, target_q, lambda_unit=0.3):
    """
    Computes quaternion loss with geodesic distance, theta difference (angle wrap-around fixed), and alpha loss (stable).

    Args:
        pred_q (torch.Tensor): Predicted quaternion (..., 4).
        target_q (torch.Tensor): Ground truth quaternion (..., 4).
        lambda_unit (float): Weight for unit norm constraint.

    Returns:
        torch.Tensor: Combined loss value.
    """
    eps = 1e-8  # Small constant for numerical stability

    # Normalize quaternions
    target_q = F.normalize(target_q, dim=-1)
    pred_q = F.normalize(pred_q, dim=-1)

    # Ensure correct quaternion orientation
    dot_product = torch.sum(pred_q * target_q, dim=-1, keepdim=True)
    pred_q = torch.where(dot_product < 0, -pred_q, pred_q)  # Flip if necessary
    dot_product = dot_product.abs().squeeze(-1)  # Ensure positive values

    # 1. Geodesic quaternion loss
    loss_q = torch.mean((1 - dot_product) ** 2)  

    # 2. Theta (rotation angle) difference loss (fixed wrap-around)
    theta_pred = 2 * torch.acos(torch.clamp(pred_q[..., 0], -1.0, 1.0))
    theta_target = 2 * torch.acos(torch.clamp(target_q[..., 0], -1.0, 1.0))

    theta_diff = torch.abs(theta_pred - theta_target)
    loss_theta = torch.mean(torch.minimum(theta_diff, 2 * torch.pi - theta_diff) ** 2)

    # 3. Alpha (axis difference) loss
    axis_pred = F.normalize(pred_q[..., 1:], dim=-1)
    axis_target = F.normalize(target_q[..., 1:], dim=-1)

    dot_product_axis = torch.sum(axis_pred * axis_target, dim=-1).clamp(-0.999999, 0.999999)
    alpha_error = torch.atan2(torch.sqrt(1 - dot_product_axis**2 + eps), dot_product_axis)
    loss_alpha = torch.mean(alpha_error ** 2)

    # 4. Unit norm constraint
    unit_loss = torch.mean((torch.norm(pred_q, dim=-1) - 1) ** 2)

    # Final weighted loss
    return loss_q + 2* loss_theta + lambda_unit * unit_loss + 10 * loss_alpha

  

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
    Adds noise to a clean 3D trajectory using a diffusion model schedule.

    Args:
        clean_pos (torch.Tensor): Clean trajectory, shape [seq_length, 3].
        noisy_pos (torch.Tensor): Noisy trajectory, shape [seq_length, 3].
        clean_q (torch.Tensor): Clean quaternion, shape [seq_length, 4].
        noisy_q (torch.Tensor): Noisy quaternion, shape [seq_length, 4].
        force (torch.Tensor): Force values, shape [seq_length, 3].
        moment (torch.Tensor): Moment values, shape [seq_length, 3].
        max_noiseadding_steps (int): Max number of noise-adding steps.
        beta_start (float): Initial noise scale.
        beta_end (float): Final noise scale.
        noise_with_force (bool): If True, use force as noise.
        add_gaussian_noise (bool): If True, add Gaussian noise.

    Returns:
        tuple: Noisy position, noisy quaternion, noise scale.
    """
    # Compute actual noise in a single operation
    actual_noise_pos = force if noise_with_force else noisy_pos - clean_pos

    if add_gaussian_noise:
        # Generate and normalize Gaussian noise for position in one step
        gaussian_noise_pos = torch.randn_like(actual_noise_pos)
        gaussian_noise_pos /= torch.norm(gaussian_noise_pos, dim=-1, keepdim=True).clamp(min=1e-6)
        gaussian_noise_pos *= torch.norm(actual_noise_pos, dim=-1, keepdim=True)
        actual_noise_pos += gaussian_noise_pos  # In-place addition

        # Generate random axis-angle perturbation for quaternion noise
        random_axis = torch.randn_like(noisy_q[..., 1:])
        random_axis /= torch.norm(random_axis, dim=-1, keepdim=True).clamp(min=1e-6)
        random_angle = torch.randn(noisy_q.shape[:-1], device=noisy_q.device) * 0.1

        # Compute quaternion noise scaling directly
        dot_product = torch.sum(clean_q * noisy_q, dim=-1, keepdim=True).clamp(-1.0, 1.0)
        theta = 2 * torch.acos(dot_product.abs())
        scaling_factor = (theta / torch.pi).unsqueeze(-1)
        scaled_angle = random_angle * scaling_factor

        # Convert axis-angle perturbation to a quaternion in a single operation
        sin_half_angle, cos_half_angle = torch.sin(scaled_angle / 2), torch.cos(scaled_angle / 2)
        gaussian_noise_q = torch.cat([cos_half_angle, sin_half_angle * random_axis], dim=-1)

        # Apply Gaussian noise to quaternion using in-place multiplication
        noisy_q = quaternion_multiply(gaussian_noise_q, noisy_q)

    # Efficient random step selection
    noiseadding_steps = torch.randint(1, max_noiseadding_steps + 1, ())

    # Vectorized noise schedule computation
    beta_values = torch.linspace(beta_start, beta_end, noiseadding_steps, device=clean_pos.device)
    alpha_bar = torch.cumprod(1 - beta_values, dim=0)

    # Select a random timestep t efficiently
    t = torch.randint(0, noiseadding_steps, ())

    sqrt_alpha_bar_t = torch.sqrt(alpha_bar[t])

    # Compute noisy position in one operation
    noisy_pos_output = sqrt_alpha_bar_t * clean_pos + torch.sqrt(1 - alpha_bar[t]) * actual_noise_pos

    # SLERP interpolation for quaternion
    noisy_q_output = slerp(clean_q, noisy_q, sqrt_alpha_bar_t)

    # Compute noise scale directly
    noise_scale = 1 / sqrt_alpha_bar_t

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




