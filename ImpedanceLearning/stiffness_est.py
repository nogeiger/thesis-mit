import numpy as np
import scipy.optimize as opt
import pandas as pd
import os
import torch
import matplotlib.pyplot as plt

import numpy as np
from scipy.optimize import least_squares

def estimate_stiffness_per_sequence(force_np, moment_np, clean_pos_np, denoised_pos_np, clean_q_np, denoised_q_np, time_array):
    """
    Estimates translational and rotational stiffness per sequence using Nonlinear Least Squares (NLS).
    
    Args:
        force_np (np.array): External force array (T, 3).
        moment_np (np.array): External moment array (T, 3).
        clean_pos_np (np.array): Ground truth position array (T, 3).
        denoised_pos_np (np.array): Denoised position array (T, 3).
        clean_q_np (np.array): Ground truth quaternion array (T, 4).
        denoised_q_np (np.array): Denoised quaternion array (T, 4).
        time_array (np.array): Time array (T,).
        
    Returns:
        k_t_estimated (np.array): Estimated translation stiffness (3,).
        k_r_estimated (np.array): Estimated rotation stiffness (3,).
    """
    denoised_pos_np = denoised_pos_np.squeeze(0)
    clean_pos_np = clean_pos_np.squeeze(0)  # Remove batch dim if necessary
    clean_q_np = clean_q_np.squeeze(0)  # Remove batch dim if necessary
    denoised_q_np = denoised_q_np.squeeze(0)  # Remove batch dim if necessary
    force_np = force_np.squeeze(0)  # Remove batch dim if necessary
    moment_np = moment_np.squeeze(0)  # Remove batch dim if necessary


    ### 1. Compute Displacement (Δp) and Velocity (ẋ)
    delta_p = clean_pos_np - denoised_pos_np  # Position displacement

    # Ensure correct time_array length
    if len(time_array) != denoised_pos_np.shape[0]:
        time_array = np.linspace(0, (denoised_pos_np.shape[0]-1) * 0.005, denoised_pos_np.shape[0])  # Adjust time step if needed

    time_array = time_array.squeeze()  # Ensure it's 1D


    velocity = np.gradient(denoised_pos_np, time_array, axis=0)  # Velocity estimate

    ### 2. Compute Rotation Axis (u₀) and Angle (θ)
    u_0 = clean_q_np[:, 1:]  # Extract vector part of quaternion (x, y, z)
    u_0 = u_0 / np.linalg.norm(u_0, axis=1, keepdims=True)  # Normalize
    theta = 2 * np.arccos(np.clip(clean_q_np[:, 0], -1.0, 1.0))  # Rotation angle

    ### 3. Define Residual Functions for NLS

    def translation_residuals(k_t):
        """ Residual function for translational stiffness. """
        alpha_t = 0.5  # Assumed damping parameter (adjustable)
        predicted_force = np.diag(k_t) @ (delta_p.T - alpha_t * velocity.T)
        return (force_np.T - predicted_force).flatten()

    def rotation_residuals(k_r):
        """ Residual function for rotational stiffness. """
        alpha_r = 0.5  # Assumed damping parameter (adjustable)
        predicted_moment = np.diag(k_r) @ (u_0.T * theta - alpha_r * moment_np.T)
        return (moment_np.T - predicted_moment).flatten()

    ### 4. Solve for Stiffness using NLS

    # Initial guess for stiffness
    k_t_init = np.array([650, 650, 650])  # Initial guess for translation stiffness
    k_r_init = np.array([10, 10, 10])  # Initial guess for rotational stiffness

    # Solve NLS for translational stiffness
    k_t_solution = least_squares(translation_residuals, k_t_init, bounds=(1e-6, 5000))
    k_t_estimated = k_t_solution.x  # Extract estimated values

    # Solve NLS for rotational stiffness
    k_r_solution = least_squares(rotation_residuals, k_r_init, bounds=(1e-6, 500))
    k_r_estimated = k_r_solution.x  # Extract estimated values

    return k_t_estimated, k_r_estimated
