import numpy as np
import scipy.optimize as opt
import pandas as pd
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
import numpy as np

def estimate_stiffness_per_sequence(force_np, moment_np, zero_force_pos_np, observed_pos_np, 
                                    zero_force_q_np, observed_q_np, time_array, Lambda_t, Lambda_r):
    """
    Estimates translational and rotational stiffness per timestep using Nonlinear Least Squares (NLS).
    
    Returns:
        k_t_estimated_over_time (np.array): Estimated translational stiffness over time (T, 3).
        k_r_estimated_over_time (np.array): Estimated rotational stiffness over time (T, 3).
    """
    
    def compute_alpha(Lambda, K, damping_factor=0.7):
        """Computes damping parameter α using mass-inertia properties."""
        U, Sigma, _ = np.linalg.svd(Lambda)  # Eigen decomposition
        Sigma = np.diag(np.sqrt(np.maximum(Sigma, 1e-6)))  # Avoid NaNs
        sqrt_Lambda = U @ Sigma @ U.T

        # Ensure only negative values in K are replaced, without affecting positive values
        K[K < 0] = 1e-6  # Replace only negative values with 1e-6
        sqrt_K = np.diag(np.sqrt(K))

        D = np.eye(3) * damping_factor
        b_t = sqrt_Lambda @ D @ sqrt_K + sqrt_K @ D @ sqrt_Lambda
        alpha = (2 * np.trace(b_t)) / np.sum(K)
        return alpha

    # Compute Displacement (Δp) and Velocity (ẋ)
    delta_p = zero_force_pos_np - observed_pos_np  # Position displacement
    velocity = np.gradient(observed_pos_np, time_array, axis=0)  # Velocity estimate

    # Compute Rotation Axis (u₀) and Angle (θ)
    u_0 = zero_force_q_np[:, 1:]  # Extract vector part of quaternion
    u_0 = u_0 / np.linalg.norm(u_0, axis=1, keepdims=True)  # Normalize
    theta =  2* np.arccos(np.clip(zero_force_q_np[:, 0], -1.0, 1.0))  # Rotation angle

    # Store stiffness values for each timestep
    k_t_estimated_over_time = []
    k_r_estimated_over_time = []
    print("here")
    for t in range(force_np.shape[0]):  # Loop over each timestep

        def translation_residuals(k_t):
            """Residual function for translational stiffness at timestep t."""
            alpha_t = compute_alpha(Lambda_t, k_t)  # Compute α dynamically
            predicted_force = np.diag(k_t) @ (delta_p[t] - alpha_t * velocity[t])
            return (force_np[t] - predicted_force).flatten()

        def rotation_residuals(k_r):
            """Residual function for rotational stiffness at timestep t."""
            alpha_r = compute_alpha(Lambda_r, k_r)  # Compute α dynamically
            predicted_moment = np.diag(k_r) @ (u_0[t] * theta[t] - alpha_r * moment_np[t])
            return (moment_np[t] - predicted_moment).flatten()

        # Initial stiffness estimates
        k_t_init = np.array([650, 650, 650])
        k_r_init = np.array([10, 10, 10])

        k_t_solution = least_squares(translation_residuals, k_t_init, method='lm')  # Levenberg-Marquardt
        k_r_solution = least_squares(rotation_residuals, k_r_init, method='lm')

        # Store per-timestep stiffness values
        k_t_estimated_over_time.append(k_t_solution.x)
        k_r_estimated_over_time.append(k_r_solution.x)
         
        if t==100:
            break

    # Convert lists to NumPy arrays
    k_t_estimated_over_time = np.array(k_t_estimated_over_time)  # Shape (T, 3)
    k_r_estimated_over_time = np.array(k_r_estimated_over_time)  # Shape (T, 3)

    return k_t_estimated_over_time, k_r_estimated_over_time



# Load the data from the text file
file_path = "Data/TEST_NLS_FREE.txt"
data = pd.read_csv(file_path, delimiter="\t")

# Extract relevant columns
time_array = data["time"].values

force_np = data[["f_x", "f_y", "f_z"]].values
moment_np = data[["m_x", "m_y", "m_z"]].values

zero_force_pos_np = data[["x0", "y0", "z0"]].values
observed_pos_np = data[["x", "y", "z"]].values

zero_force_q_np = data[["u0_x", "u0_y", "u0_z", "theta0"]].values
observed_q_np = data[["u_x", "u_y", "u_z", "theta"]].values

# Extract the translational and rotational mass matrices (Lambda_t and Lambda_r)
Lambda_t = data[["lambda_11", "lambda_12", "lambda_13",
                 "lambda_21", "lambda_22", "lambda_23",
                 "lambda_31", "lambda_32", "lambda_33"]].values.reshape(-1, 3, 3)

Lambda_r = data[["lambda_w_11", "lambda_w_12", "lambda_w_13",
                 "lambda_w_21", "lambda_w_22", "lambda_w_23",
                 "lambda_w_31", "lambda_w_32", "lambda_w_33"]].values.reshape(-1, 3, 3)

# Ensure matrices are averaged over time (if needed)
Lambda_t_mean = np.mean(Lambda_t, axis=0)  # Mean over time steps
Lambda_r_mean = np.mean(Lambda_r, axis=0)

# Call the function with extracted data
k_t_estimated_over_time, k_r_estimated_over_time = estimate_stiffness_per_sequence(
    force_np, moment_np, zero_force_pos_np, observed_pos_np,
    zero_force_q_np, observed_q_np, time_array, Lambda_t_mean, Lambda_r_mean
)

# Print the shape to confirm that we now have a time-series of stiffness values
print("Shape of k_t_estimated_over_time:", k_t_estimated_over_time.shape)  # Expected: (T, 3)
print("Shape of k_r_estimated_over_time:", k_r_estimated_over_time.shape)  # Expected: (T, 3)

# Print some sample values
print("First few estimated translational stiffness values:")
print(k_t_estimated_over_time[:20])

print("First few estimated rotational stiffness values:")
print(k_r_estimated_over_time[:20])