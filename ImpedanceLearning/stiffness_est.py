import numpy as np
import scipy.optimize as opt
import pandas as pd
import os
import torch
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R


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

        K = np.clip(K, 1e-6, None)  # Ensure no negative values
        sqrt_K = np.diag(np.sqrt(K))

        D = np.eye(3) * damping_factor
        b_t = sqrt_Lambda @ D @ sqrt_K + sqrt_K @ D @ sqrt_Lambda
        alpha = (2 * np.trace(b_t)) / np.sum(K)
        return alpha

    # Compute Displacement (Δp) and Velocity (ẋ)
    delta_p = zero_force_pos_np - observed_pos_np  # Position displacement
    velocity = np.gradient(observed_pos_np, time_array, axis=0)  # Velocity estimate

    # Compute quaternion difference (rotation from zero-force to observed)
    q_diff = R.from_quat(observed_q_np) * R.from_quat(zero_force_q_np).inv()

    # Extract rotation angle using rotation vector (safe method)
    theta = np.linalg.norm(q_diff.as_rotvec(), axis=1)
    u_0 = q_diff.as_rotvec() / np.expand_dims(theta + 1e-6, axis=1)  # Normalize

    # Store stiffness values for each timestep
    k_t_estimated_over_time = []
    k_r_estimated_over_time = []
    
    for t in range(force_np.shape[0]):  # Loop over each timestep

        def translation_residuals(k_t):
            """Residual function for translational stiffness at timestep t."""
            alpha_t = compute_alpha(Lambda_t, k_t)  # Compute α dynamically
            predicted_force = np.diag(k_t) @ (delta_p[t] - alpha_t * velocity[t])
            return (force_np[t] - predicted_force).flatten()

        def rotation_residuals(k_r):
            """Residual function for rotational stiffness at timestep t."""
            alpha_r = compute_alpha(Lambda_r, k_r)
            predicted_moment = np.diag(k_r) @ (u_0[t] * theta[t] - alpha_r * moment_np[t])
            residuals = moment_np[t] - predicted_moment
            return residuals.flatten()

        # Initial stiffness estimates
        k_t_init = np.array([650, 650, 650])
        k_r_init = np.array([10, 10, 10])

        # Use bounded optimization methods
        k_t_solution = least_squares(translation_residuals, k_t_init, method='trf', bounds=(1e-6, np.inf))
        k_r_solution = least_squares(rotation_residuals, k_r_init, method='trf', bounds=(1e-6, 1000))

        # Store per-timestep stiffness values
        k_t_estimated_over_time.append(k_t_solution.x)
        k_r_estimated_over_time.append(k_r_solution.x)

    # Convert lists to NumPy arrays
    k_t_estimated_over_time = np.array(k_t_estimated_over_time)  # Shape (T, 3)
    k_r_estimated_over_time = np.array(k_r_estimated_over_time)  # Shape (T, 3)

    return k_t_estimated_over_time, k_r_estimated_over_time


'''
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
print(k_r_estimated_over_time[:20])'
'''
# Set random seed for reproducibility
np.random.seed(42)

# Number of time steps
T = 200

# Define ground truth stiffness values
true_k_t = np.array([700, 800, 900])  # Translational stiffness
true_k_r = np.array([15, 20, 25])  # Rotational stiffness

# Generate time array (assuming uniform sampling)
time_array = np.linspace(0, 10, T)  # 10s total

# Generate small displacements around equilibrium
zero_force_pos_np = np.zeros((T, 3))  # Reference positions
observed_pos_np = zero_force_pos_np + np.random.uniform(-0.01, 0.01, (T, 3))  # Small random shifts

# Compute forces using the stiffness equation F = k * Δx
force_np = (true_k_t * (zero_force_pos_np - observed_pos_np)) + np.random.uniform(-1, 1, (T, 3))  # Add noise

# Generate random quaternions (small rotations)
zero_force_q_np = np.zeros((T, 4))
zero_force_q_np[:, 0] = 1  # Identity quaternion

angles = np.random.uniform(-0.05, 0.05, (T, 3))  # Small rotations
rotations = R.from_euler('xyz', angles).as_quat()
observed_q_np = rotations  # Rotated quaternions

# Compute rotation moments using M = k_r * (θ * u₀)
theta = 2 * np.arccos(observed_q_np[:, 0])  # Rotation angle
u_0 = observed_q_np[:, 1:]  # Axis of rotation
u_0 /= np.linalg.norm(u_0, axis=1, keepdims=True)  # Normalize

moment_np = (true_k_r * (u_0 * theta[:, None])) + np.random.uniform(-0.5, 0.5, (T, 3))  # Add noise

# Define mass matrices (identity with slight perturbations)
Lambda_t = np.eye(3) + np.random.uniform(-0.1, 0.1, (3, 3))
Lambda_r = np.eye(3) + np.random.uniform(-0.1, 0.1, (3, 3))


print("zero_force_q_np (first 10):", zero_force_q_np[:10])
print("observed_q_np (first 10):", observed_q_np[:10])
# Run the estimation function
k_t_estimated_over_time, k_r_estimated_over_time = estimate_stiffness_per_sequence(
    force_np, moment_np, zero_force_pos_np, observed_pos_np,
    zero_force_q_np, observed_q_np, time_array, Lambda_t, Lambda_r
)

# Print results
print("\nEstimated Translational Stiffness (first 10 timesteps):")
print(k_t_estimated_over_time[:10])

print("\nEstimated Rotational Stiffness (first 10 timesteps):")
print(k_r_estimated_over_time[:10])

# Plot estimated stiffness over time
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(k_t_estimated_over_time, label=['k_t_x', 'k_t_y', 'k_t_z'])
plt.axhline(true_k_t[0], color='r', linestyle='--', label='True k_t_x')
plt.axhline(true_k_t[1], color='g', linestyle='--', label='True k_t_y')
plt.axhline(true_k_t[2], color='b', linestyle='--', label='True k_t_z')
plt.xlabel("Time Step")
plt.ylabel("Translational Stiffness")
plt.legend()
plt.title("Estimated Translational Stiffness")

plt.subplot(1, 2, 2)
plt.plot(k_r_estimated_over_time, label=['k_r_x', 'k_r_y', 'k_r_z'])
plt.axhline(true_k_r[0], color='r', linestyle='--', label='True k_r_x')
plt.axhline(true_k_r[1], color='g', linestyle='--', label='True k_r_y')
plt.axhline(true_k_r[2], color='b', linestyle='--', label='True k_r_z')
plt.xlabel("Time Step")
plt.ylabel("Rotational Stiffness")
plt.legend()
plt.title("Estimated Rotational Stiffness")

plt.show()