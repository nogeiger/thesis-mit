import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

np.random.seed(42)  # for reproducibility - remove this later


def quat_inverse(q):
    q = np.array(q)
    return np.array([q[0], -q[1], -q[2], -q[3]]) / np.dot(q, q)

def quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def quat_to_axis_angle(q_err):
    q_err = q_err / np.linalg.norm(q_err)
    w = np.clip(q_err[0], -1.0, 1.0)
    theta = 2 * np.arccos(w)
    sin_half_theta = np.sqrt(1 - w**2)
    if sin_half_theta < 1e-6:
        axis = np.array([1.0, 0.0, 0.0])
    else:
        axis = q_err[1:] / sin_half_theta
    return axis, theta


#compute alpha
def compute_alpha(Lambda, k, damping_factor=0.7):
    k = np.maximum(k, 1e-3)
    eigvals, U = np.linalg.eigh(Lambda)
    sqrt_Lambda = U @ np.diag(np.sqrt(eigvals)) @ U.T
    sqrt_k = np.diag(np.sqrt(k))
    D = np.eye(3) * damping_factor
    b_t = sqrt_Lambda @ D @ sqrt_k + sqrt_k @ D @ sqrt_Lambda
    return (2 * np.trace(b_t)) / np.sum(k)

def estimate_stiffness(mode, data, prev_k=None):
    if mode == 'rotation':
        # Now input: m_ext, q (current), q0 (desired), omega, Lambda
        m_ext, q, q0, omega, Lambda = data

        # Compute orientation error (q_err = q0⁻¹ * q)
        q_err = quat_multiply(quat_inverse(q0), q)
        u0, theta = quat_to_axis_angle(q_err)

        def residual(k_r):
            alpha_r = compute_alpha(Lambda, k_r)
            m_est = np.diag(k_r) @ (u0 * theta) - alpha_r * (np.diag(k_r) @ omega)
            return m_ext - m_est

        k_r0_default = np.array([10.0, 10.0, 10.0])
        k_r0_adaptive = np.maximum(np.abs(m_ext) / max(np.linalg.norm(u0 * theta), 1e-3), 5)
        k_r0 = 0.5 * k_r0_default + 0.5 * k_r0_adaptive

        result = least_squares(residual, k_r0, bounds=(1e-6, np.inf), method='trf', ftol=1e-6, xtol=1e-6)
        k_r_est = result.x

        # Outlier handling
        if prev_k is not None:
            if np.any(k_r_est < 1) or np.any(k_r_est > 50):
                print("Warning: Outlier detected – reverting to previous value.")
                k_r_est = prev_k
            elif np.any(np.abs(k_r_est - prev_k) > 10):
                print("Warning: Sudden jump – applying smoothing.")
                k_r_est = 0.7 * prev_k + 0.3 * k_r_est

        return k_r_est

    elif mode == 'translation':
        f_ext, delta_p, dot_p, Lambda = data

        def residual(k_t):
            alpha_t = compute_alpha(Lambda, k_t)
            f_est = np.diag(k_t) @ (delta_p - alpha_t * dot_p)
            return f_ext - f_est

        k_t0 = np.array([100.0, 90.0, 150.0])
        result = least_squares(residual, k_t0, bounds=(1e-6, np.inf), method='trf', ftol=1e-6, xtol=1e-6)
        return result.x

# plotting
def plot_stiffness(true_vals, est_vals, title):
    axes = ['x', 'y', 'z']
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for idx in range(3):
        axs[idx].plot(true_vals[:, idx], label='True', marker='o')
        axs[idx].plot(est_vals[:, idx], label='Estimated', marker='x', linestyle='--')
        axs[idx].set_ylabel(f'Stiffness ({axes[idx]})')
        axs[idx].legend()
        axs[idx].grid(True)
    axs[-1].set_xlabel('Test Number')
    fig.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

#Rotation test
def run_rotation_tests(num_tests=50):
    prev_k_r = np.array([10.0, 10.0, 10.0])
    true_stiff = np.zeros((num_tests, 3))
    est_stiff = np.zeros((num_tests, 3))
    errors = np.zeros((num_tests, 3))

    for i in range(num_tests):
        k_r_true = np.random.randint(5, 16, 3)
        Lambda_r = np.diag(np.random.randint(20, 101, 3)) #load lambda values here
        omega = 0.01 + 0.02 * np.random.rand(3) #load omega values here
        # Simulate rotation deviation via quaternion
        axis = np.random.randn(3)
        axis /= np.linalg.norm(axis)
        theta = max(0.05, 0.05 + 0.2 * np.random.rand())  # rotation magnitude
        q0 = np.array([1.0, 0.0, 0.0, 0.0])  # Desired orientation (identity)
        q = np.array([np.cos(theta / 2), *(axis * np.sin(theta / 2))])  # Current orientation (rotated)

        # Create torque from simulated deviation
        alpha_r = compute_alpha(Lambda_r, k_r_true)
        B_r = alpha_r * np.diag(k_r_true)
        m_ext = np.diag(k_r_true) @ (axis * theta) - B_r @ omega  # What sensors would measure

        # Now use q, q0 directly
        k_r_est = estimate_stiffness('rotation', (m_ext, q, q0, omega, Lambda_r), prev_k_r)

        true_stiff[i] = k_r_true
        est_stiff[i] = k_r_est
        errors[i] = np.abs(k_r_est - k_r_true)
        prev_k_r = k_r_est

    print("\n=== Rotation Stiffness Estimation ===")
    print("Mean Error:", np.mean(errors, axis=0))
    print("Estimated Avg:", np.mean(est_stiff, axis=0))
    print("True Avg:", np.mean(true_stiff, axis=0))
    plot_stiffness(true_stiff, est_stiff, "Rotational Stiffness Estimation")
    
#Translation test
def run_translation_tests(num_tests=50):
    k_t_true_samples = np.column_stack([
        np.random.randint(100, 141, num_tests),
        np.random.randint(80, 101, num_tests),
        np.random.randint(130, 171, num_tests)
    ])
    print("k_t_true shape:", k_t_true_samples.shape)
    Lambda_t = np.diag([200, 150, 100])  # Load lambda values form script here
    delta_p = np.array([0.01, 0.02, 0.015]) # calc x-x0, y-y0, z-z0 here
    dot_p = np.array([0.05, 0.04, 0.03]) # use dx, dy, dz here

    est_stiff = np.zeros((num_tests, 3))
    errors = np.zeros((num_tests, 3))

    for i in range(num_tests):
        k_t_true = k_t_true_samples[i]
        alpha_t = compute_alpha(Lambda_t, k_t_true)
        f_ext = np.diag(k_t_true) @ (delta_p - alpha_t * dot_p) #load f_x, f_y, f_z here

        k_t_est = estimate_stiffness('translation', (f_ext, delta_p, dot_p, Lambda_t))
        est_stiff[i] = k_t_est
        errors[i] = np.abs(k_t_est - k_t_true)

    print("\n=== Translation Stiffness Estimation ===")
    print("Mean Error:", np.mean(errors, axis=0))
    print("Estimated Avg:", np.mean(est_stiff, axis=0))
    print("True Avg:", np.mean(k_t_true_samples, axis=0))
    plot_stiffness(k_t_true_samples, est_stiff, "Translational Stiffness Estimation")

#Main script
if __name__ == "__main__":
    run_rotation_tests()
    run_translation_tests()
