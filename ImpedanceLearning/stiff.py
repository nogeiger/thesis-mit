import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

np.random.seed(42)  # for reproducibility - remove this later

#compute alpha
def compute_alpha(Lambda, k, damping_factor=0.7):
    k = np.maximum(k, 1e-3)
    eigvals, U = np.linalg.eigh(Lambda)
    sqrt_Lambda = U @ np.diag(np.sqrt(eigvals)) @ U.T
    sqrt_k = np.diag(np.sqrt(k))
    D = np.eye(3) * damping_factor
    b_t = sqrt_Lambda @ D @ sqrt_k + sqrt_k @ D @ sqrt_Lambda
    return (2 * np.trace(b_t)) / np.sum(k)

# estimation function
def estimate_stiffness(mode, data, prev_k=None):
    if mode == 'rotation':
        m_ext, u0, theta, omega, Lambda = data

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
        u0 = np.random.randn(3)
        u0 /= np.linalg.norm(u0) #calc the u_error here
        theta = max(0.05, 0.05 + 0.2 * np.random.rand()) # calc theta_error here
        alpha_r = compute_alpha(Lambda_r, k_r_true)
        B_r = alpha_r * np.diag(k_r_true)
        m_ext = np.diag(k_r_true) @ (u0 * theta) - B_r @ omega 

        k_r_est = estimate_stiffness('rotation', (m_ext, u0, theta, omega, Lambda_r), prev_k_r)

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
def run_translation_tests(lambda_matrix, noisy_pos_np, clean_pos_np, dx_np,f_ext_np):
    num_tests =1
    k_t_true_samples = np.column_stack([
        np.random.randint(100, 141, num_tests),
        np.random.randint(80, 101, num_tests),
        np.random.randint(130, 171, num_tests)
    ])

    Lambda_t_np = lambda_matrix#np.diag([200, 150, 100])  # Load lambda values form script here
    #Lambda_t = np.diag([200, 150, 100])  # Load lambda values form script here
    
    delta_p_np = noisy_pos_np - clean_pos_np#np.array([0.01, 0.02, 0.015]) # calc x-x0, y-y0, z-z0 here
    delta_p = np.array([0.01, 0.02, 0.015]) # calc x-x0, y-y0, z-z0 here

    dot_p = np.array([0.05, 0.04, 0.03]) # use dx, dy, dz here     #dx_np
    dot_p_np = dx_np

    #f_ext = f_ext_np
    est_stiff = np.zeros((f_ext_np.shape[1], 3))
    errors = np.zeros((num_tests, 3))
    #print("f ext shape",f_ext_np.shape)
    for i in range(f_ext_np.shape[1]):
        #k_t_true = k_t_true_samples[i]
        #alpha_t = compute_alpha(Lambda_t, k_t_true)
        #f_ext = np.diag(k_t_true) @ (delta_p - alpha_t * dot_p) #load f_x, f_y, f_z here
        #print("f ext shape",f_ext.shape, "f_ext_np shape",f_ext_np.shape, "f_ext_np used shape", f_ext_np[0][i].shape)
        #print("delta_p shape",delta_p.shape, "delta_p_np shape",delta_p_np.shape, "delta_p_np used shape", delta_p_np[0][i].shape)
        #print("dot_p shape",dot_p.shape, "dot_p_np shape",dot_p_np.shape, "dot_p_np used shape", dot_p_np[0][i].shape)
        #print("Lambda_t shape",Lambda_t.shape, "Lambda_t_np shape",Lambda_t_np.shape, "Lambda_t_np used shape", Lambda_t_np[0][i].shape)
        #print("lambda generated",Lambda_t, "lambda loaded",Lambda_t_np[0][i])
        Lambda_t = np.diag(Lambda_t_np[0][i]) * np.eye(3)
        #print("lambda new", Lambda_t)
        print("f_ext_np[0][i]",f_ext_np[0][i])
        print("delta_p_np[0][i]",delta_p_np[0][i])
        print("dot_p_np[0][i]",dot_p_np[0][i])
        print("Lambda_t",Lambda_t)
        k_t_est = estimate_stiffness('translation', (f_ext_np[0][i], delta_p_np[0][i], dot_p_np[0][i], Lambda_t))
        #k_t_est = estimate_stiffness('translation', (f_ext[0][i], delta_p[0][i], dot_p[0][1], Lambda_t[0][i]))
        est_stiff[i] = k_t_est
        #errors[i] = np.abs(k_t_est - k_t_true)
        break

    print("\n=== Translation Stiffness Estimation ===")
    #print("Mean Error:", np.mean(errors, axis=0))
    print("Estimated Avg:", np.mean(est_stiff, axis=0))
    #print("True Avg:", np.mean(k_t_true_samples, axis=0))
    #plot_stiffness(k_t_true_samples, est_stiff, "Translational Stiffness Estimation")


#Translation test
def run_translation_inference(num_tests=50):
    #k_t_true_samples = np.column_stack([
    #    np.random.randint(100, 141, num_tests),
    #    np.random.randint(80, 101, num_tests),
    #    np.random.randint(130, 171, num_tests)
    #])
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
