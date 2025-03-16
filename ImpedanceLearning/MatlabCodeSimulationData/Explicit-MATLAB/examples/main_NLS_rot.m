%% Clean-up
clc; clear; close all;

%% Rotational Stiffness Test
num_tests = 50;  % Number of test cases

% Store results
errors = zeros(3, num_tests);
estimated_stiffness = zeros(3, num_tests);

% Initialize previous stiffness estimate with reasonable values
prev_k_r = [10; 10; 10];  

for i = 1:num_tests
    % Random ground truth stiffness
    k_r_true = randi([5, 15], 3, 1); 

    % Random task-space mass matrix
    Lambda_r = diag(randi([20, 100], 3, 1));  

    % Random angular velocity
    omega = 0.01 + 0.02 * rand(3, 1);  

    % Randomized unit rotation axis
    u_0 = randn(3, 1);
    u_0 = u_0 / norm(u_0);  

    % Random rotation angle (ensuring reasonable range)
    theta = max(0.05, 0.05 + 0.2 * rand());  

    % Compute damping matrix
    alpha_r_true = compute_alpha(Lambda_r, k_r_true);
    B_r_true = alpha_r_true * diag(k_r_true);

    % Compute external moment
    m_ext = (diag(k_r_true) * u_0 * theta) - (B_r_true * omega);

    % Call function to estimate rotational stiffness (passing prev_k_r)
    k_r_estimated = estimate_stiffness_nls(m_ext, u_0, theta, omega, Lambda_r, prev_k_r);

    % Store results
    estimated_stiffness(:, i) = k_r_estimated;
    errors(:, i) = abs(k_r_estimated - k_r_true);
    
    % Update prev_k_r for the next iteration
    prev_k_r = k_r_estimated;

    fprintf('Test %d/%d completed\n', i, num_tests);
end

% Calculate results 
mean_error = mean(errors, 2);
max_error = max(errors, [], 2);
min_error = min(errors, [], 2);

% Print summary
fprintf('\n=== Error Summary ===\n');
fprintf('Mean Error: \n');
disp(mean_error);
fprintf('Max Error: \n');
disp(max_error);
fprintf('Min Error: \n');
disp(min_error);


%% Nonlinear Least Squares (NLS) for Rotational Stiffness
function k_r_est = estimate_stiffness_nls(m_ext, u0, theta, omega, Lambda, prev_k_r)
    % Initial guess: blend default and adaptive estimates
    k_r0_default = [10; 10; 10];  
    k_r0_adaptive = max(abs(m_ext) ./ max(abs(u0 * theta), 1e-3), 5);  
    k_r0 = 0.5 * k_r0_default + 0.5 * k_r0_adaptive;  

    % Define the objective function
    objective = @(k_r) residual_function_rotation(k_r, m_ext, u0, theta, omega, Lambda);

    % Set optimization options
    opts = optimoptions('lsqnonlin', 'Algorithm', 'levenberg-marquardt', ...
                        'Display', 'off', 'TolFun', 1e-6, 'TolX', 1e-6);

    % Solve for k_r
    k_r_est = lsqnonlin(objective, k_r0, [1e-6; 1e-6; 1e-6], [], opts);

    %% **OUTLIER DETECTION & MITIGATION**
    k_r_min = 1;     % Minimum plausible stiffness
    k_r_max = 50;    % Maximum plausible stiffness
    max_change = 10; % Max allowed stiffness jump per iteration

    % Clamp values if they exceed plausible range
    if any(k_r_est < k_r_min) || any(k_r_est > k_r_max)
        warning('Outlier detected: k_r_est out of bounds. Reverting to previous value.');
        k_r_est = prev_k_r;
    end

    % Detect sudden jumps and smooth them
    if any(abs(k_r_est - prev_k_r) > max_change)
        warning('Sudden jump in stiffness detected. Applying smoothing.');
        k_r_est = 0.7 * prev_k_r + 0.3 * k_r_est;
    end
end

%% Residual Function for NLS
function residuals = residual_function_rotation(k_r, m_ext, u0, theta, omega, Lambda)
    % Compute damping parameter
    alpha_r = compute_alpha(Lambda, k_r);

    % Reformulated moment equation
    m_est = diag(k_r) * (u0 * theta) - (alpha_r * diag(k_r) * omega);

    % Compute residuals
    residuals = m_ext - m_est;
end

%% Compute alpha function
function alpha = compute_alpha(Lambda, k, damping_factor)
    if nargin < 3, damping_factor = 0.7; end

    % Ensure k values are reasonable
    k = max(k, 1e-3);

    % Eigenvalue decomposition
    [U, Sigma] = eig(Lambda);
    sqrt_Lambda = U * diag(sqrt(diag(Sigma))) * U';

    % Compute sqrt_k
    sqrt_k = diag(sqrt(k));

    % Compute b_t
    D = eye(3) * damping_factor;
    b_t = sqrt_Lambda * D * sqrt_k + sqrt_k * D * sqrt_Lambda;

    % Improved alpha scaling
    alpha = (2 * trace(b_t)) / max(sum(k), 5);  
end