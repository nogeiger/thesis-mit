%% Clean-up
clc; clear; close all;

%% Parameters to test
num_tests = 50;  % Number of test cases

% Store results for translation stiffness
errors_t = zeros(num_tests, 3);
estimated_stiffness_t = zeros(3, num_tests);

% Store results for rotational stiffness
errors_r = zeros(3, num_tests);
estimated_stiffness_r = zeros(3, num_tests);

%% Translation Stiffness Tests
% Define known translational stiffness range (Ground Truth)
k_t_true_samples = [...
    randi([100, 140], num_tests, 1), ...
    randi([80, 100], num_tests, 1), ...
    randi([130, 170], num_tests, 1)];

% Define test parameters for translation
Lambda_t = diag([200, 150, 100]); % Example mass matrix
delta_p = [0.01; 0.02; 0.015]; % Position displacement
dot_p = [0.05; 0.04; 0.03]; % Velocity

%% Rotational Stiffness Tests
k_r_true = randi([5, 15], 3, 1); % Random ground truth values
prev_k_r = k_r_true;
Lambda_r = diag(randi([20, 100], 3, 1));  % Random task-space mass matrix

omega = 0.01 + 0.02 * rand(3, 1);  % Random angular velocity

% Randomized unit rotation axis
u_0 = randn(3, 1);
u_0 = u_0 / norm(u_0);  % Normalize to make it a unit vector

theta = 0.05 + 0.2 * rand();  % Random rotation angle

%% Stiffness Tests
for i = 1:num_tests

    %%%%%%%%%%%%%%%%%%%%%
    % Translational Stiffness
    k_t_true = k_t_true_samples(i, :)';
    gamma_t_true = compute_alpha(Lambda_t, k_t_true);
    f_ext = diag(k_t_true) * (delta_p - gamma_t_true * dot_p);

    % Estimate stiffness using NLS
    k_t_estimated = estimate_stiffness_nls_translation(f_ext, delta_p, dot_p, Lambda_t);

    % Store results
    estimated_stiffness_t(:, i) = k_t_estimated;
    errors_t(i, :) = abs(k_t_estimated - k_t_true)';

    % Compute expected Br using the ground truth
    gamma_r_true = compute_alpha(Lambda_r, k_r_true);
    B_r_true = gamma_r_true * diag(k_r_true);

    % Compute external moment
    m_ext = (diag(k_r_true) * u_0 * theta) - (B_r_true * omega);

    % Estimate stiffness using NLS
    k_r_estimated = estimate_stiffness_nls_rotation(m_ext, u_0, theta, omega, Lambda_r, prev_k_r);
    prev_k_r = k_r_estimated;

    % Store results
    estimated_stiffness_r(:, i) = k_r_estimated;
    errors_r(:, i) = abs(k_r_estimated - k_r_true);

    % Display progress
    fprintf('Test %d/%d completed\n', i, num_tests);
end

%% Results Summary
disp('=== Translation Stiffness Estimation ===');
disp('Mean Error across 50 test cases:');
disp(mean(errors_t, 1));

disp('=== Rotational Stiffness Estimation ===');
disp('Mean Error across 50 test cases:');
disp(mean(errors_r, 2));


%% Nonlinear Least Squares (NLS) for Translation
function k_t_estimated = estimate_stiffness_nls_translation(f_ext, delta_p, dot_p, Lambda)
% Initial guess for stiffness values
k_t0 = [100; 90; 150];

% Define the objective function for lsqnonlin
objective = @(k_t) residual_function_translation(k_t, f_ext, delta_p, dot_p, Lambda);

% Set optimization options
opts = optimoptions('lsqnonlin', 'Algorithm', 'levenberg-marquardt', ...
    'Display', 'off', 'TolFun', 1e-6, 'TolX', 1e-6);

% Solve for k_t
k_t_estimated = lsqnonlin(objective, k_t0, [1e-6; 1e-6; 1e-6], [], opts);
end


%% Nonlinear Least Squares (NLS) for Rotation
function k_r_est = estimate_stiffness_nls_rotation(m_ext, u0, theta, omega, Lambda, prev_k_r)
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


%% Residual function for translation
function residuals = residual_function_translation(k_t, f_ext, delta_p, dot_p, Lambda)
% Compute damping parameter
alpha_t = compute_gamma(Lambda, k_t);

% Compute estimated force
f_est = diag(k_t) * (delta_p - alpha_t * dot_p);

% Compute residuals (difference between actual and estimated forces)
residuals = f_ext - f_est;
end


%% Residual function for rotation
function residuals = residual_function_rotation(k_r, m_ext, u0, theta, omega, Lambda)
% Compute damping parameter
alpha_r = compute_gamma(Lambda, k_r);

% Reformulated moment equation
m_est = diag(k_r) * (u0 * theta) - (alpha_r * diag(k_r) * omega);

% Compute residuals
residuals = m_ext - m_est;
end


%% Compute damping parameter
function gamma = compute_gamma(Lambda, k, damping_factor)
if nargin < 3, damping_factor = 0.7; end

% Eigenvalue decomposition
[U, Sigma] = eig(Lambda);
Sigma = diag(sqrt(diag(Sigma))); % Take square root of eigenvalues

% Compute sqrt(Lambda)
sqrt_Lambda = U * Sigma * U';

% Convert k_t to a diagonal matrix
sqrt_k = diag(sqrt(k));

% Compute b_t
D = eye(3) * damping_factor;
b_t = sqrt_Lambda * D * sqrt_k + sqrt_k * D * sqrt_Lambda;

% Compute alpha
gamma = (2 * trace(b_t)) / sum(k);
end