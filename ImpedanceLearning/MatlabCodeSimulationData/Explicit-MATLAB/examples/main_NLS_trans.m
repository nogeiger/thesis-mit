%% Clean-up
clc; clear; close all;

%% **Translation Stiffness Tests**
num_tests = 50;  % Number of test cases

% Define known translational stiffness range (Ground Truth)
k_t_true_samples = [...
    randi([100, 140], num_tests, 1), ...
    randi([80, 100], num_tests, 1), ...
    randi([130, 170], num_tests, 1)];

% Define test parameters for translation
Lambda_t = diag([200, 150, 100]); % Example mass matrix
delta_p = [0.01; 0.02; 0.015]; % Position displacement
dot_p = [0.05; 0.04; 0.03]; % Velocity

% Placeholder for errors
errors_t = zeros(num_tests, 3);

% Run translation stiffness tests
for i = 1:num_tests
    k_t_true = k_t_true_samples(i, :)';
    alpha_t_true = compute_alpha(Lambda_t, k_t_true);
    f_ext = diag(k_t_true) * (delta_p - alpha_t_true * dot_p);

    % Estimate stiffness using NLS
    k_t_estimated = estimate_stiffness_nls(f_ext, delta_p, dot_p, Lambda_t);

    % Compute error
    errors_t(i, :) = abs(k_t_estimated - k_t_true)';
end

% Compute mean error across all test cases
mean_error_t = mean(errors_t, 1);

% Display results
disp('=== Translation Stiffness Estimation ===');
disp('Mean Error across 50 test cases:');
disp(mean_error_t);


%% **Nonlinear Least Squares (NLS) for Translation**
function k_t_estimated = estimate_stiffness_nls(f_ext, delta_p, dot_p, Lambda)

    % Initial guess for stiffness values
    k_t0 = [100; 90; 150];  % Can be based on external force

    % Define the objective function for lsqnonlin
    objective = @(k_t) residual_function(k_t, f_ext, delta_p, dot_p, Lambda);

    % Set optimization options
    opts = optimoptions('lsqnonlin', 'Algorithm', 'levenberg-marquardt', ...
                        'Display', 'iter', 'TolFun', 1e-6, 'TolX', 1e-6);

    % Solve for k_t
    k_t_estimated = lsqnonlin(objective, k_t0, [1e-6; 1e-6; 1e-6], [], opts);
end


%% **Residual Function for NLS**
function residuals = residual_function(k_t, f_ext, delta_p, dot_p, Lambda)
    
    % Compute damping parameter
    alpha_t = compute_alpha(Lambda, k_t);

    % Compute estimated force
    f_est = diag(k_t) * (delta_p - alpha_t * dot_p);

    % Compute residuals (difference between actual and estimated forces)
    residuals = f_ext - f_est;
end


%% **Compute Alpha Function**
function alpha = compute_alpha(Lambda, k, damping_factor)

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
    alpha = (2 * trace(b_t)) / sum(k);
end