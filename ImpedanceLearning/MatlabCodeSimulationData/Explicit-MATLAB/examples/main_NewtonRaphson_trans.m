clc; clear; close all;

%% **Translation Stiffness Tests**
num_tests = 50;  % Number of test cases

% Define known translational stiffness range (Ground Truth)
k_t_true_samples = [...
    randi([100, 140], num_tests, 1), ...
    randi([80, 100], num_tests, 1), ...
    randi([130, 170], num_tests, 1)];

% Define test parameters for translation
Lambda_t = diag([200, 150, 100]); % Example stiffness matrix
delta_p = [0.01; 0.02; 0.015]; % Position displacement
dot_p = [0.05; 0.04; 0.03]; % Velocity

% Placeholder for errors
errors_t = zeros(num_tests, 3);

% Run translation stiffness tests
for i = 1:num_tests
    k_t_true = k_t_true_samples(i, :)';
    alpha_t_true = compute_alpha(Lambda_t, k_t_true);
    f_ext = k_t_true .* (delta_p - alpha_t_true .* dot_p);
    k_t_estimated = newton_raphson_kt(Lambda_t, f_ext, delta_p, dot_p);

    % Compute error
    errors_t(i, :) = abs(k_t_estimated - k_t_true)';
end

% Compute mean error across all test cases
mean_error_t = mean(errors_t, 1);

% Display results
disp('=== Translation Stiffness Estimation ===');
disp('Mean Error across 50 test cases:');
disp(mean_error_t);


%% Newton-Raphson for translation
function k_t_estimated = newton_raphson_kt(Lambda, f_ext, delta_p, dot_p, tol, max_iter)
    if nargin < 5, tol = 1e-6; end
    if nargin < 6, max_iter = 10; end

    % Initial guess for stiffness
    k_t = ones(3,1) * 100;

    for i = 1:max_iter
        % Compute alpha using the correct function
        alpha = compute_alpha(Lambda, k_t);

        % Compute force residual
        b_t = alpha * k_t;
        f_residual = k_t .* (delta_p - alpha .* dot_p) - f_ext;

        % Compute Jacobian
        d_alpha_d_k_t = - (2 * trace(diag(b_t))) / sum(k_t)^2;
        J = delta_p - alpha .* dot_p + k_t .* d_alpha_d_k_t .* dot_p;

        % Prevent division by zero
        J(J == 0) = 1e-6;

        % Check stopping condition
        if norm(f_residual) < tol * norm(k_t)
            break;
        end

        % Newton-Raphson update
        k_t_update = f_residual ./ J; % Compute update step
        k_t = k_t - k_t_update; % Apply update

        % Ensure stiffness remains positive
        k_t = max(k_t, 1e-3);
    end

    k_t_estimated = k_t;
end


%% Compute alpha function
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