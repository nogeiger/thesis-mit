%% Clean-up
clc; clear; close all;

%% Parameters to test
num_tests = 50;

% Store results for adaptation factors
alpha_estimated_t = zeros(3, num_tests);
errors_alpha_t = zeros(3, num_tests);
balance_check_errors = zeros(1, num_tests);

%% Ground Truth Stiffness (initial known stiffness)
k_old_vec = [120; 90; 150];
K_old = diag(k_old_vec);

%% Define test conditions (simulate increasing delta_p)
delta_p_1 = [0.01; 0.02; 0.015];
delta_p_2 = [0.02; 0.04; 0.03];  % larger displacement

dot_p = [0.05; 0.04; 0.03];

% Simulate true external force based on K_old and delta_p_1
gamma = compute_alpha(k_old_vec);  % simplified for test
f_ext = -K_old * (delta_p_1 - gamma .* dot_p);  % true external force

%% NLS Estimation of alpha
for i = 1:num_tests
    alpha0 = ones(3, 1);  % initial guess
    objective = @(alpha) residual_adaptive_alpha(alpha, f_ext, delta_p_2, dot_p, k_old_vec);
    opts = optimoptions('lsqnonlin', 'Display', 'off');
    alpha_est = lsqnonlin(objective, alpha0, [], [], opts);

    alpha_estimated_t(:, i) = alpha_est;
    errors_alpha_t(:, i) = abs((k_old_vec .* alpha_est) - k_old_vec);

    %% Force balance check
    k_new_vec = alpha_est .* k_old_vec;
    gamma_new = sqrt(1 ./ k_new_vec);  % assuming unit mass
    f_cmd = diag(k_new_vec) * (delta_p_2 - gamma_new .* dot_p);
    residual = f_ext + f_cmd;
    balance_check = norm(f_cmd + residual + f_ext);  % should be near zero
    balance_check_errors(i) = balance_check;

    fprintf('Test %2d: |f_cmd + res + f_ext| = %.4e\n', i, balance_check);

    if balance_check > 1e-6
        warning('Force balance not satisfied at test %d (error: %.4e)', i, balance_check);
    end
end

%% Results Summary
disp('=== Adaptive Stiffness Scaling (Alpha) ===');
disp('Mean Alpha Estimated:');
disp(mean(alpha_estimated_t, 2));
disp('Mean Absolute Error in Adapted Stiffness:');
disp(mean(errors_alpha_t, 2));
disp('Max Force Balance Residual:');
disp(max(balance_check_errors));

%% Function to compute damping gain
function gamma = compute_alpha(k_vec)
    gamma = sqrt(1 ./ k_vec);  % simplified example assuming unit mass
end

%% Residual function for alpha adaptation
function r = residual_adaptive_alpha(alpha, f_ext, delta_p, dot_p, k_old_vec)
    k_new_vec = alpha .* k_old_vec;
    gamma = sqrt(1 ./ k_new_vec);  % again assuming unit mass
    r = f_ext + diag(k_new_vec) * (delta_p - gamma .* dot_p);
end

%% Plot results
figure;

% Plot estimated stiffness values over tests
subplot(3,1,1);
plot(1:num_tests, alpha_estimated_t(1,:) .* k_old_vec(1), '-o', 'LineWidth', 1.5); hold on;
plot(1:num_tests, alpha_estimated_t(2,:) .* k_old_vec(2), '-s', 'LineWidth', 1.5);
plot(1:num_tests, alpha_estimated_t(3,:) .* k_old_vec(3), '-d', 'LineWidth', 1.5);
ylabel('Adapted Stiffness (N/m)');
title('Estimated Stiffness K (Alpha \cdot K_{old})');
legend('x', 'y', 'z');
grid on;

% Plot delta_p change (delta_p_2 is used in the estimation)
subplot(3,1,2);
bar([1;2;3], delta_p_1, 0.4, 'FaceAlpha', 0.6); hold on;
bar([1.5;2.5;3.5], delta_p_2, 0.4, 'FaceAlpha', 0.6);
ylabel('\delta p (m)');
xticks([1.25 2.25 3.25]); xticklabels({'x','y','z'});
title('Change in Position Error (\delta p)');
legend('\delta p_1 (initial)', '\delta p_2 (larger)');
grid on;

% Plot external force (should be constant across all tests)
subplot(3,1,3);
f_ext_repeated = repmat(f_ext, 1, num_tests);
plot(1:num_tests, f_ext_repeated(1,:), '-o', 'LineWidth', 1.5); hold on;
plot(1:num_tests, f_ext_repeated(2,:), '-s', 'LineWidth', 1.5);
plot(1:num_tests, f_ext_repeated(3,:), '-d', 'LineWidth', 1.5);
ylabel('f_{ext} (N)');
xlabel('Test Index');
title('External Force (should stay constant)');
legend('x', 'y', 'z');
grid on;