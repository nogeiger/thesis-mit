clc; clear; close all;

% Define grid for ξ^1 and ξ^2
[x1, x2] = meshgrid(linspace(-5, 5, 50), linspace(-5, 5, 50));

% Define the potential energy function U(ξ^1, ξ^2)
U = 0.1 * (x1.^2 + x2.^2); % Example quadratic potential

% Create VideoWriter object for recording
video_filename = 'Diffusion_Denoising_Process.mp4';
v = VideoWriter(video_filename, 'MPEG-4');
v.FrameRate = 10; % Set frame rate (adjust as needed)
open(v); % Open video file for writing

% Create figure
figure;
surf(x1, x2, U, 'EdgeColor', 'none');
colormap(cool); % Use a cool, dark colormap

% Apply visual enhancements
shading interp;
material dull;
lighting gouraud;
camlight left;
alpha(0.9);

hold on;

% Define the equilibrium point
eq_x1 = 0;
eq_x2 = 0;
eq_U = 0.2;

% Define the most distant point
far_x1 = 1.61;
far_x2 = 2.92;
far_U = 2.78;

% Generate 100 random markers within a region
num_markers = 100;
x1_vals = 0 + 2 * rand(1, num_markers);
x2_vals = 0 + 3 * rand(1, num_markers);

% Compute corresponding U values
U_vals = 0.25 * (x1_vals.^2 + x2_vals.^2);

% Compute distances from (0,0)
distances = sqrt(x1_vals.^2 + x2_vals.^2);
color_intensity = distances / max(distances); % Normalize to [0,1]

% Step 1: Plot all markers initially
scatter_handles = gobjects(1, num_markers);
for i = 1:num_markers
    scatter_handles(i) = plot3(x1_vals(i), x2_vals(i), U_vals(i), 'o', ...
        'MarkerSize', 4, ...
        'MarkerFaceColor', [color_intensity(i), 0, 0], ... % Light to dark red
        'MarkerEdgeColor', 'k', ...
        'LineWidth', 1);
end

% Plot the equilibrium and farthest points
eq_marker = plot3(eq_x1, eq_x2, eq_U, 'ro', 'MarkerSize', 12, ...
    'MarkerFaceColor', [0.3, 0, 0], 'MarkerEdgeColor', 'k', 'LineWidth', 2);
far_marker = plot3(far_x1, far_x2, far_U, 'ro', 'MarkerSize', 12, ...
    'MarkerFaceColor', [1, 0, 0], 'MarkerEdgeColor', 'k', 'LineWidth', 2);

% Capture initial frame
frame = getframe(gcf);
writeVideo(v, frame);

% Step 3: Compute the full trajectory from farthest point to equilibrium
t = linspace(0, 1, num_markers); % num_markers points along the trajectory
x1_curve = far_x1 * (1 - t) + eq_x1 * t;
x2_curve = far_x2 * (1 - t) + eq_x2 * t;
U_curve = 0.25 * (x1_curve.^2 + x2_curve.^2); % Compute U along the path

% Define when the trajectory becomes visible
trajectory_start = round(num_markers * 0.25); % Start showing after 25% disappear

% Define color gradient from light to dark red
curve_colors = linspace(1, 0.3, num_markers); % Transition from light red to dark red

% Initialize trajectory storage (always store, only show after delay)
x1_traj = x1_curve;
x2_traj = x2_curve;
U_traj = U_curve;
color_traj = [curve_colors', zeros(num_markers, 2)]; % Red gradient

% Initialize trajectory handle
trajectory_handle = gobjects(1, num_markers);

% Step 2: Animate the disappearance of markers while showing trajectory
pause(1); % Small delay before disappearing effect starts
for i = 1:num_markers
    % Remove each marker one by one
    delete(scatter_handles(i)); 

    % Start showing trajectory at the correct time
    if i >= trajectory_start
        % Plot the stored trajectory up to point i
        plot3(x1_traj(1:i), x2_traj(1:i), U_traj(1:i), '-', 'Color', color_traj(i, :), 'LineWidth', 5);
    end

    drawnow; % Forces MATLAB to update the figure

    % Capture frame for video
    frame = getframe(gcf);
    writeVideo(v, frame); 
    pause(0.01); % Small pause to create animation effect
end

% Close video file
close(v);

% Labels
xlabel('$x^1$', 'Interpreter', 'latex', 'FontSize', 16);
ylabel('$x^2$', 'Interpreter', 'latex', 'FontSize', 16);
zlabel('U($x^1, x^2$)', 'Interpreter', 'latex', 'FontSize', 16);

% Adjust view
view([-55, 50]);
grid on;
hold off;