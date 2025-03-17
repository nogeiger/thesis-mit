clc; clear; close all;

% Define grid for ξ^1 and ξ^2
[x1, x2] = meshgrid(linspace(-5, 5, 50), linspace(-5, 5, 50));

% Define the potential energy function U(ξ^1, ξ^2)
U = 0.1 * (x1.^2 + x2.^2); % Example quadratic potential

% Compute gradients (partial derivatives)
[Ux, Uy] = gradient(U, x1(1,2) - x1(1,1), x2(2,1) - x2(1,1));

% Create surface plot
figure;
surf(x1, x2, U, 'EdgeColor', 'none'); 
colormap(cool); % Use a cool, dark colormap

% Apply visual enhancements
shading interp;      % Smooth shading
material dull;       % Reduce shininess
lighting gouraud;    % Smoother lighting
camlight left;       % Add directional light
alpha(0.9);         % Slight transparency to reduce glare

hold on;

% Add a round marker at (0,0) (minimum of U)
plot3(0, 0, 0.3, 'ro', 'MarkerSize', 12, 'MarkerFaceColor', [0.3, 0, 0], 'MarkerEdgeColor', 'k', 'LineWidth', 2);

% Add a round marker at most distant point
plot3(1.61, 2.92, 2.78, 'ro', 'MarkerSize', 12, 'MarkerFaceColor', [1, 0, 0], 'MarkerEdgeColor', 'k', 'LineWidth', 2);

% Generate 50 random markers within a region (not just a line)
num_markers = 100;
x1_vals = 0 + 2 * rand(1, num_markers); % Random values in range [-3,3]
x2_vals = 0 + 3 * rand(1, num_markers); % Random values in range [-3,3]

% Compute corresponding U values
U_vals = 0.25 * (x1_vals.^2 + x2_vals.^2);

% Define colormap from dark red (near 0,0) to light red (farther away)
distances = sqrt(x1_vals.^2 + x2_vals.^2);
color_intensity = distances / max(distances); % Normalize to [0,1]

for i = 1:num_markers
    plot3(x1_vals(i), x2_vals(i), U_vals(i), 'o', ...
          'MarkerSize', 4, ...
          'MarkerFaceColor', [color_intensity(i), 0, 0], ... % Now lighter far, darker near (0,0)
          'MarkerEdgeColor', 'k', ...
          'LineWidth', 1);
end

% Labels
xlabel('$\mathbf{\xi}^1$', 'Interpreter', 'latex', 'FontSize', 16);
ylabel('$\mathbf{\xi}^2$', 'Interpreter', 'latex', 'FontSize', 16);
zlabel('U($\mathbf{\xi}^1, \mathbf{\xi}^2$)', 'Interpreter', 'latex', 'FontSize', 16);

% Adjust view
view([-55, 50]);
grid on;
hold off;