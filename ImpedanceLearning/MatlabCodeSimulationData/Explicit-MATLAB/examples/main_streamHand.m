clc; clear; close all;

%% Read out text file

% Define the file path
filename = '/Users/johanneslachner/MIT Dropbox/Johannes Lachner/forNoah/streamed_data_front_up.txt';

% Open the file
fid = fopen(filename, 'r');
if fid == -1
    error('Could not open the file. Check the file path and permissions.');
end

% Initialize storage variables
time = [];
H_rw = zeros(4, 4, 0);

% Read through the file line by line
while true
    raw_line = fgetl(fid); % Read a line from the file
    
    % Check if we have reached the end of the file
    if ~ischar(raw_line)
        break; % Exit loop if end of file is reached
    end
    
    line = strtrim(raw_line);  % Trim leading/trailing whitespace
    
    % Check for timestamp
    if contains(line, 'Timestamp:')
        t = sscanf(line, 'Timestamp: %f');
        time(end+1) = t;
        
    % Process transformation matrices
    elseif contains(line, 'Right Wrist:')
        mat_str = extractAfter(line, 'Right Wrist: ');
        mat_values = str2num(mat_str); %#ok<ST2NM>
        if numel(mat_values) == 16
            H_rw(:, :, end+1) = reshape(mat_values, 4, 4)';
        end
    end
end

% Close the file
fclose(fid);

%% Extract the wrist trajectory

if isempty(H_rw)
    error('No wrist transformation data found.');
end

% Extract the wrist position trajectory from the transformation matrices
wrist_trajectory = squeeze(H_rw(1:3, 4, :))'; % Extract translation part

%% Define Start and End Markers

p_start = wrist_trajectory(1, :); % Start position
p_end = wrist_trajectory(end, :); % End position

%% Set up the visualization (Animated Figure Only)

figure;
ax = axes; % Create axis handle
hold(ax, 'on'); grid(ax, 'on');
axis(ax, 'equal');
xlabel(ax, 'X [m]'); ylabel(ax, 'Y [m]'); zlabel(ax, 'Z [m]');
title(ax, 'Wrist Motion', 'FontSize', 16);
view(ax, 3);

% Apply specified limits
xlim(ax, [0.15, 0.35]);
ylim(ax, [0.35, 0.55]);
zlim(ax, [1.25, 1.65]);

% Create markers for **start** and **end** positions
plot3(p_start(1), p_start(2), p_start(3), 'o', 'MarkerSize', 10, 'MarkerFaceColor', [0.5, 0.5, 0.5], 'MarkerEdgeColor', [0.5, 0.5, 0.5]); % Grey Start
plot3(p_end(1), p_end(2), p_end(3), 'o', 'MarkerSize', 10, 'MarkerFaceColor', 'k', 'MarkerEdgeColor', 'k'); % Black End

% Create a **single moving marker** for real-time animation
h_rw = plot3(nan, nan, nan, 'ro', 'MarkerSize', 12, 'MarkerFaceColor', 'r');

% Get cycle time
if length(time) > 1
    dt = time(2) - time(1);
else
    dt = 0.01; % Default if only one time step is available
end

% Initialize storage for real-time trajectory
wrist_traj_dynamic = [];

%% Define Video Writer
video_filename = 'wrist_motion_animation.mp4'; % Change name if needed
video_writer = VideoWriter(video_filename, 'MPEG-4'); % Create MP4 video
video_writer.FrameRate = 30; % Set frame rate (adjustable)
open(video_writer); % Open the video file

%% Animation loop with real-time trajectory update
for i = 1:length(time)
    tic 

    % Extract current transformation
    T_rw = H_rw(:, :, i);

    % Extract the translation part (position)
    current_position = T_rw(1:3, 4)';

    % Append to trajectory
    wrist_traj_dynamic = [wrist_traj_dynamic; current_position];

    % Update moving marker position
    set(h_rw, 'XData', current_position(1), ...
              'YData', current_position(2), ...
              'ZData', current_position(3));

    drawnow;

    % Capture and write the frame to the video
    frame = getframe(gcf);
    writeVideo(video_writer, frame);

    % Do not go faster than real time
    while toc < dt
        % do nothing
    end
end

%% Plot Full Trajectory After Animation Completes
plot3(wrist_traj_dynamic(:,1), wrist_traj_dynamic(:,2), wrist_traj_dynamic(:,3), 'b-', 'LineWidth', 2);

% Capture final frame with full trajectory and write to video
frame = getframe(gcf);
writeVideo(video_writer, frame);

% Close the video file
close(video_writer);

disp(['Video saved as ', video_filename]);