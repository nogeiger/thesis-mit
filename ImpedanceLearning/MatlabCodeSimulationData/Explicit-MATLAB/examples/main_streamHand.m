clc; clear; close all;

%% Read out text file

% Define the file path
filename = '/Users/johanneslachner/MIT Dropbox/Johannes Lachner/forNoah/streamed_data.txt';

% Open the file
fid = fopen(filename, 'r');
if fid == -1
    error('Could not open the file. Check the file path and permissions.');
end

% Initialize storage variables
time = [];
H_rw = zeros(4, 4, 0);
H_rfk = zeros(4, 4, 0);
H_rfib = zeros(4, 4, 0);
H_rft = zeros(4, 4, 0);

% Read through the file line by line
while ~feof(fid)
    line = strtrim(fgetl(fid));
    
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
    elseif contains(line, 'Right Finger Knuckle:')
        mat_str = extractAfter(line, 'Right Finger Knuckle: ');
        mat_values = str2num(mat_str); %#ok<ST2NM>
        if numel(mat_values) == 16
            H_rfk(:, :, end+1) = reshape(mat_values, 4, 4)';
        end
    elseif contains(line, 'Right Finger intermediate base:')
        mat_str = extractAfter(line, 'Right Finger intermediate base: ');
        mat_values = str2num(mat_str); %#ok<ST2NM>
        if numel(mat_values) == 16
            H_rfib(:, :, end+1) = reshape(mat_values, 4, 4)';
        end
    elseif contains(line, 'Right Finger tip:')
        mat_str = extractAfter(line, 'Right Finger tip: ');
        mat_values = str2num(mat_str); %#ok<ST2NM>
        if numel(mat_values) == 16
            H_rft(:, :, end+1) = reshape(mat_values, 4, 4)';
        end
    end
end

% Close the file
fclose(fid);

%% Animate the Movement

%% Create Figure and Axes for Visualization
figure;
ax = axes; % Create axis handle
hold(ax, 'on'); grid(ax, 'on');
axis(ax, 'equal');
xlabel(ax, 'X [m]'); ylabel(ax, 'Y [m]'); zlabel(ax, 'Z [m]');
title(ax, 'Finger Motion');
view(ax, 3);
xlim(ax, [-1.5 1.5]); ylim(ax, [-1.5 1.5]); zlim(ax, [0 3]);

% Initialize hgtransform objects with hierarchical nesting
hg_rw = hgtransform('Parent', ax);      % Root: Right Wrist
hg_rfk = hgtransform('Parent', hg_rw);  % Child of Right Wrist
hg_rfib = hgtransform('Parent', hg_rfk); % Child of Right Finger Knuckle
hg_rft = hgtransform('Parent', hg_rfib); % Child of Right Finger Intermediate Base

% Create marker objects as children of the correct transformation objects
h_rw = plot3(0, 0, 0, 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r', 'Parent', hg_rw);
h_rfk = plot3(0, 0, 0, 'go', 'MarkerSize', 8, 'MarkerFaceColor', 'g', 'Parent', hg_rfk);
h_rfib = plot3(0, 0, 0, 'bo', 'MarkerSize', 8, 'MarkerFaceColor', 'b', 'Parent', hg_rfib);
h_rft = plot3(0, 0, 0, 'ko', 'MarkerSize', 8, 'MarkerFaceColor', 'k', 'Parent', hg_rft);

% Get cycle time
dt = time(2) - time(1);

% Animation
%% Animate the Movement
for i = 1:length(time)

    tic 

    % Create transformation matrices for hgtransform
    T_rw = H_rw(:, :, i);
    T_rfk = H_rfk(:, :, i);
    T_rfib = H_rfib(:, :, i);
    T_rft = H_rft(:, :, i);

    % Apply transformations using hgtransform
    set(hg_rw, 'Matrix', T_rw);
    set(hg_rfk, 'Matrix', T_rfk);
    set(hg_rfib, 'Matrix', T_rfib);
    set(hg_rft, 'Matrix', T_rft);

    % Apply transformations hierarchically
    set(hg_rw, 'Matrix', T_rw);   % Move Right Wrist (root)
    set(hg_rfk, 'Matrix', T_rfk); % Move Right Finger Knuckle relative to wrist
    set(hg_rfib, 'Matrix', T_rfib); % Move Right Finger Intermediate Base relative to knuckle
    set(hg_rft, 'Matrix', T_rft); % Move Right Finger Tip relative to base
    
    drawnow;
    
    % Do not go faster than real time
    while toc < dt
        % do nothing
    end
end

