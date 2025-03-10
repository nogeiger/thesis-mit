% [Project]        Stream data from Apple Vision Pro
% Authors                       Email
%   [1] Johannes Lachner        jlachner@mit.edu
%
%
% The code is heavily commented. A famous quote says:
% "Code is read more often than it is written"
%           - Guido Van Rossum, the Creator of Python

%% Cleaning up + Environment Setup
clear; close all; clc;

% %% Read out data
% % Open the text file
% filename = 'streamed_data.txt'; % Replace with your file name
% fid = fopen(filename, 'r');
% 
% % Check if the file was successfully opened
% if fid == -1
%     error('Failed to open file: %s', filename);
% end
% 
% % Initialize variables
% t_record = []; % Store all timestamps
% H_record = [];   % Store all transformation matrices
% i = 1;
% j = 1;
% 
% % Read file line by line
% while ~feof(fid)
%     % Read the current line
%     line = strtrim(fgetl(fid));
% 
%     % Check for the timestamp line
%     if startsWith(line, 'Timestamp:')
%         % Extract and convert the timestamp
%         t_str = extractAfter(line, 'Timestamp:');
%         current_time = str2double(strtrim(t_str));
%         t_record(i) = current_time; %#ok<SAGROW>
% 
%         i = i+1;
% 
%     end
% 
%     % Check for the transformation matrix line
%     if startsWith(line, 'Right Wrist:')
%         % Extract the transformation matrix
%         matrix_str = extractAfter(line, 'Right Wrist:');
%         matrix_values = str2num(matrix_str); %#ok<ST2NM>
% 
%         % Check if we have exactly 16 elements
%         if numel(matrix_values) == 16
%             % Reshape to 4x4 and transpose to match row-major order
%             H_record(:, :, j) = reshape(matrix_values, [4, 4])'; %#ok<SAGROW>
%         else
%             warning('Unexpected number of elements in transformation matrix: %d', numel(matrix_values));
%         end
% 
%         j = j+1;
%     end
% 
% end
% 
% % Close the file
% fclose(fid);
% 
% % Display results
% disp('Total Timestamps Extracted:');
% disp(numel(t_record));
% disp('Total Transformation Matrices Extracted:');
% disp(size(H_record, 3));
% 
% % Rotations
% for k=1:length(H_record)
%     H_streamed(:, :, k) = H_record(1:4, 1:4, k);
% end

%% Read out data


filename = 'right_wrist_matrices.txt';
fid = fopen(filename, 'r');

if fid == -1
    error('Failed to open file: %s', filename);
end

H_streamed = [];
j = 1;

% Read the entire file as one string
file_content = fscanf(fid, '%c');
fclose(fid);

% Extract the part after "Doubles:"
start_index = strfind(file_content, 'Doubles:') + length('Doubles:');
doubles_str = file_content(start_index:end);

% Replace commas with spaces and convert to numbers
doubles_str = strrep(doubles_str, ',', ' ');
matrix_values = str2num(doubles_str); %#ok<ST2NM>

% Process values: 16 doubles for each H_streamed, skip 1 timestamp
num_blocks = floor(numel(matrix_values) / 17); % 16 values + 1 timestamp each block

for k = 1:num_blocks
    start_idx = (k-1)*17 + 1;
    end_idx = start_idx + 15;  % Take 16 values for the matrix
    matrix_data = reshape(matrix_values(start_idx:end_idx), [4,4])';
    H_streamed(:, :, j) = matrix_data;
    j = j + 1;
end

disp('Total Transformation Matrices Extracted:');
disp(size(H_streamed, 3));


% Swap Y and Z axes in the rotation matrices
for idx = 1:size(H_streamed, 3)
    R_corrected = H_streamed(1:3,1:3,idx);
    R_corrected = R_corrected(:, [1 3 2]);  % Swap Y and Z columns
    R_corrected(:, 2) = -R_corrected(:, 2); % Invert the new Y axis
    H_streamed(1:3,1:3,idx) = R_corrected;
end



%% Simulation settings
% simTime = 5;        % Total simulation time
simTime = size(H_streamed, 3) * 0.005;
t  = 0;             % The current time of simulation
dt = 0.001;          % Time-step of simulation

% Set figure size and attach robot to simulation
robot = iiwa14( 'low' );
robot.init( );

% Initial joint values
q = robot.q_init;
q_ini = q;
dq = zeros( robot.nq, 1 );
nq = robot.nq;

%% Create animation
anim = Animation( 'Dimension', 3, 'xLim', [-0.9,0.9], 'yLim', [-0.9,0.9], 'zLim', [0,1.4] );
anim.init( );
anim.attachRobot( robot )

%% Update kinematics
robot.updateKinematics( robot.q_init );

H_ini = robot.getForwardKinematics( q );
p_ini = H_ini( 1:3, 4 );
R_ini = H_ini( 1:3, 1:3 );

% %% Interpolated rotations
% % x -> blue
% % y -> green
% % z -> red
% angles_ini = deg2rad( [ 0, 0, 0 ] );
% R_start = eul2rotm( angles_ini, "ZYX" );
% 
% angles_goal = deg2rad( [ 0, -45, -45 ] );
% R_end = eul2rotm( angles_goal, "ZYX" );                      
% 
% % Number of interpolation steps
% numSteps = simTime / dt + 1;         % Based on simulation time and timestep
% 
% % Generate SLERP interpolation
% R_traj = zeros(3,3,numSteps);
% for i = 1:numSteps
%     t_int = (i-1)/(numSteps-1);
%     R_traj(:,:,i) = R_start * expm(t_int * logm(R_end / R_start));
% end

%% Draw coordinate systems

% Before the loop, create a line for the desired rotation
hg = hgtransform('parent',gca, 'matrix', eye(4));
%hg_0 = hgtransform('parent',gca, 'matrix', eye(4));

% Current rotation
[VFC{1:3}]= func_create_VFC_data('Koordinatensystem',12);
VFC{1} = VFC{1} / 7;

[V,F,C] = VFC{:};
patchTCPeef = patch('Faces', F,    'Vertices' ,V, 'FaceVertexCData', C, 'FaceC', 'flat',...
    'EdgeColor','none', 'Parent', hg, 'faceAlpha', 1, 'tag', 'TCP');

H_start = H_ini;
set(hg, 'matrix', H_start)

%% Init animation
anim.update(0);
step = 1;


%% Cyclic code starts here
while t <= simTime
    tic

    % FK
    H = robot.getForwardKinematics( q );
    set(hg, 'matrix', H)
    p = H( 1:3, 4 );
    R = H( 1:3, 1:3 );

    % Jacobian
    J = robot.getHybridJacobian( q );
    J_t = J( 1:3, : );
    J_r = J( 4:6, : );

    % Angular velocity end-effector
    dp = J_t * dq;
    w = J_r * dq;

    % Mass matrix
    M = robot.getMassMatrix( q );
    M(7,7) = 40 * M(7,7);
    M_inv = M \ eye( size( M ) );

    % Translational impedance control
    kp_t = 800;
    kd_t = 50;

    del_p = H_streamed(1:3,4,step) - H_streamed(1:3,4,1);
    del_p_4d = ones(4,1);
    del_p_4d(1:3,1) = del_p;

    H_rel = eye(4);
    R_z = [ 0, -1, 0; 1, 0, 0; 0, 0, 1];
    H_rel( 1:3, 1:3 ) = R_z;
    H_rel( 1:3, 4 ) = p_ini;

    p_0 = H_rel * del_p_4d;

    f = kp_t * ( p_0(1:3,1) - p ) - kd_t * dp;
    tau_t = J_t' * f;

    % Recorded Rotations
    %R_rel = [ 1, 0, 0; 0, 0, 1; 0, 1, 0 ]; 
    del_R = H_streamed(1:3,1:3,1)' * H_streamed(1:3,1:3,step);
    R_ee_des = R' * R_ini * del_R;

    axang = rotm2axang( R_ee_des );
    u_ee_des = axang( 1:3 );
    phi = axang( 4 );

    % Rotational impedance control
    kp_r = 50;
    kd_r = 2.1;

    u_0_des = R * u_ee_des';
    m = kp_r * u_0_des * phi - kd_r * w;
    tau_r = J_r' * m;

    % Joint space stiffness
    kq = 2;
    tau_q = kq * ( q_ini - q );

    % Nullspace projector
    minSingVal = 0.02;
    Lambda = func_getLambdaLeastSquaresAndSqrt( M, J, minSingVal );
    J_bar = M_inv * J' * Lambda;
    N = eye( robot.nq ) - J' * J_bar';

    % Interpolation robot
    rhs = M \ ( tau_t + tau_r + (N * tau_q) );                                                % We assume gravity and Coriolis are compensated
    [ q, dq ] = func_symplecticEuler( q, dq, rhs, dt );

    % Updates
    t = t + dt;
    step = step + 1;
    robot.updateKinematics( q );
    anim.update( t );

    % Do not go faster than real time
    while toc < dt
        % do nothing
    end

end
