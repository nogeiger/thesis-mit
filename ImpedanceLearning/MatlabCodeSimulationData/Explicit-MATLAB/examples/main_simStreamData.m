% [Project]        Robot Simulator - Snake
% Authors                       Email
%   [1] Johannes Lachner        jlachner@mit.edu
%   [2] Moses C. Nah            mosesnah@mit.edu
%
%
% The code is heavily commented. A famous quote says:
% "Code is read more often than it is written"
%           - Guido Van Rossum, the Creator of Python

%% Cleaning up + Environment Setup
clear; close all; clc;

%% Read out data
% Open the text file
filename = 'streamed_data.txt'; % Replace with your file name
fid = fopen(filename, 'r');

% Check if the file was successfully opened
if fid == -1
    error('Failed to open file: %s', filename);
end

% Initialize variables
t_record = []; % Store all timestamps
H_record = [];   % Store all transformation matrices
i = 1;
j = 1;

% Read file line by line
while ~feof(fid)
    % Read the current line
    line = strtrim(fgetl(fid));

    % Check for the timestamp line
    if startsWith(line, 'Timestamp:')
        % Extract and convert the timestamp
        t_str = extractAfter(line, 'Timestamp:');
        current_time = str2double(strtrim(t_str));
        t_record(i) = current_time; %#ok<SAGROW>

        i = i+1;

    end

    % Check for the transformation matrix line
    if startsWith(line, 'Right Wrist:')
        % Extract the transformation matrix
        matrix_str = extractAfter(line, 'Right Wrist:');
        matrix_values = str2num(matrix_str); %#ok<ST2NM>

        % Check if we have exactly 16 elements
        if numel(matrix_values) == 16
            % Reshape to 4x4 and transpose to match row-major order
            H_record(:, :, j) = reshape(matrix_values, [4, 4])'; %#ok<SAGROW>
        else
            warning('Unexpected number of elements in transformation matrix: %d', numel(matrix_values));
        end

        j = j+1;
    end

end

% Close the file
fclose(fid);

% Display results
disp('Total Timestamps Extracted:');
disp(numel(t_record));
disp('Total Transformation Matrices Extracted:');
disp(size(H_record, 3));

% Rotations
for k=1:length(H_record)
    H_streamed(:, :, k) = H_record(1:4, 1:4, k);
end

%% Simulation settings
% simTime = 5;        % Total simulation time
simTime = t_record( end );
t  = 0;             % The current time of simulation
dt = 0.005;          % Time-step of simulation

% Set figure size and attach robot to simulation
robot = iiwa14( 'high' );
robot.init( );

% Initial joint values
q = robot.q_init;
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

% Desired rotation
% [VFC_0{1:3}]= func_create_VFC_data('Koordinatensystem',14);
% VFC_0{1} = VFC_0{1} / 4;
% 
% [V_0,F_0,C_0] = VFC_0{:};
% patchTCPeef = patch('Faces', F_0,    'Vertices' ,V_0, 'FaceVertexCData', C_0, 'FaceC', 'flat',...
%     'EdgeColor','none', 'Parent', hg_0, 'faceAlpha', 1, 'tag', 'TCP');
% 
% H_end = H_ini;
% % H_end(1:3,1:3) = ( R_ini * R_traj(:,:,1)' ) * R_traj(:,:,numSteps);
% H_end(1:3,1:3) = R_ini * H_streamed(1:3,1:3,1)' * H_streamed(1:3,1:3, length(t_record) );
% set(hg_0, 'matrix', H_end)

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
    %del_R = R_traj(:,:,1)' * R_traj(:,:,step);
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

    tau_q = -0.4 * dq;

    % Interpolation robot
    rhs = M \ ( tau_t + tau_r + tau_q );                                                % We assume gravity and Coriolis are compensated
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
