% [Project]        Noah Thesis - Creating trajectories for impedance learning
% Author                        Email
% Johannes Lachner              jlachner@mit.edu


%% Cleaning up + Environment Setup
clear; close all; clc;

% Simulation settings
dt = 0.005;             % Time-step of simulation

% Impedances
k_t = 400;          % Translational stiffness
b_t = 40;           % Translational damping
k_r = 10;           % Rotational stiffness
b_r = 0.5;            % Rotational damping
b_q = 0.1;          % Joint space damping

% External force
F_ext = zeros( 3, 1 );

% Amplitute of trajectory
A = 0.1;                % Fixed movement amplitude
n = 2;                  % Always move back and forth

% Simulation time
traj_flag = 'lissajous';

switch traj_flag
    case 'linear'
        v_peak = 0.1;                   % m/s
        simTime = A / v_peak;
    case 'circular'
        simTime = 5;
    case 'lissajous'
        simTime = 5;
end

steps = simTime / dt;   % Total steps per simulation

%% Parameter combinations

force_shape_flags = { 'bezier', 'cosine', 'sigmoid' };
%force_shape_flags  = { 'cosine' };
movement_directions = { 2 };
force_directions = { 1, 2, 3, [1,2], [1,3], [2,3], [1,2,3] };
%force_directions = { 3 };
A_f_values = 10:1:17;  % Force amplitude range
%A_f_values = 1:1:1;


%% Outer Loop: Iterate through all combinations
for s_idx = 1:length( force_shape_flags )
    shape_flag = force_shape_flags{ s_idx };

    % Define file for current shape flag
    fileName_x = [shape_flag, '2.txt'];
    fileID_x = fopen(fileName_x, 'w');

    % Write headers
    fprintf(fileID_x, 'Time [s]\tPos_0_x [m]\tPos_0_y [m]\tPos_0_z [m]\t');
    fprintf(fileID_x, 'Pos_x [m]\tPos_y [m]\tPos_z [m]\t');
    fprintf(fileID_x, 'Force_x [N]\tForce_y [N]\tForce_z [N]\n');

    % Store joint data
    fileName_q = [ 'q_data_', shape_flag, '.txt' ];

    % Open file for writing
    fileID_q = fopen(fileName_q, 'w');

    % Write headers for clarity
    fprintf(fileID_q, 'Time [s]\tq1 [rad]\tq2 [rad]\tq3 [rad]\tq4 [rad]\tq5 [rad]\tq6 [rad]\tq7 [rad]\n');

    for f_idx = 1:length( force_directions )
        dir_f = force_directions{ f_idx };  % External force direction

        for A_f = A_f_values  % Force amplitude

            for p_idx = 1:length( movement_directions )
                dir_p = movement_directions{ p_idx };  % Movement direction

                % Ensure dir_f and dir_p are numeric before using mat2str
                if iscell(dir_f)
                    dir_f = cell2mat(dir_f);  % Convert cell to numeric array
                end
                if iscell(dir_p)
                    dir_p = cell2mat(dir_p);  % Convert cell to numeric array
                end

                % Convert to string safely
                fprintf(fileID_x, 'Force Directions: %s\t A_f: %f\t Movement Directions: %s\n', ...
                    mat2str(dir_f), A_f, mat2str(dir_p));

                % Reset simulation variables
                t = 0;
                step = 1;

                % Reset robot
                robot = iiwa14( 'low' );
                robot.init( );

                % Reset initial robot configuration and velocity
                q = robot.q_init;
                q(2) = deg2rad(40);
                q(6) = deg2rad(75);
                dq = zeros( robot.nq, 1 );

                robot.updateKinematics( q );

                % Get initial transformation matrix
                H_ee_ini = robot.getForwardKinematics( q );
                p_ee_ini = H_ee_ini( 1:3, 4 );
                R_ee_ini = H_ee_ini( 1:3, 1:3 );

                % Preallocate storage
                x0_print = zeros( 3, steps );
                x_print = zeros( 3, steps );
                f_ext_print = zeros( 3, steps );
                q_print = zeros( robot.nq, steps );
                tau_print = zeros( robot.nq, steps );
                time = zeros( 1, steps );

                %% Cyclic code starts here
                while t < n*simTime
                    % tic

                    % ========================== %
                    % ====== Get robot data ==== %
                    % ========================== %

                    % Get current robot transformation matrix of end-effector
                    H_ee = robot.getForwardKinematics( q );

                    p_ee = H_ee( 1:3, 4 );
                    x_print( :, step ) = p_ee;

                    R_0_ee = H_ee( 1:3, 1:3 );
                    R_0_des = R_ee_ini;
                    R_ee_des = R_0_ee' * R_0_des;
                    axang = rotm2axang( R_ee_des );
                    u_ee_des = axang( 1:3 );
                    u_0_des = R_0_ee * u_ee_des';
                    theta = axang( 4 );

                    % Get Hybrid Jacobian of a point on end-effector
                    J_ee = robot.getHybridJacobian( q );
                    J_ee_t = J_ee( 1:3, : );
                    J_ee_r = J_ee( 4:6, : );

                    % Calculate linear end-effector velocity
                    dp_ee = J_ee_t * dq;

                    % Calculate angular end-effector velocity
                    omega = J_ee_r * dq;

                    % Get mass matrix of robot
                    M = robot.getMassMatrix( q );
                    M(7,7) = 40 * M(7,7);                   % Virtually increase mass of last joint

                    % ============================ %
                    % ======== Trajectory   ====== %
                    % ============================ %

                    % Get desired position on circular trajectory
                    p_ee_0 = p_ee_ini;

                    switch traj_flag
                        case 'linear'
                            p_ee_0( dir_p ) = func_cosineInterp( p_ee_ini( 2 ), A , t, simTime, n );        % Fixed cosine interpolation
                        case 'circular'
                            p_ee_0 = func_circularInterp( p_ee_ini, A , t, simTime, n );
                        case 'lissajous'
                            A_vec = [0.05, 0.1, 0.08]';  % Amplitudes
                            freq = [2, 3, 1]';   % Lissajous frequency multipliers
                            phi = pi / 2;       % Phase shift for x-direction
                            p_ee_0 = func_lissajousInterp( p_ee_ini, A_vec, t, simTime, n, freq, phi );
                    end

                    x0_print( :, step ) = p_ee_0;                                                   % Stored ZFT variable

                    % ============================ %
                    % ====== External Force  ===== %
                    % ============================ %

                    switch shape_flag
                        case 'cosine'
                            F_ext( dir_f ) = func_cosineInterp( 0, A_f , t, simTime, n );
                        case 'sigmoid'
                            F_ext( dir_f ) = func_sigmoidInterp( 0, A_f , t, simTime, n );
                        case 'tanh'
                            F_ext( dir_f ) = func_tanHInterp( 0, A_f , t, simTime, n );
                        case 'bezier'
                            F_ext( dir_f ) = func_bezierInterp( 0, A_f , t, simTime, n );
                        otherwise
                            disp('Error: wrong desired trajectory.')
                    end

                    % TODO
                    %F_ext = zeros( 3, 1 );
                    
                    f_ext_print( :, step ) = F_ext;                                                 % Stored external force variable

                    % ============================ %
                    % ======== Controller   ====== %
                    % ============================ %

                    % Simple impedance control: translation
                    F_ee = k_t * ( p_ee_0( :, 1 ) - p_ee ) - b_t * dp_ee;

                    % Simple impedance control: rotation
                    m_ee = k_r * u_0_des * theta - b_r * omega;

                    % Transform force and moment to torque
                    tau_t = J_ee_t' * F_ee;
                    tau_r = J_ee_r' * m_ee;

                    % Add external force applied to the end-effector
                    tau_ext = J_ee_t' * F_ext;

                    % Add a small joint damping to increase stability
                    tau_q = - b_q * dq;

                    % Superimpose two torques
                    tau = tau_t + tau_r + tau_q + tau_ext;
                    tau_print( :, step ) = tau;


                    % ============================================= %
                    % ======== Proceed one simulation step ======== %
                    % ============================================= %

                    % Interpolation robot
                    rhs = M \ tau;                                                % We assume gravity and Coriolis are compensated
                    [ q, dq ] = func_symplecticEuler( q, dq, rhs, dt );
                    q_print( :, step ) = q;

                    % Update the linkage plot
                    robot.updateKinematics( q );

                    % Update control time and counter
                    time( step ) = t;                                     % Stored time variable
                    t = t + dt;
                    step = step + 1;

                    % Do not go faster than real time
                    % while toc < dt
                    %     % do nothing
                    % end
                end

                %% Write data to file

                % Cartesian data
                exportData = [ time', x0_print', x_print', f_ext_print' ];
                for i = 1:size(exportData, 1)
                    fprintf(fileID_x, '%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n', exportData(i, :));
                end

                % Joint data
                exportData_q = [time', q_print' ];

                for i = 1:size(exportData_q, 1)
                    fprintf(fileID_q, '%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n', exportData_q(i, :));
                end

            end % End movement direction loop
        end % End force amplitude loop
    end % End force direction loop

    % Close file for this shape flag
    fclose(fileID_x);
    fclose(fileID_q);
    disp(['Data successfully exported to ', fileName_x]);
    disp(['Data successfully exported to ', fileName_q]);

end % End shape flag loop


%% Read out robot configurations and plot them

% Read the data, skipping the first header line
data_q = readmatrix(fileName_q, 'Delimiter', '\t', 'NumHeaderLines', 1);
data_x = readmatrix(fileName_x, 'Delimiter', '\t', 'NumHeaderLines', 1);

% Extract time column
time_q = data_q(:, 1); % First column is time
time_x = data_x(:, 1); % First column is time

% Extract joint angles (q) - all columns except the first
q_check = data_q(:, 2:end); % Remaining columns are joint angles

% Extract force values
f_check = data_x(:, 8:end);

% Plot the joint configuration over time
figure;
plot(time_q, rad2deg(q_check) );
title('Joint Trajectories');
xlabel('Time (s)');
ylabel('Joint Angles (deg)');

% Plot the force over time
figure;
plot(time_x, f_check );
title('Force Trajectory');
xlabel('Time (s)');
ylabel('Force (N)');

%% Simulation

robot = iiwa14( 'high' );
robot.init( );

Nt = length( time_q );

anim = Animation( 'Dimension', 3, 'xLim', [-0.4,0.8], ...
    'yLim', [-0.4,0.8], 'zLim', [-0,1]);
anim.init( );
anim.attachRobot( robot );

% Get traceplot to plot robot trajectory
tracePlot = findobj( 'tag', 'tracePlot' );

simT = time_q( 1 );
for i = 1:Nt
    q = q_check( i, : );

    H_ee = robot.getForwardKinematics( q, 'bodyID', 7, 'position', [0,0,0.05] );
    p_ee = H_ee( 1:3, 4 );

    % Update traceplot
    p_tr_x = get( tracePlot, 'XData' );
    p_tr_y = get( tracePlot, 'YData' );
    p_tr_z = get( tracePlot, 'ZData' );
    set( tracePlot, 'XData', [p_tr_x, p_ee(1,1)], 'YData', [p_tr_y, p_ee(2,1)], 'ZData', [p_tr_z, p_ee(3,1)], 'LineWidth', 4 );


    robot.updateKinematics( q );
    anim.update( simT );
    simT = simT + dt;
end

anim.close( );





