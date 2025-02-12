function [ x_cur ] = func_sigmoidInterp( x_ini, A_vec, t_cur, t_fin, nrRounds )
% ===========================================================================
% func_sigmoidInterp - Simple interpolator for robot trajectories
%
% Input:
%       x_ini: column vector (max. 3d)
%       A_vec: column vector with amplitudes [ A_x; A_y; A_z ] (max. 3d)
%       t_cur: current cycle time
%       t_fin: cycle time where endpoint is reached
% Output:
%       x_cur: column vector with interpolated array
%
% Authors                       Email                   Created
%   [1] Johannes Lachner        jlachner@mit.edu        2023
%   [2] Moses C. Nah            mosesnah@mit.edu


% Make sure that x_ini is a scalar value
assert( length( x_ini ) == 1 , 'Input x-value must be a scalar' );

% Cosine interpolation of variable x_cur
if t_cur <= t_fin * nrRounds

    % Logistic Sigmoid Function
    alpha_sig = 10; % Sharpness parameter
    x_cur = x_ini + A_vec .* (1 ./ (1 + exp(-alpha_sig .* ((t_cur ./ t_fin) - 0.5))));

else

    x_cur = x_ini + A_vec * x_ini;

end


end



