function [ x_cur ] = func_lissajousInterp( x_ini, A_vec, t_cur, t_fin, nrRounds, freq, phi )
% ===========================================================================
% func_lissajousInterp - Interpolator for Lissajous curve robot trajectories
%
% Input:
%       x_ini: column vector (max. 3d)
%       A_vec: column vector with amplitudes [ A_x; A_y; A_z ] (max. 3d)
%       t_cur: current cycle time
%       t_fin: cycle time where endpoint is reached
%       nrRounds: Number of cycles
%       freq: Vector with frequency multipliers [a; b; c] for each axis
%       phi: Phase shift in radians for x-direction
%
% Output:
%       x_cur: column vector with interpolated array
%
% Authors                       Email                   Created
%   [1] Johannes Lachner        jlachner@mit.edu        2023
%   [2] Moses C. Nah            mosesnah@mit.edu
% ===========================================================================

% Ensure input sizes are correct
assert(length(A_vec) == 3, 'A_vec must be a 3D vector [ A_x; A_y; A_z ]');
assert(length(freq) == 3, 'freq must be a vector of three frequency multipliers [ a; b; c ]');

% Define time variable with frequency scaling
t_scaled = pi * t_cur / t_fin; 

% Lissajous Curve Equations
if t_cur <= t_fin * nrRounds
    x_cur(1) = x_ini(1) + A_vec(1) * sin(freq(1) * t_scaled + phi); % x-axis
    x_cur(2) = x_ini(2) + A_vec(2) * cos(freq(2) * t_scaled);       % y-axis
    x_cur(3) = x_ini(3) + A_vec(3) * sin(freq(3) * t_scaled);       % z-axis
else
    % Maintain final value after completion
    x_cur = x_ini + A_vec .* x_ini;
end

% Ensure output is a column vector
x_cur = x_cur(:);

end