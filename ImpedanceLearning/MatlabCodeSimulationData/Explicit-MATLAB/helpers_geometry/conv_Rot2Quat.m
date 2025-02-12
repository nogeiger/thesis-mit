function quat = conv_Rot2Quat(T)
% make sure your only get the rotation matrix, not the transformation
R = T(1:3,1:3);

tr = trace(R);

if tr > 0
    S   = sqrt(tr + 1) * 2;
    q_w = 0.25 * S;
    q_x = (R(3,2) - R(2,3)) / S;
    q_y = (R(1,3) - R(3,1)) / S;
    q_z = (R(2,1) - R(1,2)) / S;
elseif ((R(1,1) > R(2,2)) && (R(1,1) > R(3,3)))
    S   = sqrt(1 + R(1,1) - R(2,2) - R(3,3)) * 2;
    q_w = (R(3,2) - R(2,3)) / S;
    q_x = 0.25 * S;
    q_y = (R(2,1) + R(1,2)) / S;
    q_z = (R(1,3) + R(3,1)) / S;
elseif (R(2,2) > R(3,3))
    S   = sqrt(1 + R(2,2) - R(1,1) - R(3,3)) * 2;
    q_w = (R(1,3) - R(3,1)) / S;
    q_x = (R(2,1) + R(1,2)) / S;
    q_y = 0.25 * S;
    q_z = (R(3,2) + R(2,3)) / S;
else
    S   = sqrt(1 + R(3,3) - R(1,1) - R(2,2)) * 2;
    q_w = (R(2,1) - R(1,2)) / S;
    q_x = (R(1,3) + R(3,1)) / S;
    q_y = (R(3,2) + R(2,3)) / S;
    q_z = 0.25 * S;
end
    
quat = [q_w, q_x, q_y, q_z]';
quat = quat / norm(quat);