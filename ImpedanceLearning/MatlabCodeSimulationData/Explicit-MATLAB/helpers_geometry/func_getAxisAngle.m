function [u,theta] = func_getAxisAngle( R )
%% Algorithm "R2Quat" of "Implement orientation tasks"
R = R(1:3,1:3);
Roo=trace(R);
if abs(Roo +1) < 1e-7   % if Roo == -1 with calculation tolerance
%     Roo = -1 <=> theta = 90°
%     In this case, the only R(k,k) which is positiv SEEMS to indicate the
%     correct k. This has not been proved.
    [~,k] = max([Roo,R(1,1),R(2,2),R(3,3)]);
else
    [~,k] = max(abs([Roo,R(1,1),R(2,2),R(3,3)]));
end
k = k -1;

switch k
    case 0
        S=sqrt(1+Roo)*2;
        q(1)=0.25*S;
        q(2)=(R(3,2)-R(2,3))/S;
        q(3)=(R(1,3)-R(3,1))/S;
        q(4)=(R(2,1)-R(1,2))/S;
    case 1
        S=sqrt(1 + 2*R(k,k) - Roo)*2;
        q(1)=(R(3,2)-R(2,3))/S;
        q(2)=0.25*S;
        q(3)=(R(1,2)+R(2,1))/S;
        q(4)=(R(1,3)+R(3,1))/S;
    case 2
        S=sqrt(1 + 2*R(k,k) - Roo)*2;
        q(1)=(R(1,3)-R(3,1))/S;
        q(2)=(R(1,2)+R(2,1))/S;
        q(3)=0.25*S;
        q(4)=(R(2,3)+R(3,2))/S;
    case 3
        S=sqrt(1 + 2*R(k,k) - Roo)*2;
        q(1)=(R(2,1)-R(1,2))/S;
        q(2)=(R(1,3)+R(3,1))/S;
        q(3)=(R(2,3)+R(3,2))/S;
        q(4)=0.25*S;
end

%normalize quaternion
q=q'/norm(q);

if q(1) < 0
    q = -q;
end

%% Quaternion to axis angle (theta, u)
% quaternion formula: q = [cos(theta/2); u * sin(theta/2)]
theta=2*acos(q(1));
u = sign(theta) * q(2:4);
if norm(u) > 0
    u = u / norm(u);
end
end