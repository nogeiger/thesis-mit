function [Lambda,Lambda_sqrt] = func_getLambdaLeastSquaresAndSqrt(M,J,minVal)
%FUNC_GETLAMBDALEASTSQUARES Summary of this function goes here
%   Detailed explanation goes here
M_inv = M\eye(size(M));

Lambda_inv = J * M_inv * J';
[u, s, v] = svd(Lambda_inv);                    % least-square solutions for damping design
s_diag = diag(s);
if(min(s_diag) < minVal)
    %s_diag(find(s_diag<min_singular_value))= min_singular_value
    for i=1:1:length(s_diag)
        if s_diag(i) < minVal
            s_diag(i) = minVal;
        end
    end
    not_singular = 0;
%     disp('Lambda singular!');
else
    not_singular = 1;
end
s_inv = 1./s_diag;
s_C_inv_sqrt =sqrt(s_inv);
Lambda = u*diag(s_inv)*u';   % you can use u instead of v because the matrix is p.s.d.
Lambda_sqrt = u*diag(s_C_inv_sqrt)*u';

end

