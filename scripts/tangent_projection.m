function Q = tangent_projection(Q1, Q2)

% Q1 is reference, Q2 is the FC
id = eye(size(Q1));
Q1 = Q1 + id;
%% Pessoa Method
% Compute Q1^-1/2
[U,S,V] = svd(Q1);
s = diag(S);
% s(s < eig_thresh) = eig_thresh;
S = diag(s.^(-1/2));
Q1_inv_sqrt = U*S*V';

% Compute Q
Q = Q1_inv_sqrt*Q2*Q1_inv_sqrt;
% 
Q = logm(Q);
Q = (Q+Q')/2; % Ensure symmetry

%% Alternative Method
% Q = Q1\Q2;
% e = eig(Q);
% dG = sqrt(sum(log(e).^2));