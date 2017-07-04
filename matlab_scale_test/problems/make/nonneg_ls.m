function data = nonneg_ls(m, n, dens_lvl, seed)
% Nonnegative least-squares problem is defined as
%         minimize	  || Ax - b ||^2
%         subject to  x >= 0
%
% Arguments
% ---------
% m, n        - Dimensions of matrix A        <int>
% dens_lvl    - Density level of matrix A     <double>
% seed        - Random number generator seed  <int>

rng(seed);

% Generate data
A = sprandn(m, n, dens_lvl);
x_true = 2 * rand(n, 1) / sqrt(n);
b = A * (x_true + 0.5 * randn(n, 1) / sqrt(n)) + 0.3 * randn(m, 1);


% Construct the problem
%       minimize	1/2 y.T*y
%       subject to  y = Ax - b
%                   x >= 0
Im = speye(m);
data.P = blkdiag(sparse(n, n), Im);
data.q = zeros(n + m, 1);
data.A = [A, -Im;
          speye(n), sparse(n, m)];
data.l = [b; zeros(n, 1)];
data.u = [b; inf*ones(n, 1)];

end
