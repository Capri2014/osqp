function data = lp(m, n, dens_lvl, seed)
% Linear program in the inequality form is
%         minimize	  c.T * x
%         subject to  Ax <= b
%
% Arguments
% ---------
% m, n        - Dimensions of matrix A        <int>
% dens_lvl    - Density level of matrix A     <double>
% seed        - Random number generator seed  <int>

rng(seed);

% Generate data
A = sprandn(m, n, dens_lvl);
x_true = randn(n, 1) / sqrt(n);

data.P = sparse(n, n);
data.q = -A'*rand(m, 1);
data.A = A;
data.l = -inf*ones(m, 1);
data.u = A*x_true + 0.1*rand(m, 1);

end
