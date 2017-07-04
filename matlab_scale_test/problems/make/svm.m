function data = svm(m, n, dens_lvl, seed)
% Support vector machine problem is
%         minimize	|| x ||^2 + gamma * 1.T * max(0, diag(b) A x + 1)
%
% Arguments
% ---------
% m, n        - Dimensions of matrix A        <int>
% dens_lvl    - Density level of matrix A     <double>
% seed        - Random number generator seed  <int>

rng(seed);

% Generate data
N = ceil(m/2);
gamma = 1;
b = [ones(N, 1); -ones(N,1)];
A_upp = sprandn(N, n, dens_lvl);
A_low = sprandn(N, n, dens_lvl);
A = [A_upp / sqrt(n) + (A_upp ~= 0) / n;
     A_low / sqrt(n) - (A_low ~= 0) / n];

% Construct the problem
%       minimize	 x.T * x + gamma 1.T * t
%       subject to  t >= diag(b) A x + 1
%                   t >= 0
data.P = blkdiag(2*speye(n), sparse(m, m));
data.q = [zeros(n, 1); gamma*ones(m, 1)];
data.A = [diag(b)*A, -speye(m);
          sparse(m, n), speye(m)];
data.l = [-inf*ones(m, 1); zeros(m, 1)];
data.u = [-ones(m, 1); inf*ones(m, 1)];

end
