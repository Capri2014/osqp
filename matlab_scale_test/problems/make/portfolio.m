function data = portfolio(n, k, dens_lvl, seed)
% Portfolio optimization problem is
%         maximize	  mu.T * x - gamma x.T (F * F.T + D) x
%         subject to  1.T x = 1
%                     x >= 0
%
% Arguments
% ---------
% k, n        - Dimensions of matrix F        <int>
% dens_lvl    - Density level of matrix A     <double>
% seed        - Random number generator seed  <int>

rng(seed);

% Generate data
F = sprandn(n, k, dens_lvl);
D = sparse(diag(sqrt(k)*rand(n, 1)));
mu = randn(n, 1);
gamma = 1;

% Construct the problem
%      minimize	x.T*D*x + y.T*y - mu.T / gamma * x
%      subject to  1.T x = 1
%                  F.T x = y
%                  0 <= x <= 1
data.P = blkdiag(2*D, 2*speye(k));
data.q = [-mu/gamma; zeros(k, 1)];
data.A = [ones(1, n), zeros(1, k);
          F', -speye(k);
          speye(n), sparse(n, k)];
data.l = [1; zeros(k, 1); zeros(n, 1)];
data.u = [1; zeros(k, 1); ones(n, 1)];

end
