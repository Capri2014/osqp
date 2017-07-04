function data = basis_pursuit(m, n, dens_lvl, seed)
% The basis purusit problem is
%             minimize	|| x ||_1
%             subject to   Ax = b
%
% Arguments
% ---------
% m, n        - Dimensions of matrix A        <int>
% dens_lvl    - Density level of matrix A     <double>
% seed        - Random number generator seed  <int>

rng(seed);

% Generate data
A = sprandn(m, n, dens_lvl);
b = A * ((rand(n, 1) > 0.5) .* randn(n, 1) / sqrt(n));

% Construct the problem
%      minimize	np.ones(n).T * t
%      subject to  Ax = b
%                  -t <= x <= t
In = speye(n);
data.P = sparse(2*n, 2*n);
data.q = [zeros(n, 1); ones(n, 1)];
data.A = [A, sparse(m, n);
          In, -In;
          In, In;];
data.l = [b; -inf*ones(n, 1); zeros(n, 1)];
data.u = [b; zeros(n, 1); inf*ones(n, 1)];

end
