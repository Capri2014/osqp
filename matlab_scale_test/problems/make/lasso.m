function data = lasso(m, n, dens_lvl, seed)
% Lasso problem is defined as
%         minimize	|| Ax - b ||^2 + gamma * || x ||_1
%
% Arguments
% ---------
% m, n        - Dimensions of matrix A        <int>
% dens_lvl    - Density level of matrix A     <double>
% seed        - Random number generator seed  <int>

rng(seed);

% Generate data
A = sprandn(m, n, dens_lvl);
x_true = (randn(n, 1) > 0.8) .* randn(n, 1) / sqrt(n);
b = A * x_true + 0.5 * randn(m, 1);
gamma = rand;


% Construct the problem
%     minimize	  y.T * y + gamma * np.ones(n).T * t
%     subject to  y = Ax
%                 -t <= x <= t
In = speye(n);
Onm = sparse(n, m);
data.P = blkdiag(sparse(n, n), 2*speye(m), sparse(n, n));
data.q = [zeros(m + n, 1); gamma*ones(n, 1)];
data.A = [A, -speye(m), Onm';
          In, Onm, -In;
          In, Onm, In];
data.l = [b; -inf*ones(n, 1); zeros(n, 1)];
data.u = [b; zeros(n, 1); inf*ones(n, 1)];

end
