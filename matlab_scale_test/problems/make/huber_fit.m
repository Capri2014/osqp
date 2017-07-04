function data = huber_fit(m, n, dens_lvl, seed)
% Huber fitting problem is defined as
%         minimize	sum( huber(ai'x - bi) ),
%
% where huber() is the Huber penalty function defined as
%                     | 1/2 x^2       |x| <= 1
%         huber(x) = <
%                     | |x| - 1/2     |x| > 1
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
b = A * x_true + 10 * rand(m, 1) .* (rand(m, 1) > 0.95);

% Construct the problem
%      minimize	1/2 u.T * u + np.ones(m).T * v
%      subject to  -u - v <= Ax - b <= u + v
%                  0 <= u <= 1
%                  v >= 0
Im = speye(m);
data.P = blkdiag(sparse(n, n), Im, sparse(m, m));
data.q = [zeros(m + n, 1); ones(m, 1)];
data.A = [A, Im, Im;
          A, -Im, -Im;
          sparse(m, n), Im, sparse(m, m);
          sparse(m, n + m), Im];
data.l = [b; -inf*ones(m, 1); zeros(2*m, 1)];
data.u = [inf*ones(m, 1); b; ones(m, 1); inf*ones(m, 1)];

end
