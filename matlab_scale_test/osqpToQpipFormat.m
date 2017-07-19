function [ipProblem] = osqpToQpipFormat(osqpProblem)

% osqpToIPFormat :  convert a problem in OSQP format to
% one suitable for the interior point solver qpip.m

ipProblem.P = osqpProblem.P;
ipProblem.q = osqpProblem.q;

%find the equalities
idx = osqpProblem.l == osqpProblem.u;
ipProblem.A = osqpProblem.A(idx,:);
ipProblem.b = osqpProblem.u(idx,:);

%find the inequalities
ipProblem.C = [osqpProblem.A(~idx,:); -osqpProblem.A(~idx,:)];
ipProblem.d = [osqpProblem.u(~idx,:); -osqpProblem.l(~idx,:)];

%eliminate infinite bounds
idx = isinf(ipProblem.d);
ipProblem.C(idx,:) = [];
ipProblem.d(idx,:) = [];


