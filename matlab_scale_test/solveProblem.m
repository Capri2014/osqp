function [sol,solver] = solveProblem(problem,options)

% Solve a QP with the given options
%
% Usage : [sol,solver] = solveProblem(problem,options)
%
% Returns : a solution structure 'sol' and the osqp object 
% used to produce it

if(nargin == 1)
    options = [];
end
solver = osqp();

solver.setup(problem.P,problem.q,problem.A,problem.l,problem.u,options);
sol = solver.solve();
