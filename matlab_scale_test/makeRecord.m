function [stats] = makeRecord(filename,rhoVals,sigVals,osqpOptions,readOptions,preSolveFlag)

% Solve problem <filename> for a collection of values rho/sigma

% Usage : [ stats, optStats ] = makeRecord(filename,rhoVals,sigVals,osqpOptions, readOptions)
%
% Inputs :
%   filename     : file to read
%   rhoVals      : array of rho values
%   sigVals      : array of sigma values
%   osqpOptions  : options tructure for osqp solver
%   readOptions  : options structure for readProblem.m
%   preSolveFlag : if 0, the solver is actually called and iterate
%                : statistics are produced.  If 1, pre-solution data
%                  statstics are produced instead
%
% Note that the readOptions structure controls how the problem is
% to be prescaled etc by the reading function
%
% Outputs : returns an array of pre- or post-solve statistics structures
%           


problem = readProblem(filename,readOptions);
stats = [];

%if we are solving the problem
if(~preSolveFlag)
    for i = 1:length(rhoVals)
        %fprintf('%s: Solving with rho = %f....',filename,rhoVals(i));
        for j = 1:length(sigVals)
            %solve the problem with new options (new solver)
            %options.alpha = alphaVals(i);
            osqpOptions.rho_ineq   = rhoVals(i);
            osqpOptions.rho_eq     = rhoVals(i)*readOptions.rho_eqScale + readOptions.rho_eqShift;
            osqpOptions.sigma      = sigVals(j);
            
            solver = osqp;
            solver.setup(problem.P,problem.q,problem.A,problem.l,problem.u,osqpOptions);
            sol   = solver.solve();
            stats = [stats makePostSolveStats(problem,solver,sol)];
        end
    end
    
else
    %just use default rho/sigma values, make the presolve data only once
    solver = osqp;
    solver.setup(problem.P,problem.q,problem.A,problem.l,problem.u,osqpOptions);
	stats = [stats makePreSolveStats(problem,solver)];
end


