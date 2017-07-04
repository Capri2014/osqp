function [ stats, optStats ] = makeRecord(filename,rhoVals,sigVals,osqpOptions, readOptions)

% Solve problem <filename> for a collection of values rho/sigma

% Usage : [ stats, optStats ] = makeRecord(filename,rhoVals,sigVals,osqpOptions, readOptions)
%
% Inputs :
%   filename    : file to read
%   rhoVals     : array of rho values
%   sigVals     : array of sigma values
%   osqpOptions : options tructure for osqp solver
%   readOptions : options structure for readProblem.m
%
% Note that the readOptions structure controls how the problem is
% to be prescaled etc by the reading function
%
% Outputs : returns an array of statistics structures created by
%           getWorksStats.m, plus a structure optStats summarizing
%           information about the solution with optimal rho


if(nargin < 4)
    osqpOptions = osqp().default_settings;
end
if(nargin < 5)
    readOptions = [];
end

problem = readProblem(filename,readOptions);
stats = [];

for i = 1:length(rhoVals)
    %fprintf('%s: Solving with rho = %f....',filename,rhoVals(i));
    for j = 1:length(sigVals)
        %solve the problem with new options (new solver)
        %options.alpha = alphaVals(i);
        osqpOptions.rho   = rhoVals(i);
        osqpOptions.sigma = sigVals(j);
        
        [sol,solver] = solveProblem(problem,osqpOptions);
        
        %get the solver workspace
        work = solver.workspace();
        
        %post process -- adds statistics to basic info
        stats = [stats getWorkStats(work,problem,sol)];
        %fprintf('%i iterations\n',stats(end).iter);
    end
    
end

[stats, optStats] = postProcessRecord(stats);

