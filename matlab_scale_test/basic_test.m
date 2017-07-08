clear all

%a small problem
file = '/Users/pgoulart/Desktop/osqp/matlab_scale_test/problems/maros/marosHS76.mat';


%generic solver settings
solver = osqp();

%read the problem without enlargement
problem = readProblem(file);
solver = osqp();
solver.setup(problem.P,problem.q,problem.A,problem.l,problem.u);
sol = solver.solve();