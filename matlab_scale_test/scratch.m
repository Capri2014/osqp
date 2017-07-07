clear all

%a small problem
%'/Users/pgoulart/Desktop/osqp/matlab_scale_test/problems/maros/marosHS76.mat

%an easy medium one
%file = '/Users/pgoulart/Desktop/osqp/matlab_scale_test/problems/maros/marosQAFIRO.mat'

file = '/Users/pgoulart/Desktop/osqp/matlab_scale_test/problems/random/rNONNEG_LS_32_2_3_0_645.mat';


%generic solver settings
solver = osqp();
osqpOptions = solver.default_settings();
osqpOptions.early_terminate_interval = 1;
osqpOptions.scaling = 0;  %0 if scaling done in the file read
osqpOptions.polish  = 0;
osqpOptions.eps_rel = 1e-4;
osqpOptions.eps_abs = 1e-4;
osqpOptions.eps_prim_inf = 1e-5;
osqpOptions.eps_dual_inf = 1e-5;
osqpOptions.max_iter = 2500;
osqpOptions.alpha = 1.6;
osqpOptions.verbose = 1;
osqpOptions.rho     = 1;

readOptions.primalPreScaling        = true;
readOptions.primalPreScalingNorm    = 1;
readOptions.ruizNorm                = 2;

%create the problem import settings
readOptions = struct;
readOptions.manualScaling = true;

%read the problem without enlargement
problem1 = readProblem(file,readOptions);


%read the problem with enlargement
readOptions.equalityRescaling = 1e+3;
problem2 = readProblem(file,readOptions);


%solve it both ways
[sol1,solver1] = solveProblem(problem1,osqpOptions);
[sol2,solver2] = solveProblem(problem2,osqpOptions);