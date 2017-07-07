targetDir = fullfile('solutions', mfilename);

%create the solver settings for this run
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
osqpOptions.verbose = 0;

%create the problem import settings
readOptions.makeOneSided            = false;
readOptions.primalPreScaling        = true;
readOptions.primalPreScalingNorm    = 1;
readOptions.dualPreScaling          = false;
readOptions.dualPreScalingNorm      = 1;
readOptions.perfectScaling          = false;
readOptions.nonConvexScaling        = true;
readOptions.manualScaling           = false;  %scale manually if not required of OSQP and not done elsewhere
readOptions.ruizNorm                = 2;