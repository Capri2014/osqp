function stats = makePostSolveStats(problem,solver,solution)

% Extract possibly useful statistics that vary with rho/sigma, which
% includes primarly post solve data
%
%   Usage : stats = getPostSolveStats(problem,solver,solution)

%append all data to the solution info structure
stats = solution.info;

%get the solver work data
work = solver.workspace();

%iteration bound
stats.max_iter = work.settings.max_iter;

%% Extract the scaling data and problem info used by the solver

[n,m] = solver.get_dimensions;
%get the solver scalings
if(isempty(work.scaling))
    sD = speye(n);
    sE = speye(m);
else
    sD = spdiags(work.scaling.D,0,length(work.scaling.D),length(work.scaling.D));
    sE = spdiags(work.scaling.E,0,length(work.scaling.E),length(work.scaling.E));
end

%get the solver's scaled A and P
sP = cscToSparse(work.data.P);  sP = symmetrize(sP);
sA = cscToSparse(work.data.A);

%get the solver's unscaled A and P
uP = problem.P; % same as: inv(sD)*sP*inv(sD);
uA = problem.A; % same as: inv(sE)*sA*inv(sD);

%get scaled and unscaled linear cost
uq = problem.q;
sq = work.data.q;  %same as sD*uq;

%% A bunch of statistics relating to the KKT matrix
sRho   = work.settings.rho_ineq;
sSigma = work.settings.sigma;

[m,n] = deal(size(sA,1),size(sP,1));
Im = speye(m);
In = speye(n);
KKTs = [sP + sSigma*In, sA'; sA, -(1/sRho)*Im];
KKTd = [sP +      0*In, sA'; sA, -(0/sRho)*Im];  %data only
KKTu = [uP +      0*In, uA'; uA, -(0/sRho)*Im];  %unscaled data only

%find the column norms for this KKT matrix (with rho/sigma, scaling)
infNorms_s = full(max(abs(KKTs)));
oneNorms_s = full(sum(abs(KKTs)));
twoNorms_s = full(sqrt(sum((KKTs.^2))));

%find the column norms for this KKT matrix (scaled data only)
infNorms_d = full(max(abs(KKTd)));
oneNorms_d = full(sum(abs(KKTd)));
twoNorms_d = full(sqrt(sum((KKTd.^2))));

%find the column norms for this KKT matrix (unscaled data only)
infNorms_u = full(max(abs(KKTu)));
oneNorms_u = full(sum(abs(KKTu)));
twoNorms_u = full(sqrt(sum((KKTu.^2))));

%compute a bunch of possible interesting statistics from problem data
stats.infNormCond_s      = max(infNorms_s)./min(infNorms_s);
stats.oneNormCond_s      = max(oneNorms_s)./min(oneNorms_s);
stats.twoNormCond_s      = max(twoNorms_s)./min(twoNorms_s);
stats.infNormCond_d      = max(infNorms_d)./min(infNorms_d);
stats.oneNormCond_d      = max(oneNorms_d)./min(oneNorms_d);
stats.twoNormCond_d      = max(twoNorms_d)./min(twoNorms_d);
stats.infNormCond_u      = max(infNorms_u)./min(infNorms_u);
stats.oneNormCond_u      = max(oneNorms_u)./min(oneNorms_u);
stats.twoNormCond_u      = max(twoNorms_u)./min(twoNorms_u);
%stats.condNumberKKT_s  = condest(KKTs);

stats.rho_ineq = work.settings.rho_ineq;
stats.rho_eq = work.settings.rho_eq;
stats.rho = work.settings.rho_ineq;
stats.sigma = sSigma;
stats.alpha = work.settings.alpha;


%%
%compute a bunch of possibly interesting statistics from iterates
%NB : I use here the scaled internal iterates as reported from the internal
%workspace, and then scale back to get the unscaled x/y/z.  I did this
%because the matlab interface does not report back the unscaled z iterate
%as part of the solution, so I modified osqp_mex to report internal
%iterates, since then I can recover both scaled and unscaled x/y/z

%get the solution (scaled, from internal working vectors)
sx = work.xiter;
sy = work.yiter;
sz = work.ziter;

%get the descaled solution 
ux = sD*sx;
uy = sE*sy;
uz = sE\sz;

%get the norms of A*x, P*x and A'y, both scaled and unscaled, in all norms
%These are basically the components of the RHS termination criteria also
%used below.  Just recompute the norms here since speed doesn't matter
stats.AxNorm_s1   = norm(sA*sx,1);
stats.AxNorm_s2   = norm(sA*sx,2);
stats.AxNorm_sinf = norm(sA*sx,inf);
stats.AxNorm_u1   = norm(uA*ux,1);
stats.AxNorm_u2   = norm(uA*ux,2);
stats.AxNorm_uinf = norm(uA*ux,inf);

stats.PxNorm_s1   = norm(sP*sx,1);
stats.PxNorm_s2   = norm(sP*sx,2);
stats.PxNorm_sinf = norm(sP*sx,inf);
stats.PxNorm_u1   = norm(uP*ux,1);
stats.PxNorm_u2   = norm(uP*ux,2);
stats.PxNorm_uinf = norm(uP*ux,inf);

stats.AtyNorm_s1   = norm(sA'*sy,1);
stats.AtyNorm_s2   = norm(sA'*sy,2);
stats.AtyNorm_sinf = norm(sA'*sy,inf);
stats.AtyNorm_u1   = norm(uA'*uy,1);
stats.AtyNorm_u2   = norm(uA'*uy,2);
stats.AtyNorm_uinf = norm(uA'*uy,inf);

%get the RHS terms in the termination max criteria, all norms
stats.resPriMax_s1   = max([norm(sA*sx,1),norm(sz,1)]);
stats.resPriMax_s2   = max([norm(sA*sx,2),norm(sz,2)]);
stats.resPriMax_sinf = max([norm(sA*sx,inf),norm(sz,inf)]);
stats.resPriMax_u1   = max([norm(uA*ux,1),norm(uz,1)]);
stats.resPriMax_u2   = max([norm(uA*ux,2),norm(uz,2)]);
stats.resPriMax_uinf = max([norm(uA*ux,inf),norm(uz,inf)]);

stats.resDuaMax_s1   = max([norm(sP*sx,1),norm(sA'*sy,1),norm(sq,1)]);
stats.resDuaMax_s2   = max([norm(sP*sx,2),norm(sA'*sy,2),norm(sq,2)]);
stats.resDuaMax_sinf = max([norm(sP*sx,inf),norm(sA'*sy,inf),norm(sq,inf)]);
stats.resDuaMax_u1   = max([norm(uP*ux,1),norm(uA'*uy,1),norm(uq,1)]);
stats.resDuaMax_u2   = max([norm(uP*ux,2),norm(uA'*uy,2),norm(uq,2)]);
stats.resDuaMax_uinf = max([norm(uP*ux,inf),norm(uA'*uy,inf),norm(uq,inf)]);

%compute the residuals, all norms
stats.resPri_s1    = norm(sA*sx - sz,1);
stats.resPri_s2    = norm(sA*sx - sz,2);
stats.resPri_sinf  = norm(sA*sx - sz,inf);
stats.resPri_u1    = norm(uA*ux - uz,1);
stats.resPri_u2    = norm(uA*ux - uz,2);
stats.resPri_uinf  = norm(uA*ux - uz,inf);
stats.resDua_s1    = norm(sP*sx + sq + sA'*sy,1);
stats.resDua_s2    = norm(sP*sx + sq + sA'*sy,2);
stats.resDua_sinf  = norm(sP*sx + sq + sA'*sy,inf);
stats.resDua_u1    = norm(uP*ux + uq + uA'*uy,1);
stats.resDua_u2    = norm(uP*ux + uq + uA'*uy,2);
stats.resDua_uinf  = norm(uP*ux + uq + uA'*uy,inf);

%compute norms of primal (x) and dual (y) variables, and z variables
stats.xNorm_s1   = norm(sx,1);
stats.xNorm_s2   = norm(sx,2);
stats.xNorm_sinf = norm(sx,inf);
stats.xNorm_u1   = norm(ux,1);
stats.xNorm_u2   = norm(ux,2);
stats.xNorm_uinf = norm(ux,inf);
stats.yNorm_s1   = norm(sy,1);
stats.yNorm_s2   = norm(sy,2);
stats.yNorm_sinf = norm(sy,inf);
stats.yNorm_u1   = norm(uy,1);
stats.yNorm_u2   = norm(uy,2);
stats.yNorm_uinf = norm(uy,inf);
stats.zNorm_s1   = norm(sz,1);
stats.zNorm_s2   = norm(sz,2);
stats.zNorm_sinf = norm(sz,inf);
stats.zNorm_u1   = norm(uz,1);
stats.zNorm_u2   = norm(uz,2);
stats.zNorm_uinf = norm(uz,inf);

%overwrite the status field to give an integer value
stats.status = strcmp(stats.status,'Solved');




function S = symmetrize(S)

S = (S+S') - diag(diag(S));