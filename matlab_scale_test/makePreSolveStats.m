function stats = makePreSolveStats(problem,solver)

% Extract possibly useful statistics that do NOT vary with rho/sigma, which
% includes mostly various ways of looking at the scaled A,P,u,l,q

%add number of constraints, variables, equalities etc
[stats.n,stats.m] = solver.get_dimensions;
stats.nEqualities = sum(problem.l == problem.u); 

%% Extract the scaling data and problem info used by the solver

%get the solver work data
work = solver.workspace();

%get the solver scalings
if(isempty(work.scaling))
    sD = speye(stats.n);
    sE = speye(stats.m);
else
    sD = spdiags(work.scaling.D,0,length(work.scaling.D),length(work.scaling.D));
    sE = spdiags(work.scaling.E,0,length(work.scaling.E),length(work.scaling.E));
end

%get the unscaled A and P
uP = problem.P; % same as: inv(sD)*sP*inv(sD);
uA = problem.A; % same as: inv(sE)*sA*inv(sD);

%get the scaled A and P
%sP = cscToSparse(work.data.P);  sP = symmetrize(sP);
%sA = cscToSparse(work.data.A);
sP = sD*uP*sD;
sA = sE*uA*sD;

%get scaled and unscaled linear cost
uq = problem.q;
sq = work.data.q;  %same as sD*uq;

%get the scaled and unscaled bounds
ul = problem.l;
uu = problem.u;
sl = work.data.l;
su = work.data.u;


%% A bunch of other stuff related to problem data and settings


%stats.condA_s = condest(sA'*sA);
stats.froA_u = norm(uA,'fro');
stats.froA_s = norm(sA,'fro');
stats.trP_s  = trace(sP);
stats.trP_u  = trace(uP);
stats.normA2_u = normest((uA),1e-4);
stats.normA2_s = normest((sA),1e-4);
stats.normP2_u = normest((uP),1e-4);
stats.normP2_s = normest((sP),1e-4);
stats.normq_s2 = norm(sq);
stats.normq_u2 = norm(uq);
stats.normq_s1 = norm(sq,1);
stats.normq_u1 = norm(uq,1);
stats.normq_sinf = norm(sq,inf);
stats.normq_uinf  = norm(uq,inf);

%statistics from the scaling
stats.normD_1 = norm(diag(sD),1);
stats.normD_2 = norm(diag(sD),2);
stats.normD_inf = norm(diag(sD),inf);
stats.normE_1 = norm(diag(sE),1);
stats.normE_2 = norm(diag(sE),2);
stats.normE_inf = norm(diag(sE),inf);

stats.normDinv_1 = norm(1./diag(sD),1);
stats.normDinv_2 = norm(1./diag(sD),2);
stats.normDinv_inf = norm(1./diag(sD),inf);
stats.normEinv_1 = norm(1./diag(sE),1);
stats.normEinv_2 = norm(1./diag(sE),2);
stats.normEinv_inf = norm(1./diag(sE),inf);


%a very bad estimate of the dual solution norm
dualSol = [sP sA']\sq;
dualEsty = dualSol((stats.n+1):end,1);
stats.dualNormMagEst_0 = nnz(dualEsty);
stats.dualNormMagEst_h = (sum(sqrt(dualEsty)))^2;
stats.dualNormMagEst_1 = norm(dualEsty,1);
stats.dualNormMagEst_2 = norm(dualEsty,2);
stats.dualNormMagEst_inf = norm(dualEsty,inf);
stats.normAq_1 = norm(sA*sq,1);
stats.normAq_2 = norm(sA*sq,2);
stats.normAq_inf = norm(sA*sq,inf);
tmp = sA'\sq;
stats.normApinvq_1 = norm(tmp,1);
stats.normApinvq_2 = norm(tmp,2);
stats.normApinvq_inf = norm(tmp,inf);


%%
%try to find some info about the linear constraints

% inverse width of the bounds
stats.diffBndInvNorm_u1   = norm(1./(uu-ul),1);
stats.diffBndInvNorm_u2   = norm(1./(uu-ul),2);
stats.diffBndInvNorm_uinv = norm(1./(uu-ul),inf);
stats.diffBndInvNorm_s1   = norm(1./(su-sl),1);
stats.diffBndInvNorm_s2   = norm(1./(su-sl),2);
stats.diffBndInvNorm_sinf = norm(1./(su-sl),inf);

eqidx = su == sl;
m = 1./(su-sl);  
m = min(1000,m);
m = max(1e-3,m);
stats.diffBndInvRule = norm(m(~eqidx)) ./ sum(~eqidx);   
stats.diffBndRule = norm(1./m(~eqidx)) ./ sum(~eqidx);   

function S = symmetrize(S)

S = (S+S') - diag(diag(S));
