function out = getWorkStats(work,problem,solution)


%append all data to the solution info structure
out = solution.info;

%iteration bound
out.max_iter = work.settings.max_iter;

%add number of constraints, variables, equalities etc
out.m           = work.data.m;
out.n           = work.data.n;
out.nEqualities = sum(problem.l == problem.u); 

%% Extract the data used by the solver

%get the solver scalings
if(isempty(work.scaling))
    sD = speye(out.n);
    sE = speye(out.m);
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

%get the solvers sigma and rho
sSigma = work.settings.sigma;
sRho   = work.settings.rho;

%form the KKT matrix for the system, including rho and sigma
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
out.infNormCond_s      = max(infNorms_s)./min(infNorms_s);
out.oneNormCond_s      = max(oneNorms_s)./min(oneNorms_s);
out.twoNormCond_s      = max(twoNorms_s)./min(twoNorms_s);
out.infNormCond_d      = max(infNorms_d)./min(infNorms_d);
out.oneNormCond_d      = max(oneNorms_d)./min(oneNorms_d);
out.twoNormCond_d      = max(twoNorms_d)./min(twoNorms_d);
out.infNormCond_u      = max(infNorms_u)./min(infNorms_u);
out.oneNormCond_u      = max(oneNorms_u)./min(oneNorms_u);
out.twoNormCond_u      = max(twoNorms_u)./min(twoNorms_u);
out.condNumberKKT_s  = condest(KKTs);
out.rho = sRho;
out.sigma = sSigma;
out.alpha = work.settings.alpha;
out.condA_s = condest(sA'*sA);
out.condA_u = condest(uA'*uA);
out.froA_u = norm(uA,'fro');
out.froA_s = norm(sA,'fro');
out.trP_s  = trace(sP);
out.trP_u  = trace(uP);
out.normA2_u = normest((uA),1e-4);
out.normA2_s = normest((sA),1e-4);
out.normP2_u = normest((uP),1e-4);
out.normP2_s = normest((sP),1e-4);
out.normq_s2 = norm(sq);
out.normq_u2 = norm(uq);
out.normq_s1 = norm(sq,1);
out.normq_u1 = norm(uq,1);
out.normq_sinf = norm(sq,inf);
out.normq_uinf  = norm(uq,inf);

%statistics from the scaling
out.normD_1 = norm(diag(sD),1);
out.normD_2 = norm(diag(sD),2);
out.normD_inf = norm(diag(sD),inf);
out.normE_1 = norm(diag(sE),1);
out.normE_2 = norm(diag(sE),2);
out.normE_inf = norm(diag(sE),inf);

out.normDinv_1 = norm(1./diag(sD),1);
out.normDinv_2 = norm(1./diag(sD),2);
out.normDinv_inf = norm(1./diag(sD),inf);
out.normEinv_1 = norm(1./diag(sE),1);
out.normEinv_2 = norm(1./diag(sE),2);
out.normEinv_inf = norm(1./diag(sE),inf);


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

%get the RHS terms in the termination max criteria
out.resPriMax_s = max([norm(sA*sx,inf),norm(sz,inf)]);
out.resDuaMax_s = max([norm(sP*sx,inf),norm(sA'*sy,inf),norm(sq,inf)]);
out.resPriMax_u = max([norm(uA*ux,inf),norm(uz,inf)]);
out.resDuaMax_u = max([norm(uP*ux,inf),norm(uA'*uy,inf),norm(uq,inf)]);

%compute the residuals
out.resPri_s    = norm(sA*sx - sz,inf);
out.resDua_s    = norm(sP*sx + sq + sA'*sy,inf);
out.resPri_u    = norm(uA*ux - uz,inf);
out.resDua_u    = norm(uP*ux + uq + uA'*uy,inf);

%compute norms of primal and dual variables
out.priNorm_s1   = norm(sx,1);
out.priNorm_s2   = norm(sx,2);
out.priNorm_sinf = norm(sx,inf);
out.priNorm_u1   = norm(ux,1);
out.priNorm_u2   = norm(ux,2);
out.priNorm_uinf = norm(ux,inf);
out.duaNorm_s1   = norm(sy,1);
out.duaNorm_s2   = norm(sy,2);
out.duaNorm_sinf = norm(sy,inf);
out.duaNorm_u1   = norm(uy,1);
out.duaNorm_u2   = norm(uy,2);
out.duaNorm_uinf = norm(uy,inf);

%record the relative and absolute tolerances
out.eps_abs = work.settings.eps_abs;
out.eps_rel = work.settings.eps_rel;

%overwrite the status field to give an integer value
out.status = strcmp(out.status,'Solved');




function S = symmetrize(S)

S = (S+S') - diag(diag(S));