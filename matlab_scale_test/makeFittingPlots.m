function makeFittingPlots(targetDir)

[stats,optStats,optStatsUnFlat] = loadPlottingData(targetDir);

%% 
l = [1e-3,1e5];
figure(1); clf
semilogy(optStats.rhoOpt); hold on
semilogy(optStats.rhoGoodLower,'k-.');
semilogy(optStats.rhoGoodUpper,'k-.');
ylim(l)

yyaxis 'right'; hold on  
val1 =  ( (optStats.trP_s./optStats.n) + optStats.sigma)   ./   (optStats.froA_s ./ min(optStats.m - optStats.nEqualities,optStats.n));
%val1 =  (optStats.trP_s./optStats.n)    ./   (optStats.froA_s ./ optStats.m) ;
val1 = ( (optStats.trP_s ) )   ./   (optStats.froA_s );
val2 = 1./(optStats.normq_uinf + 1);
valNorm2_s = (optStats.duaNorm_s2./optStats.priNorm_s2);
valNorm1_s = (optStats.duaNorm_s1./optStats.priNorm_s1);
valNormInf_s = (optStats.duaNorm_sinf./optStats.priNorm_sinf);
valNorm2_u = (optStats.duaNorm_u2./optStats.priNorm_u2);
valNorm1_u = (optStats.duaNorm_u1./optStats.priNorm_u1);
valNormInf_u = (optStats.duaNorm_uinf./optStats.priNorm_uinf);

valNorm_s_P = max(optStats.normA2_s,1);
valNorm_s_D = max(max(optStats.normP2_s,optStats.normA2_s));
valNorm_u_P = max(optStats.normA2_u,1);
valNorm_u_D = max(max(optStats.normP2_u,optStats.normA2_u));

%semilogy((valNorm2_s),'-.');
%semilogy(valNorm2,'-.k');
%semilogy(optStats.froA_s,'k-.');
semilogy((optStats.duaNorm_s2)./sqrt(optStats.priNorm_s2),'r-.');
%semilogy(sqrt(optStats.normq_s2.*valNorm2_s),'r-.');
%semilogy((optStats.normq_sinf),'r-.');
%semilogy(1./optStats.normEinv_2);
%%semilogy((valNormInf),'-.');
%semilogy(1./optStats.oneNormCond_d,'-.')
set(gca,'yscale','log')
ylim(l)
grid on
title('Optimal \rho (blue) and primal/dual solution ratios (red)');

%%

figure(2); clf
semilogy(optStats.duaNorm_sinf,'r-.'); hold on
semilogy(optStats.priNorm_sinf,'b-.'); hold on
grid on
title('Primal (blue) and dual (red) scaled solution norms');



%%
% figure(3); clf
% loglog(optStats.normA2_u,optStats.duaNorm_u2,'x')
% figure(4); clf
% loglog(optStats.trP_u,optStats.priNorm_u2,'x')



%%
figure(3);clf

f = flattenStructArray(stats);
idx = f.wasSolvable & ~f.rhoOptIsExtreme;% & f.rho < 100 & f.rho > 1e-2 ;

idxs = f.status == 1;
idxu = f.status == 0;

x = f.rho./f.rhoOpt;
ys = (f.resPri_s./f.resPriMax_s)./(f.resDua_s./f.resDuaMax_s);
yu = (f.resPri_u./f.resPriMax_u)./(f.resDua_u./f.resDuaMax_u);
y = yu;
loglog(x(idx & idxs),1./y(idx & idxs),'b.'), hold on
loglog(x(idx & idxu),1./y(idx & idxu),'r.')
loglog(x(idx & idxu),1./y(idx & idxu),'r.')
grid on
l = [1e-8,1e8];
xlim(l); ylim(l);
title('Residual ratios vs \rho miscalibration');


%%
figure(4);clf

f = flattenStructArray(stats);
idx = f.wasSolvable & ~f.rhoOptIsExtreme;% & f.rho < 100 & f.rho > 1e-2 ;

idxs = f.status == 1;
idxu = f.status == 0;

x = f.rho./f.rhoOpt;
y = (f.priNorm_s2./f.duaNorm_s2);
loglog(x(idx & idxs),y(idx & idxs),'b.'), hold on
loglog(x(idx & idxu),y(idx & idxu),'r.')
grid on
l = [1e-8,1e8];
xlim(l); ylim(l);
title('Primal norm vs dual norm');

%%
figure(5);

x = optStats.iter;
plot(sort(x)); hold on
grid on
title('Iteration counts (sorted)');

%%
figure(6);clf

x = optStats.iter;
[m,idxm] = sort(optStats.m);
[n,idxn] = sort(optStats.n);
plot(n,x(idxn),'bx');
grid on
yyaxis 'right'; hold on  
plot(m,x(idxm),'rx');
title('Iteration counts vs variables (red) and contraints (blue)');


%%
%print out some statistics
fprintf('Number of problems solved : %i/%i (%f %%)\n', length(optStatsUnFlat),length(stats),length(optStatsUnFlat)/length(stats)*100);
fprintf('Total optimal solve times : %f s \n', sum(optStats.solve_time));
fprintf('Mean optimal solve time : %f s \n', mean(optStats.solve_time));
fprintf('Average acceptable RHO window: %.2f\n', mean(log10(optStats.rhoGoodUpper)-log10(optStats.rhoGoodLower)));






function [stats,optStats,optStatsUnFlat] = loadPlottingData(targetDir)

%all targets are a subdirectory of solutions
targetDir = fullfile('solutions',targetDir);
plotDataFile = fullfile(targetDir,'plottingData.mat');

if(exist(plotDataFile,'file'))
    fprintf('Loading pre-processed plotting file %s\n',plotDataFile);
    load(plotDataFile); return;
end

%if it didn't exist, make it before returning
d = dir(fullfile(targetDir,'*_sol.mat'));

%do the newest ones first
[~,k] = sort([d.datenum]);
d = d(fliplr(k));

%load all of the files into a joint structure
fnames = {d.name};
for i = 1:length(fnames)
    fprintf('Loading %s\n',fnames{i});
    fname = fnames{i};
    tmp = load(fullfile(targetDir,fname),'stats','optStats');
    stats(i) = tmp.stats;
    optStats(i) = tmp.optStats;
end

allOptStats = optStats;

% Sort all of the data in optStats according to size 
% of the optimal rho.  Then flatten this structure
allRhoOpt = [optStats.rhoOpt];
[~,idx]   = sort(allRhoOpt);
optStats = optStats(idx);
fnames   = fnames(idx);

%delete the problems with no solution or an extreme rho
delidx = [optStats.rhoOptIsExtreme] | ~[optStats.wasSolvable] | ...
    ([optStats.rhoGoodLower] == stats(1).rho(1) & [optStats.rhoGoodUpper] == stats(1).rho(end));
optStats(delidx) = [];
fnames(delidx) = [];

%try to eliminate those problems with too wide of an optimal rho range
%delidx = diff(log10([optStats.rhoGoodLower;optStats.rhoGoodUpper])) > 2;
%optStats(delidx) = [];

optStatsUnFlat = optStats;

%flatten to a nice structure 
optStats = flattenStructArray(optStats);

save(plotDataFile,'stats','optStats','optStatsUnFlat');
return;
