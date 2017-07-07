function makeFittingPlots(targetDir)

[stats,optStats,optStatsUnFlat] = loadPlottingData(targetDir);
optStats


%% 
l = [1e-5,1e4];
figure(1); clf
semilogy(optStats.rhoOpt); hold on
semilogy(optStats.rhoGoodLower,'k-.');
semilogy(optStats.rhoGoodUpper,'k-.');
ylim(l)

yyaxis 'right'; hold on  
val1 = 1./((optStats.froA_s) ./ optStats.n);
val2 = 2*(optStats.yNorm_s2)./sqrt(optStats.xNorm_s2);
val3 = (optStats.yNorm_s2);

semilogy(val1,'b-.');
semilogy(val2,'r-.');
semilogy(val3,'g-.');
set(gca,'yscale','log')
ylim(l)
grid on
title('Optimal \rho (blue) and primal/dual solution ratios (red)');

%%

figure(2); clf
semilogy(optStats.yNorm_sinf,'r-.'); hold on
semilogy(optStats.xNorm_sinf,'b-.'); hold on
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
ys = (f.resPri_s2./f.resPriMax_s2)./(f.resDua_s2./f.resDuaMax_s2);
%yu = (f.resPri_u1./f.resPriMax_u1)./(f.resDua_u1./f.resDuaMax_u1);
y = ys;
loglog(x(idx & idxs),1./y(idx & idxs),'b.'), hold on
loglog(x(idx & idxu),1./y(idx & idxu),'r.')
loglog(x(idx & idxu),1./y(idx & idxu),'r.')
grid on
l = [1e-8,1e8];
xlim(l); ylim(l);
title('Residual ratios vs \rho miscalibration (2 norm)');


%%
figure(4); clf
tags = {%'normD_1',...
        %'normD_inf',...%'normE_1',...%'normE_inf',...%'normDinv_1',...
        %'normq_s1',... %'normq_sinf',...%'normq_u1',...%'normA2_s',...'froA_s',...%'infNormCond_s'%'normA2_s',...%'froA_s',...
        %'diffBndInvNorm_s2',...
        %'diffBndInvRule',...
        %'diffBndRule',...
        'normq_s1',...
        'dualNormMagEst_0',...
        'normAq_1',...
        'normApinvq_1',...
        %'normAq_inf'
        };
    
for i = 1:(length(tags))
    h(i) = semilogy(getfield(optStats,tags{i})./optStats.normq_s1); hold on
end
legend(tags{:});
%semilogy(optStats.condA_s,'k*'); hold on
%semilogy(optStats.mnormq_s1,'k*'); hold on




%%
figure(5);

x = optStats.iter;
plot(sort(x)); hold on
grid on
title('Iteration counts (sorted)');

%%
% figure(6);clf
% 
% x = optStats.iter;
% [m,idxm] = sort(optStats.m);
% [n,idxn] = sort(optStats.n);
% plot(n,x(idxn),'bx');
% grid on
% yyaxis 'right'; hold on  
% plot(m,x(idxm),'rx');
% title('Iteration counts vs variables (red) and contraints (blue)');
% 

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
    %make sure it is up to date
    d = dir(fullfile(targetDir,'*.mat'))
    idx = find(strcmp({d.name},'plottingData.mat'));
    if(all( [d.datenum] <= d(idx).datenum) )
        fprintf('Loading pre-processed plotting file %s\n',plotDataFile);
        load(plotDataFile); return;
    end
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
