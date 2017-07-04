function makePlots(targetDir)


%all targets are a subdirectory of solutions
targetDir = fullfile('solutions',targetDir);

d = dir(fullfile(targetDir,'*.mat'));

%do the newest ones first
[~,k] = sort([d.datenum]);
d = d(fliplr(k));

fnames = {d.name};
for i = 1:length(fnames)
    fname = fnames{i};
    load(fullfile(targetDir,fname));
    if(all([stats.iter] == max([stats.iter]) ) )
        continue  %skip unsolvable problems
        fprintf('%s : skipping plots.  Couldn''t solve \n',fname);
    end
    
    [~,k] = min([stats.iter]);
    stats(k)
    
    figure(1); clf
    rho       = [stats.rho];
    sigma     = [stats.sigma];
    sigmaVals = min(sort(unique(sigma)));
    
    yyaxis 'right'; 
    for j = 1:length(sigmaVals)
        idx = sigma == sigmaVals(j);
        loglog(rho(idx),([stats(idx).iter]),'r--');hold on
        loglog(rho(idx),([stats(idx).condNumberKKT_s]),'m:','linewidth',2);
    end
    legend(num2str(sigmaVals(:)));
    ylabel('iterations');
    title(fname,'Interpreter','none');
    xlabel('\rho');
    grid on
    set(gca,'yscale','log')
    
    yyaxis 'left'; hold on
    for j = 1:length(sigmaVals)
        idx = sigma == sigmaVals(j);
        %loglog(rho(idx),([stats(idx).resPri_u]./[stats(idx).resPriMax_u])./([stats(idx).resDua_u]./[stats(idx).resDuaMax_u]),'-'); hold on
        semilogy(rho(idx),([stats(idx).resPri_s]./[stats(idx).resPriMax_s])./([stats(idx).resDua_s]./[stats(idx).resDuaMax_s]),'b-.'); hold on
    end
    ylabel('residual ratio')
    set(gca,'yscale','log')

    
    uResRatio = ([stats.resPri_u]./[stats.resPriMax_u])./([stats.resDua_u]./[stats.resDuaMax_u]);
    rho       = [stats.rho];
    yyaxis 'left';
    loglog(rho([1 end]),uResRatio([1 end]),'k');
    text(1,1,sprintf('S = %f',diff(log10(rho([1,end])))/diff(log10(uResRatio([1 end])))));
    drawnow;
    
    
    figure(2); clf
    sigma = [stats.sigma]; 
    rho   = [stats.rho];   
    iter  = [stats.iter];  
    m = length(unique(sigma));
    n = length(unique(rho));
    rho   = reshape(rho,[m n]);
    sigma = reshape(sigma,[m n]);
    iter  = reshape(iter,[m n]);
    
    try
        pcolor(rho,sigma,iter);
        set(gca,'xscale','log','yscale','log');
        shading interp
        title([fname ': Iterations'],'Interpreter','none');
        xlabel('\rho'); ylabel('\sigma');
        colorbar;
        
        figure(3); clf
        Z = reshape(max([uResRatio;1./uResRatio]),[m n]);
        pcolor(rho,sigma,log10(Z));
        set(gca,'xscale','log','yscale','log');
        shading interp
        title([fname ': Residual Ratio'],'Interpreter','none');
        xlabel('\rho'); ylabel('\sigma');
        colorbar
        
    catch
        fprintf('Not enough data to make a pcolor plot\n');
    end
    
    if(i ~= length(fnames))
        pause
    end
    
    
end






%%

