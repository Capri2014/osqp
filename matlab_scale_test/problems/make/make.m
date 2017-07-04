function make

% Generate a collection of random QPs in the style of the POGS paper

try 
   !rm ../random/r*.mat 
   fprintf('Deleting old random problems\n');
catch
end

problemSize = ceil(logspace(1,1.5,8));
dens_lvls = [0.2 0.5];
costScaling = [1e-3 1 1e3];
seed = 1;

for s = problemSize
    
    fprintf('Creating set %i of %i\n',find(s == problemSize),length(problemSize));
    
    for costS = costScaling
        
        for dens_lvl = dens_lvls
            
            for scaleq = [false true]
                
                label =  ['_' num2str(s) '_' num2str(dens_lvl*10) '_' num2str(log10(costS)) '_' num2str(scaleq) '_' num2str(seed)];
                
                % Basis pursuit
                problem = basis_pursuit(ceil(2*s), ceil(20*s), dens_lvl, seed);
                problem = scaleCosts(problem,costS,scaleq);
                save(['r' upper('basis_pursuit') label],'problem');
                seed = seed + 1;
                
                % Huber fitting
                problem = huber_fit(ceil(20*s), ceil(2*s), dens_lvl, seed);
                problem = scaleCosts(problem,costS,scaleq);
                save(['r' upper('huber_fit') label],'problem');
                seed = seed + 1;
                
                % Lasso
                problem = lasso(ceil(1*s), ceil(10*s), dens_lvl, seed);
                problem = scaleCosts(problem,costS,scaleq);
                save(['r' upper('lasso') label],'problem');
                seed = seed + 1;
                
                % Linear program
                problem = lp(ceil(10*s), ceil(2*s), dens_lvl, seed);
                problem = scaleCosts(problem,costS,scaleq);
                save(['r' upper('lp') label],'problem');
                seed = seed + 1;
                
                % Nonnegative least-squares
                problem = nonneg_ls(ceil(10*s), ceil(2*s), dens_lvl, seed);
                problem = scaleCosts(problem,costS,scaleq);
                save(['r' upper('nonneg_ls') label],'problem');
                seed = seed + 1;
                
                % Portfolio optimization
                problem = portfolio(ceil(2*s), ceil(20*s), dens_lvl, seed);
                problem = scaleCosts(problem,costS,scaleq);
                save(['r' upper('portfolio') label],'problem');
                seed = seed + 1;
                
                % Support vector machine
                problem = svm(ceil(ceil(20*s)/2)*2, ceil(ceil(2*s)/2)*2, dens_lvl, seed);
                problem = scaleCosts(problem,costS,scaleq);
                save(['r' upper('svm') label],'problem');
                seed = seed + 1;
                
            end  %end scaleq
        end %end dens_lvl
    end %end costScaling
end %end problemSize

%move everything over to the random problems directory
fprintf('Moving files to ../random problems directory\n');
!mv r*.mat ../random

        
        
end

function problem = scaleCosts(problem,s,scaleq)

problem.P = problem.P.*s;
if(scaleq)
    problem.q = problem.q.*s;
end

end

