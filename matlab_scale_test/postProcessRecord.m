function [stats,optStats] = postProcessRecord(stats)


%find the optimal rho for the problem
[minIter,optIdx] = min([stats.iter]);
rhoOpt = stats.rho(optIdx);
stats.rhoOpt = repmat(rhoOpt,[1 length(stats.iter)]);
stats.iterOpt = repmat(minIter,[1 length(stats.iter)]);

%for each problem, see if it was solvable for ANY rho
if(any([stats.status_val] == 1))
    stats.wasSolvable = true(size(stats.iter));
else
    stats.wasSolvable =  false(size(stats.iter));
end

%see if the optimal rho was an extreme value
if(minIter == stats.max_iter(1))
    stats.rhoOptIsExtreme = true(size(stats.iter));   %always true
elseif(optIdx == 1 || optIdx == length(stats.iter))
    stats.rhoOptIsExtreme = true(size(stats.iter));
else
    stats.rhoOptIsExtreme = false(size(stats.iter));
end

%try to find a range of 'good' rho values for this problem.
%Take this to be the range over which we are within a factor
%of 3 of the best one
rhoGoodLower = min(stats.rho( stats.iter <= 2*minIter ));
rhoGoodUpper = max(stats.rho( stats.iter <= 2*minIter ));

stats.rhoGoodLower = repmat(rhoGoodLower,[1 length(stats.iter)]);
stats.rhoGoodUpper = repmat(rhoGoodUpper,[1 length(stats.iter)]);


%make a structure with the optimal rho only
foo = @(M)M(optIdx);
optStats = structfun(foo,stats,'Uniform',0);






