function run_tests(targetDir, readDir,osqpOptions,readOptions)

%make the target if it doesn't exist
if(~exist(targetDir,'dir')), mkdir(targetDir); end

%save all of the options to the targetdir
save(fullfile(targetDir,'solveOptions.mat'),'osqpOptions','readOptions');

%create the problem solution grids
rhoVals = logspace(-5,4,40);
sigVals = logspace(-7,3,11);
sigVals = sigVals(1);

d      = dir(fullfile(readDir,'r*.mat'));
fnames = {d.name};
sizes  = [d.bytes];
[~,idx] = sort(sizes);
fnames  = fnames(idx(1:450)); %subset of problems, in order of size

for i = 1:length(fnames)
    theFile = fullfile(readDir,fnames{i});
    [~,root,ext] = fileparts(theFile);
    target  = fullfile(targetDir,[root '_sol',ext]);
    
    %look for a saved record in solution_data
    if(exist(target,'file'))
        %skip this one
        fprintf('%i/%i -- %s : skipping   \n',i, length(fnames),target);
    else
        fprintf('%i/%i -- %s : processing \n',i, length(fnames), target);
        [stats,optStats] = makeRecord(theFile,rhoVals,sigVals,osqpOptions,readOptions);
        stats.filename = fnames{i};
        optStats.filename = fnames{i};
        save(target,'stats','optStats');
    end
    
end





