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

% A loop to compute pre-solve statistics
fprintf('\nStarting %s pre-solve\n',targetDir);
parfor i = 1:length(fnames)
    theFile = fullfile(readDir,fnames{i});
    [~,root,ext] = fileparts(theFile);
    target  = fullfile(targetDir,[root '_preSolve',ext]);
    
    %look for a saved record in solution_data
    if(exist(target,'file'))
        %skip this one
        %fprintf('%i/%i -- %s : preSolve -  skipping   \n',i, length(fnames),target);
        fprintf('.');
    else
        fprintf('\n%i/%i -- %s : preSolve - processing',i, length(fnames), target);
        stats = makeRecord(theFile,rhoVals,sigVals,osqpOptions,readOptions,true);
        export_file(target,stats);
    end
    
end


%% A Loop to actually solve the problems
fprintf('\nStarting %s solve\n',targetDir);
parfor i = 1:length(fnames)
    theFile = fullfile(readDir,fnames{i});
    [~,root,ext] = fileparts(theFile);
    target  = fullfile(targetDir,[root '_postSolve',ext]);
    
    %look for a saved record in solution_data
    if(exist(target,'file'))
        %skip this one
        %fprintf('%i/%i -- %s : Solve - skipping   \n',i, length(fnames),target);
        fprintf('.');
    else
        fprintf('\n%i/%i -- %s : Solve - processing',i, length(fnames), target);
        stats = makeRecord(theFile,rhoVals,sigVals,osqpOptions,readOptions,false);
        export_file(target,stats);
    end
    
end


%% A loop to merge pre- and post- solve data
fprintf('\nStarting %s merge\n',targetDir);
parfor i = 1:length(fnames)
    theFile = fullfile(readDir,fnames{i});
    [~,root,ext]  = fileparts(theFile);
    targetFile    = fullfile(targetDir,[root '_sol',ext]);
    preSolveFile  = fullfile(targetDir,[root '_preSolve',ext]);
    postSolveFile = fullfile(targetDir,[root '_postSolve',ext]);
    
    %look for a saved record in solution_data
    if(~exist(targetFile,'file') || (fileIsOlder(targetFile,preSolveFile) | fileIsOlder(targetFile,preSolveFile) ) )
        fprintf('\n%i/%i -- %s : Merge - processing ',i, length(fnames), targetFile);
        merge_files(preSolveFile,postSolveFile,targetFile);
        
    else
        %skip this one
        %fprintf('%i/%i -- %s : Merge - skipping   \n',i, length(fnames),targetFile);
        fprintf('.');
    end
    

end



function  [stats,optStats] = merge_files(preSolveFile,postSolveFile,targetFile)

pre  = load(preSolveFile,'stats');
post = load(postSolveFile,'stats');
npost     = length(post.stats);

%flatten the structures
preStats  = flattenStructArray(pre.stats);
postStats = flattenStructArray(post.stats);

%merge all of the postStats fields into the preStats structure
stats  = preStats;
fnames = fieldnames(postStats);
for i = 1:length(fnames)
    stats = setfield(stats,fnames{i},getfield(postStats,fnames{i}));
end

%for any fields with a single value (i.e. those from pre-processing),
%replicate the values to make vectorization easier
fnames = fieldnames(stats);
for i = 1:length(fnames)
    fld   = getfield(stats,fnames{i});
    if(length(fld) == 1)
        stats = setfield(stats,fnames{i},repmat(fld,[1 npost]));
    end
end


[stats,optStats]  = postProcessRecord(stats);
[~,sourceFile]    = fileparts(regexprep(targetFile,'_sol',''));
optStats.filename = sourceFile;
export_file(targetFile,stats,optStats);


function export_file(target,stats,optStats)

%workaround indirection function to prevent parfor loop from bombing
if(nargin == 2)
    save(target,'stats');
else
    save(target,'stats','optStats');
end
    


function out = fileIsOlder(file1, file2)

% check to see if file1 is older than file2.  Assumes both files exist
d1 = dir(file1);
d2 = dir(file2);
out = d1.datenum < d2.datenum;




