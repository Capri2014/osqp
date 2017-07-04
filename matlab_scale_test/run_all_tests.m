d = dir('test*.m');

for i = 1:length(d)
    run(d(i).name); 
    [~,targetDir,ext] = fileparts(d(i).name);
    run_tests(fullfile('solutions',targetDir),'problems/random',osqpOptions,readOptions);
end