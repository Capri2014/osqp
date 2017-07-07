d = dir('tests/test*.m');

for i = 1:length(d)
    clear osqpOptions readOptions;
    run(fullfile('tests',d(i).name)); 
    [~,targetDir,ext] = fileparts(d(i).name);
    run_tests(fullfile('solutions',targetDir),'problems/random',osqpOptions,readOptions);
end