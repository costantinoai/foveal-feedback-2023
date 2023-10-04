
% Define paths to directories
baseDir = './data/BIDS';
BIDSdir = fullfile(baseDir);
SPMdir  = fullfile(baseDir, 'derivatives', 'SPMglm-fmriprep');
fMRIprepDir = fullfile(baseDir, 'derivatives', 'fMRIprep');

% Define classes to be processed. With 1 we process only the all vs 
% all model, with 2 we also process the faces vs vehicles model
GLMnum = [1, 2];

% Loop over specified classes
for classType = GLMnum
    
    % Define the model output directory based on the class type
    rsaDir = 'desc-bike+car+female+male';
    if classType == 2
        rsaDir = 'desc-faces+vehicles';
    end

    % Loop over subjects (2 to 25)
    for subjNum = 2:25
        subStr = sprintf('sub-%02d', subjNum);
        fprintf('##########\nSTEP: running %s\n##########\n', subStr);

        % Directories for subject's data and output
        eventsDir = fullfile(BIDSdir, subStr, 'func');
        subDir  = fullfile(fMRIprepDir, subStr, 'func');
        outDir  = fullfile(SPMdir, subStr, 'func',  rsaDir);

        % Number of runs for the current subject
        runs = length(dir(fullfile(eventsDir, '*-exp_*.tsv')));

        % Set SPM defaults and initialize job manager
        spm('defaults','fmri');
        spm_jobman('initcfg');

        % General SPM settings
        matlabbatch{1}.spm.stats.fmri_spec.dir = {outDir};
        matlabbatch{1}.spm.stats.fmri_spec.timing.units    = 'secs';
        matlabbatch{1}.spm.stats.fmri_spec.timing.RT       = 2;
        matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t   = 16;
        matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t0  = 8;

        % Specify scans, multi-condition files, and regressors for each run
        for run = 1:runs
            boldFile = fullfile(subDir, sprintf('%s_task-exp_run-%d_space-MNI152NLin2009cAsym_desc-preproc_bold.nii', subStr, run));
            blockFileSuffix = 'bike+car+female+male_SPMmulticondition.mat';
            if classType == 2
                blockFileSuffix = 'faces+vehicles_SPMmulticondition.mat';
            end
            blockFile = fullfile(BIDSdir, subStr, 'func', sprintf('%s_task-exp_run-%d_desc-%s', subStr, run, blockFileSuffix));
            confoundFile = fullfile(subDir, sprintf('%s_task-exp_run-%d_desc-6HMP_regressors.txt', subStr, run));

            matlabbatch{1}.spm.stats.fmri_spec.sess(run).scans = spm_select('expand', boldFile);
            matlabbatch{1}.spm.stats.fmri_spec.sess(run).multi = {blockFile};
            matlabbatch{1}.spm.stats.fmri_spec.sess(run).multi_reg = {confoundFile};
            matlabbatch{1}.spm.stats.fmri_spec.sess(run).hpf = 128;
        end

        % Other SPM settings
        matlabbatch{1}.spm.stats.fmri_spec.fact = struct('name', {}, 'levels', {});
        matlabbatch{1}.spm.stats.fmri_spec.bases.hrf.derivs = [0 0];
        matlabbatch{1}.spm.stats.fmri_spec.volt = 1;
        matlabbatch{1}.spm.stats.fmri_spec.global = 'None';
        matlabbatch{1}.spm.stats.fmri_spec.mthresh = 0.8;
        matlabbatch{1}.spm.stats.fmri_spec.mask = {''};
        matlabbatch{1}.spm.stats.fmri_spec.cvi = 'AR(1)';

        % Model estimation settings
        matlabbatch{2}.spm.stats.fmri_est.spmmat(1) = cfg_dep('fMRI model specification: SPM.mat File', ...
            substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), ...
            substruct('.','spmmat'));
        matlabbatch{2}.spm.stats.fmri_est.write_residuals = 0;
        matlabbatch{2}.spm.stats.fmri_est.method.Classical = 1;

        % Run the job
        spm_jobman('run',matlabbatch);
    end
end
