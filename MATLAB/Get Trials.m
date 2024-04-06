%% This code extracts the 80 Inner Speech Trials from the EEG Data 

clear; close all;

data_path = '/mnt/scratch2/users/asanni/data/';
save_path = '/mnt/scratch2/users/asanni/dataNEW/';

mat_files = dir([data_path,'*.mat']);
numel(mat_files)

for idx = 1:numel(mat_files)
    file      = mat_files(idx).name(1:end-4);
    subject   = mat_files(idx).name(1:6);
    disp(['Analysing ',subject])

    %% Reading EEG data
    load([data_path,mat_files(idx).name]);

    %% Select data
    cfg = [];
    cfg.trials = 41:120;
    dataNEW = ft_selectdata(cfg, data);
    
    %% Save new data
    save([save_path,file,'.mat'],'dataNEW','-v7.3','-nocompression')
end