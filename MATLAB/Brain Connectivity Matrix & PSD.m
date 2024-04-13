%% This code gets the Brain Connectivity Matrix  of EEG signals and their Power Spectral Densities

clear; close all;

data_path = '/mnt/scratch2/users/asanni/dataNEW/';

mat_files = dir([data_path,'*.mat']);
numel(mat_files)

freqBand = 'Alpha';

switch freqBand
    case 'Alpha'
        freqRange = [8,13];
        save_path_BCM = '/mnt/scratch2/users/asanni/Alpha/';
        save_path_PSD = '/mnt/scratch2/users/asanni/Alpha/PSD/';
    case 'Beta'
        freqRange = [13,30];
        save_path_BCM = '/mnt/scratch2/users/asanni/Beta/';
        save_path_PSD = '/mnt/scratch2/users/asanni/Beta/PSD/';
    case 'Delta'
        freqRange = [0.5,4];
        save_path_BCM = '/mnt/scratch2/users/asanni/Delta/';
        save_path_PSD = '/mnt/scratch2/users/asanni/Delta/PSD/';
    case 'Theta'
        freqRange = [4,7];
        save_path_BCM = '/mnt/scratch2/users/asanni/Theta/';
        save_path_PSD = '/mnt/scratch2/users/asanni/Theta/PSD/';
    case 'Gamma'
        freqRange = [30,45];
        save_path_BCM = '/mnt/scratch2/users/asanni/Gamma/';
        save_path_PSD = '/mnt/scratch2/users/asanni/Gamma/PSD/';
    case 'overAll'
        freqRange = [0.5,45];
        save_path_BCM = '/mnt/scratch2/users/asanni/Overall/';
        save_path_PSD = '/mnt/scratch2/users/asanni/Overall/PSD/';
end

for idx = 1:numel(mat_files)
    file      = mat_files(idx).name(1:end-4);
    subject   = mat_files(idx).name(1:6);
    disp(['Analysing ',subject])

    %% Reading EEG data
    load([data_path,mat_files(idx).name]);

    unique_labels = unique(dataNEW.trialinfo);

    for i = 1:numel(unique_labels)

        %% Frequency analysis
        cfg            = [];
        cfg.method     = 'mtmfft';
        cfg.taper      = 'hanning';
        cfg.output     = 'fourier';
        cfg.keeptrials = 'yes';
        cfg.foilim     = freqRange;
        cfg.trials  = dataNEW.trialinfo == unique_labels(i);
        cfg.toi          = 0.5:0.05:3;  
        freq        = ft_freqanalysis(cfg, dataNEW);

        %% Brain Connectivity matrix
        cfg            = [];
        cfg.method  = 'coh';
        cfg.complex = 'absimag';
        bcm_         = ft_connectivityanalysis(cfg, freq);
        BCM = mean(bcm_.cohspctrm,3); 

        %% Save Brain Connectivity matrix
        save([save_path_BCM,file,num2str(unique_labels(i)),'.mat'],'BCM','-v7.3','-nocompression')
        

        %% PSD analysis
        cfg            = [];
        cfg.method     = 'mtmfft';
        cfg.taper      = 'hanning';
        cfg.output     = 'pow';
        cfg.keeptrials = 'no';
        cfg.foilim     = freqRange;
        cfg.trials  = dataNEW.trialinfo == unique_labels(i);
        cfg.toi          = 1:0.05:3.5; 
        psd_        = ft_freqanalysis(cfg, dataNEW);
        PSD = mean(psd_.powspctrm, 2);
            
        %% Save PSD
        save([save_path_PSD,file,num2str(unique_labels(i)),'.mat'],'PSD','-v7.3','-nocompression')

    end
    clearvars -except data_path save_path_BCM save_path_PSD mat_files freqRange freqBand;
end
