clear; close all;

data_path = '/mnt/scratch2/users/asanni/dataNEW/';

mat_files = dir([data_path,'*.mat']);
numel(mat_files)

freqBand = 'Alpha';

switch freqBand
    case 'Alpha'
        freqRange = [8,13];
        save_path_BCM = '/mnt/scratch2/users/asanni/EEG/AlphaTrials/';
        save_path_PSD = '/mnt/scratch2/users/asanni/EEG/AlphaTrials/PSD/';
    case 'Beta'
        freqRange = [13,30];
        save_path_BCM = '/mnt/scratch2/users/asanni/EEG/BetaTrials/';
        save_path_PSD = '/mnt/scratch2/users/asanni/EEG/BetaTrials/PSD/';
    case 'Delta'
        freqRange = [0.5,4];
        save_path_BCM = '/mnt/scratch2/users/asanni/EEG/DeltaTrials/';
        save_path_PSD = '/mnt/scratch2/users/asanni/EEG/DeltaTrials/PSD/';
    case 'Theta'
        freqRange = [4,7];
        save_path_BCM = '/mnt/scratch2/users/asanni/EEG/ThetaTrials/';
        save_path_PSD = '/mnt/scratch2/users/asanni/EEG/ThetaTrials/PSD/';
    case 'Gamma'
        freqRange = [30,45];
        save_path_BCM = '/mnt/scratch2/users/asanni/EEG/GammaTrials/';
        save_path_PSD = '/mnt/scratch2/users/asanni/EEG/GammaTrials/PSD/';
    case 'overAll'
        freqRange = [0.5,45];
        save_path_BCM = '/mnt/scratch2/users/asanni/EEG/Overall/';
        save_path_PSD = '/mnt/scratch2/users/asanni/EEG/Overall/PSD/';
end

for idx = 1:numel(mat_files)
    file      = mat_files(idx).name(1:end-4);
    subject   = mat_files(idx).name(1:6);
    disp(['Analysing ',subject])

    %% Reading EEG data
    load([data_path,mat_files(idx).name]);

    for i = 1:numel(dataNEW.trial)

        class = dataNEW.trialinfo(i);

        %% Frequency analysis
        cfg            = [];
        cfg.method     = 'mtmfft';
        cfg.taper      = 'hanning';
        cfg.output     = 'fourier';
        cfg.keeptrials = 'yes';
        cfg.foilim     = freqRange;
        cfg.trials  = i;
        cfg.toi          = 0.5:0.05:3;  
        freq        = ft_freqanalysis(cfg, dataNEW);
    
        %% Brain Connectivity matrix
        cfg            = [];
        cfg.method  = 'coh';
        cfg.complex = 'imag';
        bcm_         = ft_connectivityanalysis(cfg, freq);
        BCM = mean(bcm_.cohspctrm,3);
        BCM = abs(BCM); 
    
        %% Save Brain Connectivity matrix
        save([save_path_BCM,file,num2str(i),'_',num2str(class),'.mat'],'BCM','-v7.3','-nocompression')
        
    
        %% PSD analysis
        cfg            = [];
        cfg.method     = 'mtmfft';
        cfg.taper      = 'hanning';
        cfg.output     = 'pow';
        cfg.keeptrials = 'no';
        cfg.foilim     = freqRange;
        cfg.trials  = i;
        cfg.toi          = 0.5:0.05:3; 
        psd_        = ft_freqanalysis(cfg, dataNEW);
        PSD = mean(psd_.powspctrm, 2);
            
        %% Save PSD
        save([save_path_PSD,file,num2str(i),'_',num2str(class),'.mat'],'PSD','-v7.3','-nocompression')
    end

    clearvars -except data_path save_path_BCM save_path_PSD mat_files freqRange freqBand;
end
