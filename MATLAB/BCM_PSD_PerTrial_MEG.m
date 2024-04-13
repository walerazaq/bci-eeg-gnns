clear; close all;

data_path = 'C:\Users\USER\Desktop\MEGData\PrepData\Session_02\';
save_path_BCM = 'C:\Users\USER\Desktop\MEGData\Training Data\';
save_path_PSD = 'C:\Users\USER\Desktop\MEGData\Training Data\PSD\';

mat_files = dir([data_path,'*.mat']);
numel(mat_files)

for idx = 1:numel(mat_files)
    file      = mat_files(idx).name(1:end-4);
    subject   = mat_files(idx).name(1:6);
    disp(['Analysing ',subject])

    %% Reading EEG data
    load([data_path,mat_files(idx).name]);

    for i = 1:numel(data.trial)

        class = data.trialinfo(i);

        %% Frequency analysis
        cfg            = [];
        cfg.method     = 'mtmfft';
        cfg.taper      = 'hanning';
        cfg.output     = 'fourier';
        cfg.foilim     = [8,30];
        cfg.trials  = i;
        freq        = ft_freqanalysis(cfg, data);
    
        %% Brain Connectivity matrix
        cfg            = [];
        cfg.method  = 'coh';
        cfg.complex = 'absimag';
        bcm_         = ft_connectivityanalysis(cfg, freq);
        BCM = mean(bcm_.cohspctrm,3); 
    
        %% Save Brain Connectivity matrix
        save([save_path_BCM,file,num2str(i),'_',num2str(class),'.mat'],'BCM','-v7.3','-nocompression')
        
    
        %% PSD analysis
        cfg            = [];
        cfg.method     = 'mtmfft';
        cfg.taper      = 'hanning';
        cfg.output     = 'pow';
        cfg.foilim     = [8,30];
        cfg.trials  = i;
        psd_        = ft_freqanalysis(cfg, data);
        PSD = mean(psd_.powspctrm, 2);
            
        %% Save PSD
        save([save_path_PSD,file,num2str(i),'_',num2str(class),'.mat'],'PSD','-v7.3','-nocompression')
    end

    clearvars -except data_path save_path_BCM save_path_PSD mat_files;
end