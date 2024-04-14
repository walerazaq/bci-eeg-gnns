clear; close all;

data_path = 'C:\Users\USER\Desktop\MEGData\PrepData\Session_01\';
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
    j = 1;

    for i = 1:numel(data1.trial)

        if data1.trialinfo(i) == 1 || data1.trialinfo(i) == 2
            class = data1.trialinfo(i);
    
            %% Frequency analysis
            cfg            = [];
            cfg.method     = 'mtmfft';
            cfg.taper      = 'hanning';
            cfg.output     = 'fourier';
            cfg.foilim     = [8,30];
            cfg.trials  = i;
            freq        = ft_freqanalysis(cfg, data1);
        
            %% Brain Connectivity matrix
            cfg            = [];
            cfg.method  = 'coh';
            cfg.complex = 'imag';
            bcm_         = ft_connectivityanalysis(cfg, freq);
            BCM = mean(bcm_.cohspctrm,3); 
        
            %% Save Brain Connectivity matrix
            save([save_path_BCM,file,num2str(j),'_',num2str(class),'.mat'],'BCM','-v7.3','-nocompression')
            
        
            %% PSD analysis
            cfg            = [];
            cfg.method     = 'mtmfft';
            cfg.taper      = 'hanning';
            cfg.output     = 'pow';
            cfg.foilim     = [8,30];
            cfg.trials  = i;
            psd_        = ft_freqanalysis(cfg, data1);
            PSD = mean(psd_.powspctrm, 2);
                
            %% Save PSD
            save([save_path_PSD,file,num2str(j),'_',num2str(class),'.mat'],'PSD','-v7.3','-nocompression')
            j = j + 1;
        end
    end

    clearvars -except data_path save_path_BCM save_path_PSD mat_files;
end
