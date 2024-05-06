% Preprocessing using FieldTrip toolbox in MATLAB

clear; close all;

datapath = 'E:\Research\EEG\Thinking out loud\';
savepath = 'E:\Research\EEG\Thinking out loud\PreProcessed Files\';

files = dir([datapath,'*.bdf']);

for idx = 1:numel(files)
    fileName = files(idx).name(1:end-4);
    disp(['Preprocessing ',fileName])
    cfg = [];
    cfg.dataset = [datapath,files(idx).name];
    cfg.trialdef.eventtype  = 'STATUS';
    cfg.trialdef.eventvalue = [31 32 33 34];
    cfg.trialdef.prestim    = 0.5;
    cfg.trialdef.poststim   = 4;
    cfg = ft_definetrial(cfg);
    cfg.reref = 'yes';
    cfg.refchannel = {'EXG1','EXG2'};
    cfg.demean = 'yes';
    cfg.baselinewindow = [-0.5 0];
    cfg.dftfilter = 'yes';
    cfg.bpfilter = 'yes';
    cfg.dftfreq = 50;
    cfg.bpfreq = [0.5 100];
    data = ft_preprocessing(cfg);

    data.classLabel = {'31','UP'; '32','Down'; '33','Right';'34','Left'};

    save([savepath,fileName,'.mat'],'data','-v7.3','-nocompression')
end

