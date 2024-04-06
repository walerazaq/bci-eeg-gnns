%% This code extracts node features for each adjacency matrix 

clear; close all;

addpath(genpath('/mnt/scratch2/users/asanni/MatCodes/'));

freqBand = 'Alpha';

switch freqBand
    case 'Alpha'
        data_path = '/mnt/scratch2/users/asanni/Alpha/';
    case 'Beta'
        data_path = '/mnt/scratch2/users/asanni/Beta/';
    case 'Delta'
        data_path = '/mnt/scratch2/users/asanni/Delta/';
    case 'Theta'
        data_path = '/mnt/scratch2/users/asanni/Theta/';
    case 'Gamma'
        data_path = '/mnt/scratch2/users/asanni/Gamma/';
    case 'overAll'
        data_path = '/mnt/scratch2/users/asanni/Overall/';
end

mat_files = dir([data_path,'*.mat']);
numel(mat_files)

%% Initialise cell arrays to store results
efficiency_results = cell(numel(mat_files), 1);
clusteringcoef_results = cell(numel(mat_files), 1);
strengths_results = cell(numel(mat_files), 1);
activity_results = cell(numel(mat_files), 1);
mobility_results = cell(numel(mat_files), 1);
complexity_results = cell(numel(mat_files), 1);

%% Start parfor loop
parfor idx = 1:numel(mat_files)
    file      = mat_files(idx).name(1:end-4);
    subject   = mat_files(idx).name(1:6);
    disp(['Analysing ',subject])

    %% Reading data
    data_cur = load([data_path,mat_files(idx).name]);

    % 1. Get Efficiency
    efficiency_results{idx} = efficiency_wei(data_cur.freqMatrix,2);

    % 2. Get Clustering Coefficients
    clusteringcoef_results{idx} = clustering_coef_wu(data_cur.freqMatrix);

    % 3. Get Strengths
    strengths_results{idx} = strengths_und(data_cur.freqMatrix);

    % 4. Get Hjorth Parameters
    [ACTIVITY, MOBILITY, COMPLEXITY] = hjorth(data_cur.freqMatrix,0);

    % Store Hjorth Parameters
    activity_results{idx} = ACTIVITY;
    mobility_results{idx} = MOBILITY;
    complexity_results{idx} = COMPLEXITY;
end

%% Save results from outside the parfor loop
for idx = 1:numel(mat_files)
    file = mat_files(idx).name(1:end-4);
    
    % Save Efficiency
    efficiency = efficiency_results{idx};
    save([data_path, 'Efficiency/', file, '.mat'], 'efficiency', '-v7.3', '-nocompression');

    % Save Clustering Coefficients
    clusteringcoef = clusteringcoef_results{idx};
    save([data_path, 'ClusteringCoef/', file, '.mat'], 'clusteringcoef', '-v7.3', '-nocompression');

    % Save Strengths
    strengths = strengths_results{idx};
    save([data_path, 'Strength/', file, '.mat'], 'strengths', '-v7.3', '-nocompression');

    % Save Hjorth Parameters
    ACTIVITY = activity_results{idx};
    save([data_path, 'Activity/', file, '.mat'], 'ACTIVITY', '-v7.3', '-nocompression');
    
    MOBILITY = mobility_results{idx};
    save([data_path, 'Mobility/', file, '.mat'], 'MOBILITY', '-v7.3', '-nocompression');
    
    COMPLEXITY = complexity_results{idx};
    save([data_path, 'Complexity/', file, '.mat'], 'COMPLEXITY', '-v7.3', '-nocompression');
end
