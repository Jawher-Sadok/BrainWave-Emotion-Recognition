function features = extract_features(eegData, fs)
    % [Previous function content remains the same...]
end

%% Test and Visualization Section
% Load test data if available
testFile = fullfile('Data', 'Preprocessed_Data', 'preprocessed_eeg.mat');
if exist(testFile, 'file')
    load(testFile);
    fs = 256; % Example sampling rate
    
    % Test the function with first 5 channels
    testFeatures = extract_features(preprocessedData(1:5,:), fs);
    
    % Visualization 1: Feature Heatmap
    figure;
    imagesc(testFeatures);
    colorbar;
    title('Feature Values Across Channels');
    xlabel('Feature Index');
    ylabel('Channel Number');
    xticks(1:size(testFeatures,2));
    xticklabels({'Mean','STD','Skew','Kurt','RMS','Delta','Theta','Alpha','Beta','Gamma','ApEn','Hurst','Mobility','Complexity','ZCR'});
    yticks(1:size(testFeatures,1));
    
    % Visualization 2: Band Power Comparison
    figure;
    bandPowerFeatures = testFeatures(:,6:10);
    bar(bandPowerFeatures');
    title('Band Power Features');
    xlabel('Frequency Band');
    ylabel('Power');
    xticklabels({'Delta','Theta','Alpha','Beta','Gamma'});
    legend('Channel 1','Channel 2','Channel 3','Channel 4','Channel 5');
    grid on;
    
    % Display feature statistics
    disp('Extracted feature statistics:');
    disp(array2table([mean(testFeatures); std(testFeatures); min(testFeatures); max(testFeatures)], ...
        'RowNames', {'Mean', 'STD', 'Min', 'Max'}, ...
        'VariableNames', {'Mean','STD','Skew','Kurt','RMS','Delta','Theta','Alpha','Beta','Gamma','ApEn','Hurst','Mobility','Complexity','ZCR'}));
end