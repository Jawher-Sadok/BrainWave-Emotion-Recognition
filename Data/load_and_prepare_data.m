function [eegData, labels] = load_and_prepare_data(filename)
    % Read the CSV file while preserving original headers
    opts = detectImportOptions(filename);
    opts.VariableNamingRule = 'preserve'; % Keep original column names
    dataTable = readtable(filename, opts);
    
    % Extract EEG data (all columns except the last)
    eegData = table2array(dataTable(:, 1:end-1));
    
    % Extract labels (last column)
    labelStrings = table2array(dataTable(:, end));
    
    % Convert labels to categorical
    validLabels = {'POSITIVE', 'NEUTRAL', 'NEGATIVE'};
    labels = categorical(labelStrings, validLabels, 'Protected', true);
    
    % Create directory if it doesn't exist
    if ~exist(fullfile('Data', 'Raw_Data'), 'dir')
        mkdir(fullfile('Data', 'Raw_Data'));
    end
    
    % Save raw data
    save(fullfile('Data', 'Raw_Data', 'raw_eeg.mat'), 'eegData', 'labels');
end

%% Test and Visualization Section
if exist('emotions.csv', 'file')
    % Test the function
    [testData, testLabels] = load_and_prepare_data('emotions.csv');
    
    % Visualization 1: Show data distribution
    figure;
    histogram(testLabels);
    title('Class Distribution');
    xlabel('Emotion Classes');
    ylabel('Count');
    grid on;
    
    % Visualization 2: Plot sample EEG channels
    figure;
    plot(testData(1:5, 1:100)'); % First 5 channels, first 100 samples
    title('Sample EEG Channels (Raw)');
    xlabel('Time (samples)');
    ylabel('Amplitude (\muV)');
    legend('Channel 1', 'Channel 2', 'Channel 3', 'Channel 4', 'Channel 5');
    grid on;
    
    % Display summary
    disp(['Loaded data with ', num2str(size(testData,1)), ' channels and ', ...
          num2str(size(testData,2)), ' samples per channel']);
    disp('Class counts:');
    tabulate(testLabels);
end