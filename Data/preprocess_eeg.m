function preprocessedData = preprocess_eeg(eegData, fs)
    % [Previous function content remains the same...]
end

%% Test and Visualization Section
% Load test data if available
testFile = fullfile('Data', 'Raw_Data', 'raw_eeg.mat');
if exist(testFile, 'file')
    load(testFile);
    fs = 256; % Example sampling rate
    
    % Test the function
    processedData = preprocess_eeg(eegData(1:10,:), fs); % Process first 10 channels
    
    % Visualization 1: Before/After Processing
    figure;
    subplot(2,1,1);
    plot(eegData(1, 1:500));
    title('Raw EEG (Channel 1)');
    xlabel('Samples');
    ylabel('Amplitude (\muV)');
    grid on;
    
    subplot(2,1,2);
    plot(processedData(1, 1:500));
    title('Processed EEG (Channel 1)');
    xlabel('Samples');
    ylabel('Amplitude (\muV)');
    grid on;
    
    % Visualization 2: PSD Comparison
    figure;
    [pxx_raw, f_raw] = pwelch(eegData(1,:), fs, fs/2, fs, fs);
    [pxx_proc, f_proc] = pwelch(processedData(1,:), fs, fs/2, fs, fs);
    
    semilogy(f_raw, pxx_raw, 'b', f_proc, pxx_proc, 'r');
    legend('Raw', 'Processed');
    title('Power Spectral Density Comparison');
    xlabel('Frequency (Hz)');
    ylabel('Power/Frequency (dB/Hz)');
    grid on;
    xlim([1 40]);
    
    % Display processing results
    disp('Preprocessing completed:');
    disp(['- Input data size: ', num2str(size(eegData))]);
    disp(['- Output data size: ', num2str(size(processedData))]);
    disp(['- NaN values in output: ', num2str(sum(isnan(processedData(:))))]);
end