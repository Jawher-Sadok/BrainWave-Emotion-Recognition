function features = extract_features(eegData, fs)
    numChannels = size(eegData, 1);
    features = [];
    
    % Time-domain features
    tdFeatures = [
        mean(eegData, 2), std(eegData, [], 2), ...
        skewness(eegData, [], 2), kurtosis(eegData, [], 2), ...
        mad(eegData, [], 2), rms(eegData, 2)
    ];
    
    % Frequency-domain features
    fdFeatures = zeros(numChannels, 5); % 5 frequency bands
    bandRanges = [0.5 4; 4 8; 8 13; 13 30; 30 40]; % delta, theta, alpha, beta, gamma
    
    for ch = 1:numChannels
        [psd, freq] = pwelch(eegData(ch, :), fs, fs/2, fs, fs);
        for b = 1:size(bandRanges, 1)
            fdFeatures(ch, b) = bandpower(psd, freq, bandRanges(b, :), 'psd');
        end
    end
    
    % Nonlinear features
    nlFeatures = zeros(numChannels, 3);
    for ch = 1:numChannels
        nlFeatures(ch, 1) = approximate_entropy(eegData(ch, :), 2, 0.2*std(eegData(ch, :)));
        nlFeatures(ch, 2) = hjorth_mobility(eegData(ch, :));
        nlFeatures(ch, 3) = hjorth_complexity(eegData(ch, :));
    end
    
    % Combine all features
    features = [tdFeatures, fdFeatures, nlFeatures];
end