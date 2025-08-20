function filteredData = filter_data(rawData, fs)
    % Parameters
    lowCutoff = 0.5;    % Hz
    highCutoff = 40;     % Hz
    notchFreq = 50;      % Hz (power line frequency)
    
    % Bandpass filter
    [b, a] = butter(4, [lowCutoff highCutoff]/(fs/2), 'bandpass');
    filteredData = filtfilt(b, a, rawData);
    
    % Notch filter
    wo = notchFreq/(fs/2);
    [b, a] = iirnotch(wo, wo/35);
    filteredData = filtfilt(b, a, filteredData);
    
    % Optional: baseline correction
    filteredData = filteredData - mean(filteredData, 2);
end