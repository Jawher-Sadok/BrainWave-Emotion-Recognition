function cleanData = artifact_removal(filteredData, fs)
    % 1. Detect bad channels
    channelStd = std(filteredData, [], 2);
    badChannels = channelStd > 5*median(channelStd);
    
    % 2. Interpolate bad channels
    goodChannels = find(~badChannels);
    badChannels = find(badChannels);
    cleanData = filteredData;
    
    for ch = badChannels'
        cleanData(ch, :) = interp1(goodChannels, filteredData(goodChannels, :), ...
                           ch, 'spline');
    end
    
    % 3. ICA for artifact removal
    [weights, sphere] = runica(cleanData);
    icaAct = weights * sphere * cleanData;
    
    % Automatic component rejection (simple threshold)
    componentKurt = kurtosis(icaAct, [], 2);
    badComponents = componentKurt > 5;
    icaAct(badComponents, :) = 0;
    
    % Reconstruct clean data
    cleanData = pinv(weights * sphere) * icaAct;
end