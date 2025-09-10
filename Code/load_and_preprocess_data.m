function [features, labels, featureNames] = load_and_preprocess_data(filename)
    % LOAD_AND_PREPROCESS_DATA Load and preprocess emotion data from CSV file
    % Input: filename - path to CSV file
    % Output: features - normalized feature matrix
    %         labels - emotion labels (1 for Relaxed, 2 for Funny/Happy)
    %         featureNames - names of all features
    
    
    % Load data from CSV
    data = readtable(filename);
    
    % Display data information
    fprintf('Loading data from: %s\n', filename);
    fprintf('Dataset size: %d samples, %d features\n', size(data, 1), size(data, 2)-2);
    
    % Extract feature names (exclude label columns)
    featureNames = data.Properties.VariableNames;
    labelCols = contains(featureNames, {'Relaxed', 'Funny_Happy'}, 'IgnoreCase', true);
    featureNames = featureNames(~labelCols);
    % Remove specific features
    featuresToRemove = {'MeanDelta10sec', 'BetaAveraged', 'SumDelta10sec','DeltaAveraged'};
    
    % Find indices of features to keep
    allFeatures = data.Properties.VariableNames;
    labelCols = contains(allFeatures, {'Relaxed', 'Funny_Happy'}, 'IgnoreCase', true);
    featureNames = allFeatures(~labelCols);
    
    % Get indices of features to remove from the feature list
    [~, idxToRemove] = intersect(featureNames, featuresToRemove, 'stable');
    featuresToKeep = setdiff(1:length(featureNames), idxToRemove);
    
    % Update feature names
    featureNames = featureNames(featuresToKeep);
    
    % Extract only the selected features and labels
    features = table2array(data(:, featuresToKeep));
    
    % Create binary labels: 1 for Relaxed, 2 for Funny/Happy
    relaxed = data.Relaxed;
    funny_happy = data.('Funny_Happy');
    
    % Ensure only one label is active per sample
    if any(relaxed & funny_happy)
        warning('Some samples have both labels active. Using priority: Relaxed > Funny_Happy');
        labels = ones(size(relaxed)) * 2; % Default to Funny/Happy
        labels(relaxed == 1) = 1; % Override with Relaxed where applicable
    else
        labels = relaxed + (funny_happy * 2);
        labels(labels == 0) = 2; % Handle cases where neither is explicitly 1
    end
    
    % Remove any rows with NaN values
    nanRows = any(isnan(features), 2) | isnan(labels);
    if any(nanRows)
        fprintf('Removing %d rows with NaN values\n', sum(nanRows));
        features = features(~nanRows, :);
        labels = labels(~nanRows);
    end
    
    % Normalize features (z-score normalization)
    features = normalize(features);
    
    % Display class distribution
    fprintf('Class distribution:\n');
    fprintf('  Relaxed: %d samples\n', sum(labels == 1));
    fprintf('  Funny/Happy: %d samples\n', sum(labels == 2));
    
    % Visualize feature distributions
    visualize_feature_distributions(features, labels, featureNames);
end

function visualize_feature_distributions(features, labels, featureNames)
    % Visualize distributions of features by class
    
    figure('Name', 'Feature Distributions by Class', 'Position', [100, 100, 1200, 800]);
    
    nFeatures = size(features, 2);
    nCols = 4;
    nRows = ceil(nFeatures / nCols);
    
    for i = 1:nFeatures
        subplot(nRows, nCols, i);
        
        % Plot histograms for each class
        histogram(features(labels == 1, i), 'FaceAlpha', 0.6, 'FaceColor', 'blue');
        hold on;
        histogram(features(labels == 2, i), 'FaceAlpha', 0.6, 'FaceColor', 'red');
        hold off;
        
        title(featureNames{i});
        xlabel('Normalized Value');
        ylabel('Frequency');
        legend('Relaxed', 'Funny_Happy');
        grid on;
    end
    
    sgtitle('Feature Distributions by Emotion Class');
end