function [importanceScores, rankedFeatures] = Feature_Importance(data, target, varargin)
    % FEATURE_IMPORTANCE - Comprehensive feature importance analysis for EEG classification
    %
    % Inputs:
    %   data - Feature matrix (samples × features)
    %   target - Target labels (categorical or numeric)
    %   Optional parameters:
    %       'Method' - 'permutation', 'gini', 'mrmr', 'relieff', 'all'
    %       'NumImportant' - Number of top features to return (default: 20)
    %       'NumPermutations' - Number of permutations (default: 100)
    %       'Visualize' - Show plots (true/false, default: true)
    %
    % Outputs:
    %   importanceScores - Structure with importance scores from different methods
    %   rankedFeatures - Feature indices ranked by importance

    % Parse input parameters
    p = inputParser;
    addRequired(p, 'data', @ismatrix);
    addRequired(p, 'target', @(x) iscategorical(x) || isnumeric(x));
    addParameter(p, 'Method', 'all', @ischar);
    addParameter(p, 'NumImportant', 20, @isnumeric);
    addParameter(p, 'NumPermutations', 100, @isnumeric);
    addParameter(p, 'Visualize', true, @islogical);
    parse(p, data, target, varargin{:});
    
    method = lower(p.Results.Method);
    numImportant = p.Results.NumImportant;
    numPermutations = p.Results.NumPermutations;
    visualize = p.Results.Visualize;
    
    % Convert target to categorical if numeric
    if isnumeric(target)
        target = categorical(target);
    end
    
    % Initialize results structure
    importanceScores = struct();
    rankedFeatures = struct();
    
    fprintf('Performing feature importance analysis...\n');
    fprintf('Dataset: %d samples, %d features\n', size(data, 1), size(data, 2));
    
    %% 1. Permutation Importance
    if ismember(method, {'all', 'permutation'})
        fprintf('Calculating permutation importance...\n');
        importanceScores.permutation = permutation_importance(data, target, numPermutations);
        [~, rankedFeatures.permutation] = sort(importanceScores.permutation, 'descend');
    end
    
    %% 2. Gini Importance (Random Forest)
    if ismember(method, {'all', 'gini'})
        fprintf('Calculating Gini importance...\n');
        importanceScores.gini = gini_importance(data, target);
        [~, rankedFeatures.gini] = sort(importanceScores.gini, 'descend');
    end
    
    %% 3. MRMR (Minimum Redundancy Maximum Relevance)
    if ismember(method, {'all', 'mrmr'})
        fprintf('Calculating MRMR scores...\n');
        try
            [idx, scores] = fscmrmr(data, target);
            importanceScores.mrmr = zeros(size(data, 2), 1);
            importanceScores.mrmr(idx) = scores;
            rankedFeatures.mrmr = idx;
        catch
            warning('MRMR computation failed. Using correlation instead.');
            importanceScores.mrmr = correlation_importance(data, target);
            [~, rankedFeatures.mrmr] = sort(importanceScores.mrmr, 'descend');
        end
    end
    
    %% 4. ReliefF
    if ismember(method, {'all', 'relieff'})
        fprintf('Calculating ReliefF scores...\n');
        try
            [idx, weights] = relieff(data, double(target), 10, 'categoricalx', false);
            importanceScores.relieff = zeros(size(data, 2), 1);
            importanceScores.relieff(idx) = weights;
            rankedFeatures.relieff = idx;
        catch
            warning('ReliefF computation failed. Using alternative method.');
            importanceScores.relieff = mutual_info_importance(data, target);
            [~, rankedFeatures.relieff] = sort(importanceScores.relieff, 'descend');
        end
    end
    
    %% 5. Correlation-based Importance
    if ismember(method, {'all', 'correlation'})
        fprintf('Calculating correlation importance...\n');
        importanceScores.correlation = correlation_importance(data, target);
        [~, rankedFeatures.correlation] = sort(importanceScores.correlation, 'descend');
    end
    
    %% Visualization
    if visualize
        visualize_feature_importance(importanceScores, rankedFeatures, numImportant);
    end
    
    %% Create consensus ranking
    fprintf('Creating consensus feature ranking...\n');
    rankedFeatures.consensus = create_consensus_ranking(rankedFeatures, numImportant);
    
    fprintf('Feature importance analysis completed.\n');
end

%% Helper Functions

function scores = permutation_importance(data, target, numPermutations)
    % Permutation importance using Random Forest
    rng(42); % For reproducibility
    
    % Train baseline model
    mdl = fitcensemble(data, target, 'Method', 'Bag', 'NumLearningCycles', 50);
    baselineAccuracy = 1 - loss(mdl, data, target, 'LossFun', 'classiferror');
    
    scores = zeros(size(data, 2), 1);
    
    parfor i = 1:size(data, 2)
        tempData = data;
        permutedAccuracy = zeros(numPermutations, 1);
        
        for p = 1:numPermutations
            % Permute the feature
            tempData(:, i) = data(randperm(size(data, 1)), i);
            
            % Calculate accuracy with permuted feature
            permutedAccuracy(p) = 1 - loss(mdl, tempData, target, 'LossFun', 'classiferror');
        end
        
        % Importance is the decrease in accuracy
        scores(i) = baselineAccuracy - mean(permutedAccuracy);
    end
end

function scores = gini_importance(data, target)
    % Gini importance from Random Forest
    mdl = fitcensemble(data, target, 'Method', 'Bag', 'NumLearningCycles', 100);
    
    if isa(mdl, 'ClassificationBaggedEnsemble')
        scores = oobPermutedPredictorImportance(mdl);
    else
        % Fallback: use mean decrease in impurity
        scores = predictorImportance(mdl);
    end
end

function scores = correlation_importance(data, target)
    % Correlation-based feature importance
    numClasses = length(unique(target));
    scores = zeros(size(data, 2), 1);
    
    if numClasses == 2
        % Binary classification: point-biserial correlation
        targetNumeric = double(target == target(1));
        for i = 1:size(data, 2)
            corrMatrix = corrcoef(data(:, i), targetNumeric);
            scores(i) = abs(corrMatrix(1, 2));
        end
    else
        % Multiclass: ANOVA F-value
        for i = 1:size(data, 2)
            [~, tbl] = anova1(data(:, i), target, 'off');
            scores(i) = tbl{2, 5}; % F-statistic
        end
    end
end

function scores = mutual_info_importance(data, target)
    % Mutual information-based importance
    scores = zeros(size(data, 2), 1);
    
    for i = 1:size(data, 2)
        try
            scores(i) = mutualinfo(data(:, i), double(target));
        catch
            % Fallback to correlation if mutual info fails
            scores(i) = abs(corr(data(:, i), double(target)));
        end
    end
end

function visualize_feature_importance(importanceScores, rankedFeatures, numImportant)
    % Create comprehensive visualization of feature importance
    
    methods = fieldnames(importanceScores);
    numMethods = length(methods);
    
    % Create figure with subplots
    figure('Position', [100, 100, 1200, 800], 'Name', 'Feature Importance Analysis');
    
    % 1. Top features for each method
    for i = 1:numMethods
        method = methods{i};
        scores = importanceScores.(method);
        [sortedScores, idx] = sort(scores, 'descend');
        
        subplot(2, 3, i);
        barh(sortedScores(1:min(numImportant, length(sortedScores)));
        set(gca, 'YTick', 1:min(numImportant, length(sortedScores)));
        set(gca, 'YTickLabel', idx(1:min(numImportant, length(sortedScores))));
        title([upper(method) ' Importance']);
        xlabel('Importance Score');
        ylabel('Feature Index');
        grid on;
    end
    
    % 2. Consensus ranking heatmap
    subplot(2, 3, 5);
    
    % Create consensus matrix
    consensusMatrix = zeros(numMethods, numImportant);
    for i = 1:numMethods
        method = methods{i};
        topFeatures = rankedFeatures.(method)(1:min(numImportant, length(rankedFeatures.(method))));
        consensusMatrix(i, 1:length(topFeatures)) = topFeatures;
    end
    
    imagesc(consensusMatrix);
    colorbar;
    title('Consensus Feature Ranking');
    xlabel('Rank Position');
    ylabel('Method');
    set(gca, 'YTick', 1:numMethods);
    set(gca, 'YTickLabel', methods);
    
    % 3. Method comparison
    subplot(2, 3, 6);
    
    % Normalize scores for comparison
    normalizedScores = zeros(numMethods, numImportant);
    for i = 1:numMethods
        method = methods{i};
        scores = importanceScores.(method);
        [sortedScores, ~] = sort(scores, 'descend');
        normalizedScores(i, :) = sortedScores(1:numImportant) ./ max(sortedScores);
    end
    
    plot(1:numImportant, normalizedScores', 'LineWidth', 2);
    legend(methods, 'Location', 'best');
    title('Normalized Importance Scores');
    xlabel('Rank');
    ylabel('Normalized Importance');
    grid on;
    
    % Save figure
    saveas(gcf, fullfile('Results', 'feature_importance_analysis.png'));
end

function consensusRanking = create_consensus_ranking(rankedFeatures, numImportant)
    % Create consensus ranking from multiple methods
    
    methods = fieldnames(rankedFeatures);
    numMethods = length(methods);
    numFeatures = max(cellfun(@(x) length(rankedFeatures.(x)), methods));
    
    % Create ranking matrix
    rankingMatrix = zeros(numFeatures, numMethods);
    
    for i = 1:numMethods
        method = methods{i};
        features = rankedFeatures.(method);
        rankingMatrix(features, i) = 1:length(features);
    end
    
    % Calculate mean rank for each feature
    meanRanks = mean(rankingMatrix, 2, 'omitnan');
    
    % Sort by mean rank
    [~, consensusRanking] = sort(meanRanks);
    
    % Return top features
    consensusRanking = consensusRanking(1:min(numImportant, length(consensusRanking)));
end

function mi = mutualinfo(x, y)
    % Simple mutual information calculation
    [counts, ~, ~] = histcounts2(x, y, 'Normalization', 'probability');
    jointProb = counts + eps; % Avoid log(0)
    marginalX = sum(jointProb, 2);
    marginalY = sum(jointProb, 1);
    
    mi = sum(sum(jointProb .* log2(jointProb ./ (marginalX * marginalY))));
end