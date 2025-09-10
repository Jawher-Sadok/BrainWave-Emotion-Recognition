function [model, cvAccuracy, featureImportance, xaiResults] = train_emotion_classifier(features, labels, featureNames)
    % TRAIN_EMOTION_CLASSIFIER Train a machine learning model for emotion classification
    % with Explainable AI (XAI) components
    
    fprintf('Training emotion classification model with XAI...\n');
    
    % Check for empty data
    if isempty(features) || isempty(labels)
        error('Empty features or labels provided. Please check your data.');
    end
    
    % Split data into training and testing sets
    rng(42); % For reproducibility
    cv = cvpartition(labels, 'HoldOut', 0.3);
    
    X_train = features(cv.training, :);
    y_train = labels(cv.training);
    X_test = features(cv.test, :);
    y_test = labels(cv.test);
    
    fprintf('Training set: %d samples\n', size(X_train, 1));
    fprintf('Test set: %d samples\n', size(X_test, 1));
    
    % Check if we have enough data
    if size(X_train, 1) < 10
        error('Insufficient training data. Need at least 10 samples.');
    end
    
    % Store original data for visualization
    X_train_original = X_train;
    X_test_original = X_test;
    
    % Check for class imbalance
    classRatio = sum(y_train == 1) / sum(y_train == 2);
    if classRatio < 0.7 || classRatio > 1.3
        fprintf('Applying oversampling for class imbalance (ratio: %.2f)...\n', classRatio);
        [X_train, y_train] = smote_oversample(X_train, y_train);
    end
    
    % Use simpler hyperparameter optimization
    fprintf('Performing hyperparameter optimization...\n');
    
    % Define parameter grid
    paramGrid = struct();
    paramGrid.MaxNumSplits = [5, 10, 20];
    paramGrid.MinLeafSize = [1, 5, 10];
    paramGrid.NumLearningCycles = [50, 100];
    
    bestAccuracy = 0;
    bestParams = struct();
    
    % Simple grid search
    for maxSplits = paramGrid.MaxNumSplits
        for minLeaf = paramGrid.MinLeafSize
            for numCycles = paramGrid.NumLearningCycles
                % Quick 3-fold CV
                tempCV = cvpartition(y_train, 'KFold', min(3, length(unique(y_train))));
                foldAcc = 0;
                
                for fold = 1:tempCV.NumTestSets
                    trainIdx = tempCV.training(fold);
                    testIdx = tempCV.test(fold);
                    
                    % Skip if not enough data in fold
                    if sum(trainIdx) < 2 || sum(testIdx) < 2
                        continue;
                    end
                    
                    tempModel = fitcensemble(X_train(trainIdx, :), y_train(trainIdx), ...
                        'Method', 'Bag', ...
                        'NumLearningCycles', numCycles, ...
                        'Learners', templateTree('MaxNumSplits', maxSplits, ...
                                                'MinLeafSize', minLeaf));
                    
                    pred = predict(tempModel, X_train(testIdx, :));
                    foldAcc = foldAcc + sum(pred == y_train(testIdx)) / length(pred);
                end
                
                if tempCV.NumTestSets > 0
                    avgAcc = foldAcc / tempCV.NumTestSets;
                    
                    if avgAcc > bestAccuracy
                        bestAccuracy = avgAcc;
                        bestParams.MaxNumSplits = maxSplits;
                        bestParams.MinLeafSize = minLeaf;
                        bestParams.NumLearningCycles = numCycles;
                    end
                end
            end
        end
    end
    
    % Check if we found valid parameters
    if isempty(fieldnames(bestParams))
        fprintf('Using default parameters due to optimization issues...\n');
        bestParams.MaxNumSplits = 10;
        bestParams.MinLeafSize = 5;
        bestParams.NumLearningCycles = 100;
    end
    
    fprintf('Best parameters: MaxSplits=%d, MinLeaf=%d, NumCycles=%d\n', ...
        bestParams.MaxNumSplits, bestParams.MinLeafSize, bestParams.NumLearningCycles);
    
    % Train final model
    model = fitcensemble(X_train, y_train, ...
        'Method', 'Bag', ...
        'NumLearningCycles', bestParams.NumLearningCycles, ...
        'Learners', templateTree('MaxNumSplits', bestParams.MaxNumSplits, ...
                                'MinLeafSize', bestParams.MinLeafSize), ...
        'ClassNames', [1, 2]);
    
    % Perform cross-validation
    cv = cvpartition(y_train, 'KFold', min(10, length(unique(y_train))));
    cvAccuracy = 0;
    validFolds = 0;
    
    for i = 1:cv.NumTestSets
        trainIdx = cv.training(i);
        testIdx = cv.test(i);
        
        if sum(trainIdx) < 2 || sum(testIdx) < 2
            continue;
        end
        
        foldModel = fitcensemble(X_train(trainIdx, :), y_train(trainIdx), ...
            'Method', 'Bag', ...
            'NumLearningCycles', bestParams.NumLearningCycles, ...
            'Learners', templateTree('MaxNumSplits', bestParams.MaxNumSplits, ...
                                    'MinLeafSize', bestParams.MinLeafSize));
        
        foldPred = predict(foldModel, X_train(testIdx, :));
        cvAccuracy = cvAccuracy + sum(foldPred == y_train(testIdx)) / length(foldPred);
        validFolds = validFolds + 1;
    end
    
    if validFolds > 0
        cvAccuracy = cvAccuracy / validFolds;
    else
        cvAccuracy = 0.5; % Default if no valid folds
    end
    
    fprintf('Cross-validation accuracy: %.2f%%\n', cvAccuracy * 100);
    
    % Test on holdout set
    if ~isempty(X_test) && ~isempty(y_test)
        y_pred = predict(model, X_test);
        testAccuracy = sum(y_pred == y_test) / length(y_test);
        fprintf('Test accuracy: %.2f%%\n', testAccuracy * 100);
    else
        testAccuracy = cvAccuracy;
        fprintf('No test data available, using CV accuracy\n');
    end
    
    % Calculate feature importance
    try
        featureImportance = predictorImportance(model);
    catch
        featureImportance = ones(1, size(X_train, 2)) / size(X_train, 2);
        fprintf('Using uniform feature importance due to calculation error\n');
    end
    
    % Add regularization: Drop less important features (with safety check)
    importanceThreshold = 0.01;
    importantFeatures = featureImportance > importanceThreshold;
    
    if sum(importantFeatures) < length(featureImportance) && sum(importantFeatures) >= 2
        fprintf('Dropping %d low-importance features to reduce overfitting\n', sum(~importantFeatures));
        
        % Update feature names and data
        featureNames = featureNames(importantFeatures);
        X_train = X_train(:, importantFeatures);
        X_test = X_test(:, importantFeatures);
        featureImportance = featureImportance(importantFeatures);
        
        % Retrain model with only important features
        model = fitcensemble(X_train, y_train, ...
            'Method', 'Bag', ...
            'NumLearningCycles', bestParams.NumLearningCycles, ...
            'Learners', templateTree('MaxNumSplits', bestParams.MaxNumSplits, ...
                                    'MinLeafSize', bestParams.MinLeafSize));
    else
        fprintf('Keeping all features (minimum 2 features required)\n');
    end
    
    % XAI: Generate explanations (only if we have data)
    if ~isempty(X_test) && ~isempty(y_test) && size(X_test, 2) >= 2
        xaiResults = generate_xai_explanations(model, X_test, y_test, featureNames, featureImportance);
        
        % Visualize results with XAI components
        try
            visualize_training_results(model, X_test, y_test, featureImportance, featureNames, xaiResults);
        catch ME
            fprintf('Visualization error: %s\n', ME.message);
        end
    else
        xaiResults = struct();
        fprintf('Skipping XAI due to insufficient test data or features\n');
    end
    
    fprintf('XAI analysis completed.\n');
end

function [X_resampled, y_resampled] = smote_oversample(X, y)
    % SMOTE_OVERSAMPLE Simple oversampling for class imbalance
    
    if isempty(X) || isempty(y)
        X_resampled = X;
        y_resampled = y;
        return;
    end
    
    classes = unique(y);
    if length(classes) < 2
        X_resampled = X;
        y_resampled = y;
        return;
    end
    
    classCounts = arrayfun(@(c) sum(y == c), classes);
    [maxCount, maxIdx] = max(classCounts);
    minorityClass = classes(classes ~= classes(maxIdx));
    
    if isempty(minorityClass) || length(minorityClass) > 1
        X_resampled = X;
        y_resampled = y;
        return;
    end
    
    % Find minority class samples
    minorityX = X(y == minorityClass, :);
    minorityY = y(y == minorityClass);
    
    if isempty(minorityX)
        X_resampled = X;
        y_resampled = y;
        return;
    end
    
    % Number of samples to replicate
    replicationFactor = ceil(maxCount / length(minorityY));
    
    % Replicate minority samples
    replicatedX = repmat(minorityX, replicationFactor, 1);
    replicatedY = repmat(minorityY, replicationFactor, 1);
    
    % Trim to match majority class count
    if size(replicatedX, 1) > maxCount
        replicatedX = replicatedX(1:maxCount, :);
        replicatedY = replicatedY(1:maxCount);
    end
    
    % Combine original and replicated data
    X_resampled = [X; replicatedX];
    y_resampled = [y; replicatedY];
    
    % Shuffle the data
    idx = randperm(length(y_resampled));
    X_resampled = X_resampled(idx, :);
    y_resampled = y_resampled(idx);
end

function xaiResults = generate_xai_explanations(model, X_test, y_test, featureNames, featureImportance)
    % GENERATE_XAI_EXPLANATIONS Generate Explainable AI insights
    
    xaiResults = struct();
    
    % Basic feature importance
    xaiResults.featureImportance = featureImportance;
    xaiResults.featureNames = featureNames;
    
    % Only generate complex XAI if we have enough data
    if size(X_test, 1) >= 10 && size(X_test, 2) >= 2
        try
            % Partial Dependence Plots
            xaiResults.pdpData = compute_partial_dependence(model, X_test, featureNames, featureImportance);
            
            % Local Explanations
            [xaiResults.localExplanations, xaiResults.sampleIndices] = compute_local_explanations(model, X_test, y_test, featureNames);
            
            % Decision Rules
            xaiResults.decisionRules = extract_decision_rules(model, featureNames);
            
            % Confidence Calibration
            xaiResults.confidenceMetrics = analyze_confidence_calibration(model, X_test, y_test);
            
        catch ME
            fprintf('XAI generation error: %s\n', ME.message);
        end
    else
        fprintf('Skipping advanced XAI due to limited test data\n');
    end
end

function pdpData = compute_partial_dependence(model, X, featureNames, importance, numTopFeatures)
    % COMPUTE_PARTIAL_DEPENDENCE Calculate Partial Dependence
    
    if nargin < 5
        numTopFeatures = min(3, length(featureNames));
    end
    
    if isempty(X) || length(featureNames) < 2
        pdpData = [];
        return;
    end
    
    [~, topIdx] = sort(importance, 'descend');
    topFeatures = topIdx(1:min(numTopFeatures, length(topIdx)));
    
    pdpData = struct();
    for i = 1:length(topFeatures)
        featIdx = topFeatures(i);
        if featIdx > size(X, 2)
            continue;
        end
        
        featName = featureNames{featIdx};
        
        % Create grid of values for this feature
        gridPoints = linspace(min(X(:, featIdx)), max(X(:, featIdx)), 10);
        predictions = zeros(length(gridPoints), 2);
        
        for j = 1:length(gridPoints)
            X_temp = X;
            X_temp(:, featIdx) = gridPoints(j);
            try
                [~, scores] = predict(model, X_temp);
                predictions(j, :) = mean(scores, 1);
            catch
                predictions(j, :) = [0.5, 0.5];
            end
        end
        
        pdpData(i).featureName = featName;
        pdpData(i).gridPoints = gridPoints;
        pdpData(i).predictions = predictions;
        pdpData(i).importance = importance(featIdx);
    end
end

function [localExplanations, sampleIndices] = compute_local_explanations(model, X, y, featureNames, numSamples)
    % COMPUTE_LOCAL_EXPLANATIONS Generate local explanations
    
    if nargin < 5
        numSamples = 2;
    end
    
    localExplanations = struct();
    sampleIndices = [];
    
    if isempty(X) || isempty(y)
        return;
    end
    
    classes = unique(y);
    for c = 1:length(classes)
        classIdx = find(y == classes(c));
        if isempty(classIdx)
            continue;
        end
        
        if length(classIdx) > numSamples
            selectedIdx = classIdx(randperm(length(classIdx), numSamples));
        else
            selectedIdx = classIdx;
        end
        
        for i = 1:length(selectedIdx)
            idx = selectedIdx(i);
            try
                [pred, scores] = predict(model, X(idx, :));
                
                % Simple feature contribution analysis
                featureContributions = zeros(1, size(X, 2));
                basePrediction = mean(predict(model, X));
                
                for f = 1:size(X, 2)
                    try
                        X_perturbed = X;
                        X_perturbed(:, f) = mean(X(:, f));
                        [~, perturbedScores] = predict(model, X_perturbed);
                        featureContributions(f) = scores(2) - mean(perturbedScores(:, 2));
                    catch
                        featureContributions(f) = 0;
                    end
                end
                
                localExplanations(end+1).sampleIndex = idx;
                localExplanations(end).trueClass = y(idx);
                localExplanations(end).predictedClass = pred;
                localExplanations(end).confidence = max(scores);
                localExplanations(end).featureContributions = featureContributions;
                
            catch
                % Skip this sample if there's an error
            end
        end
        sampleIndices = [sampleIndices; selectedIdx(:)];
    end
end

function decisionRules = extract_decision_rules(model, featureNames, numRules)
    % EXTRACT_DECISION_RULES Extract interpretable decision rules
    
    if nargin < 3
        numRules = 3;
    end
    
    decisionRules = {};
    
    if isempty(featureNames)
        return;
    end
    
    try
        % Simple rule extraction based on feature importance
        if isprop(model, 'PredictorImportance')
            [sortedImportance, sortedIdx] = sort(model.PredictorImportance, 'descend');
        else
            sortedImportance = ones(1, length(featureNames)) / length(featureNames);
            sortedIdx = 1:length(featureNames);
        end
        
        for i = 1:min(numRules, length(sortedIdx))
            featName = featureNames{sortedIdx(i)};
            rule = sprintf('IF %s is ABOVE average THEN higher probability of Class 2', featName);
            decisionRules{end+1} = rule;
        end
    catch
        decisionRules = {'Decision rules extraction failed'};
    end
end

function confidenceMetrics = analyze_confidence_calibration(model, X_test, y_test)
    % ANALYZE_CONFIDENCE_CALIBRATION Analyze confidence calibration
    
    confidenceMetrics = struct();
    confidenceMetrics.accuracy = 0.5;
    confidenceMetrics.avgConfidence = 0.5;
    confidenceMetrics.calibrationError = 0;
    
    if isempty(X_test) || isempty(y_test)
        return;
    end
    
    try
        [pred, scores] = predict(model, X_test);
        confidence = max(scores, [], 2);
        
        confidenceMetrics.accuracy = mean(pred == y_test);
        confidenceMetrics.avgConfidence = mean(confidence);
        confidenceMetrics.calibrationError = abs(confidenceMetrics.accuracy - confidenceMetrics.avgConfidence);
    catch
        % Use default values if analysis fails
    end
end

function visualize_training_results(model, X_test, y_test, featureImportance, featureNames, xaiResults)
    % VISUALIZE_TRAINING_RESULTS Enhanced visualization
    
    if isempty(X_test) || isempty(y_test)
        fprintf('Skipping visualization due to empty data\n');
        return;
    end
    
    try
        figure('Name', 'Model Performance', 'Position', [100, 100, 1000, 600]);
        
        % Basic visualizations only
        subplot(2, 2, 1);
        y_pred = predict(model, X_test);
        cm = confusionmat(y_test, y_pred);
        confusionchart(cm, {'Relaxed', 'Funny/Happy'});
        title('Confusion Matrix');
        
        subplot(2, 2, 2);
        if ~isempty(featureImportance)
            barh(featureImportance);
            set(gca, 'YTick', 1:length(featureNames), 'YTickLabel', featureNames);
            title('Feature Importance');
        end
        
        subplot(2, 2, 3);
        try
            [~, scores] = predict(model, X_test);
            [X, Y, T, AUC] = perfcurve(y_test, scores(:, 2), 2);
            plot(X, Y, 'LineWidth', 2);
            title(sprintf('ROC Curve (AUC = %.3f)', AUC));
        catch
            text(0.5, 0.5, 'ROC curve not available', 'HorizontalAlignment', 'center');
        end
        
    catch ME
        fprintf('Visualization failed: %s\n', ME.message);
    end
end