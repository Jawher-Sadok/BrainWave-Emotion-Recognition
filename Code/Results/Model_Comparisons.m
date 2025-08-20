function [comparisonResults, bestModel] = Model_Comparisons(features, labels, varargin)
    % MODEL_COMPARISONS - Compare multiple machine learning models for EEG classification
    %
    % Inputs:
    %   features - Feature matrix (samples × features)
    %   labels - Target labels (categorical)
    %   Optional parameters:
    %       'Models' - Cell array of models to compare: 
    %                 {'svm', 'rf', 'knn', 'lda', 'nn', 'xgb', 'all'}
    %       'CVFolds' - Number of cross-validation folds (default: 5)
    %       'Hyperopt' - Perform hyperparameter optimization (true/false)
    %       'Metric' - Primary metric for comparison: 
    %                  {'accuracy', 'f1', 'precision', 'recall', 'auc'}
    %       'Visualize' - Show comparison plots (true/false)
    %
    % Outputs:
    %   comparisonResults - Structure with detailed comparison results
    %   bestModel - Best performing model object

    % Parse input parameters
    p = inputParser;
    addRequired(p, 'features', @ismatrix);
    addRequired(p, 'labels', @iscategorical);
    addParameter(p, 'Models', {'all'}, @iscell);
    addParameter(p, 'CVFolds', 5, @isnumeric);
    addParameter(p, 'Hyperopt', false, @islogical);
    addParameter(p, 'Metric', 'accuracy', @ischar);
    addParameter(p, 'Visualize', true, @islogical);
    parse(p, features, labels, varargin{:});
    
    modelsToCompare = p.Results.Models;
    numFolds = p.Results.CVFolds;
    hyperopt = p.Results.Hyperopt;
    primaryMetric = lower(p.Results.Metric);
    visualize = p.Results.Visualize;
    
    % Define all available models
    allModels = {'svm', 'randomforest', 'knn', 'lda', 'neuralnet', 'xgboost'};
    
    % Determine which models to run
    if ismember('all', modelsToCompare)
        modelsToRun = allModels;
    else
        modelsToRun = intersect(allModels, lower(modelsToCompare));
    end
    
    fprintf('Comparing %d models with %d-fold cross-validation...\n', ...
        length(modelsToRun), numFolds);
    fprintf('Primary metric: %s\n', primaryMetric);
    
    % Initialize results structure
    comparisonResults = struct();
    modelPerformances = struct();
    trainingTimes = zeros(length(modelsToRun), 1);
    
    % Create cross-validation partition
    cv = cvpartition(labels, 'KFold', numFolds);
    
    %% Train and evaluate each model
    for modelIdx = 1:length(modelsToRun)
        modelName = modelsToRun{modelIdx};
        fprintf('\n=== Training %s (%d/%d) ===\n', ...
            upper(modelName), modelIdx, length(modelsToRun));
        
        % Initialize metrics storage
        metrics = initialize_metrics_storage(numFolds);
        
        % Cross-validation loop
        for fold = 1:numFolds
            fprintf('Fold %d/%d... ', fold, numFolds);
            
            % Split data
            trainIdx = cv.training(fold);
            testIdx = cv.test(fold);
            
            X_train = features(trainIdx, :);
            y_train = labels(trainIdx);
            X_test = features(testIdx, :);
            y_test = labels(testIdx);
            
            % Train model with timing
            tic;
            model = train_model(modelName, X_train, y_train, hyperopt);
            trainTime = toc;
            
            % Predict and evaluate
            [predictions, scores] = predict_model(model, modelName, X_test);
            
            % Calculate metrics
            foldMetrics = calculate_all_metrics(y_test, predictions, scores);
            
            % Store results
            metrics = store_fold_results(metrics, foldMetrics, fold, trainTime);
            
            fprintf('Accuracy: %.3f\n', foldMetrics.accuracy);
        end
        
        % Store model results
        comparisonResults.(modelName) = metrics;
        modelPerformances.(modelName) = [metrics.accuracy];
        trainingTimes(modelIdx) = mean([metrics.trainTime]);
    end
    
    %% Statistical comparison
    fprintf('\n=== Statistical Comparison ===\n');
    statisticalResults = perform_statistical_comparison(comparisonResults, modelsToRun, primaryMetric);
    
    %% Determine best model
    bestModel = select_best_model(comparisonResults, modelsToRun, primaryMetric);
    
    %% Visualization
    if visualize
        visualize_model_comparisons(comparisonResults, modelsToRun, trainingTimes, statisticalResults);
    end
    
    %% Final summary
    print_comparison_summary(comparisonResults, modelsToRun, bestModel, primaryMetric);
    
    fprintf('\nModel comparison completed successfully!\n');
end

%% Helper Functions

function metrics = initialize_metrics_storage(numFolds)
    % Initialize metrics storage structure
    metricNames = {'accuracy', 'precision', 'recall', 'f1', 'auc', ...
                   'trainTime', 'predictTime', 'confusionMatrix'};
    
    for i = 1:length(metricNames)
        metrics.(metricNames{i}) = zeros(numFolds, 1);
    end
    metrics.confusionMatrix = {};
end

function model = train_model(modelName, X_train, y_train, hyperopt)
    % Train specific model type
    rng(42); % For reproducibility
    
    switch lower(modelName)
        case 'svm'
            if hyperopt
                model = fitcecoc(X_train, y_train, 'OptimizeHyperparameters', 'auto', ...
                    'HyperparameterOptimizationOptions', struct('Verbose', 0));
            else
                template = templateSVM('KernelFunction', 'rbf', 'Standardize', true);
                model = fitcecoc(X_train, y_train, 'Learners', template);
            end
            
        case 'randomforest'
            if hyperopt
                model = fitcensemble(X_train, y_train, 'Method', 'Bag', ...
                    'OptimizeHyperparameters', {'NumLearningCycles'});
            else
                model = fitcensemble(X_train, y_train, 'Method', 'Bag', ...
                    'NumLearningCycles', 100);
            end
            
        case 'knn'
            if hyperopt
                model = fitcknn(X_train, y_train, 'OptimizeHyperparameters', 'auto');
            else
                model = fitcknn(X_train, y_train, 'NumNeighbors', 5);
            end
            
        case 'lda'
            model = fitcdiscr(X_train, y_train, 'DiscrimType', 'pseudolinear');
            
        case 'neuralnet'
            % Simple neural network
            net = patternnet([50 25]);
            net.trainParam.showWindow = false;
            net.divideParam.trainRatio = 0.8;
            net.divideParam.valRatio = 0.2;
            net.divideParam.testRatio = 0;
            
            % Convert labels to dummy variables
            y_dummy = dummyvar(double(y_train));
            
            model = train(net, X_train', y_dummy');
            
        case 'xgboost'
            % Requires XGBoost installation
            try
                model = fitcensemble(X_train, y_train, 'Method', 'GentleBoost', ...
                    'NumLearningCycles', 100);
            catch
                warning('XGBoost not available, using AdaBoost instead');
                model = fitcensemble(X_train, y_train, 'Method', 'AdaBoostM1');
            end
    end
end

function [predictions, scores] = predict_model(model, modelName, X_test)
    % Make predictions based on model type
    switch lower(modelName)
        case 'neuralnet'
            % Neural network prediction
            y_pred = model(X_test');
            [~, predictions] = max(y_pred);
            predictions = categorical(predictions, [1 2 3], {'NEGATIVE', 'NEUTRAL', 'POSITIVE'});
            scores = y_pred';
            
        otherwise
            % Standard classifier prediction
            [predictions, scores] = predict(model, X_test);
    end
end

function metrics = calculate_all_metrics(trueLabels, predLabels, scores)
    % Calculate comprehensive performance metrics
    metrics.accuracy = sum(trueLabels == predLabels) / numel(trueLabels);
    
    % Per-class metrics
    cm = confusionmat(trueLabels, predLabels);
    metrics.confusionMatrix = cm;
    
    precision = diag(cm) ./ sum(cm, 1)';
    recall = diag(cm) ./ sum(cm, 2);
    f1 = 2 * (precision .* recall) ./ (precision + recall);
    
    metrics.precision = mean(precision, 'omitnan');
    metrics.recall = mean(recall, 'omitnan');
    metrics.f1 = mean(f1, 'omitnan');
    
    % AUC (if scores available)
    if ~isempty(scores)
        try
            [~,~,~,auc] = perfcurve(double(trueLabels), scores(:,2), 2);
            metrics.auc = auc;
        catch
            metrics.auc = NaN;
        end
    else
        metrics.auc = NaN;
    end
end

function metrics = store_fold_results(metrics, foldMetrics, fold, trainTime)
    % Store results from a single fold
    metrics.accuracy(fold) = foldMetrics.accuracy;
    metrics.precision(fold) = foldMetrics.precision;
    metrics.recall(fold) = foldMetrics.recall;
    metrics.f1(fold) = foldMetrics.f1;
    metrics.auc(fold) = foldMetrics.auc;
    metrics.trainTime(fold) = trainTime;
    metrics.confusionMatrix{fold} = foldMetrics.confusionMatrix;
end

function statisticalResults = perform_statistical_comparison(results, models, primaryMetric)
    % Perform statistical comparison of models
    numModels = length(models);
    statisticalResults = struct();
    
    % Extract performance metrics
    performances = zeros(numModels, 100); % Assuming 100 samples or folds
    
    for i = 1:numModels
        modelName = models{i};
        metricData = results.(modelName).(primaryMetric);
        performances(i, 1:length(metricData)) = metricData;
    end
    
    % Friedman test (non-parametric ANOVA)
    [p, tbl, stats] = friedman(performances(:, 1:size(performances, 2)), 1, 'off');
    statisticalResults.friedmanP = p;
    statisticalResults.friedmanStats = stats;
    
    % Post-hoc Nemenyi test
    if p < 0.05
        statisticalResults.significant = true;
        fprintf('Significant differences found (p = %.4f)\n', p);
    else
        statisticalResults.significant = false;
        fprintf('No significant differences found (p = %.4f)\n', p);
    end
end

function bestModelInfo = select_best_model(results, models, primaryMetric)
    % Select the best performing model
    meanPerformance = zeros(length(models), 1);
    
    for i = 1:length(models)
        modelName = models{i};
        meanPerformance(i) = mean(results.(modelName).(primaryMetric));
    end
    
    [bestScore, bestIdx] = max(meanPerformance);
    bestModelName = models{bestIdx};
    
    bestModelInfo = struct(...
        'name', bestModelName, ...
        'score', bestScore, ...
        'index', bestIdx, ...
        'allScores', meanPerformance);
    
    fprintf('Best model: %s (%.4f %s)\n', ...
        upper(bestModelName), bestScore, primaryMetric);
end

function visualize_model_comparisons(results, models, trainingTimes, statisticalResults)
    % Create comprehensive visualization of model comparisons
    
    numModels = length(models);
    metrics = {'accuracy', 'precision', 'recall', 'f1', 'auc'};
    numMetrics = length(metrics);
    
    % Create figure
    figure('Position', [100, 100, 1400, 900], 'Name', 'Model Comparison Results');
    
    % 1. Performance comparison boxplot
    subplot(2, 3, 1);
    performanceData = [];
    modelNames = {};
    
    for i = 1:numModels
        modelName = models{i};
        performanceData = [performanceData; results.(modelName).accuracy];
        modelNames = [modelNames; repmat({upper(modelName)}, length(results.(modelName).accuracy), 1)];
    end
    
    boxplot(performanceData, modelNames);
    title('Model Accuracy Comparison');
    ylabel('Accuracy');
    grid on;
    rotateXLabels(gca, 45);
    
    % 2. Training time comparison
    subplot(2, 3, 2);
    bar(trainingTimes);
    title('Training Time Comparison');
    ylabel('Time (seconds)');
    set(gca, 'XTick', 1:numModels);
    set(gca, 'XTickLabel', upper(models));
    grid on;
    rotateXLabels(gca, 45);
    
    % 3. Multiple metrics radar chart
    subplot(2, 3, 3);
    
    radarData = zeros(numMetrics, numModels);
    for i = 1:numModels
        modelName = models{i};
        for j = 1:numMetrics
            radarData(j, i) = mean(results.(modelName).(metrics{j}));
        end
    end
    
    % Normalize for radar chart
    radarDataNorm = radarData ./ max(radarData, [], 2);
    spider_plot(radarDataNorm, 'AxesLabels', metrics, 'AxesLabelsEdge', 'none');
    legend(upper(models), 'Location', 'best');
    title('Normalized Performance Metrics');
    
    % 4. Confidence intervals
    subplot(2, 3, 4);
    
    means = zeros(numModels, 1);
    stds = zeros(numModels, 1);
    
    for i = 1:numModels
        modelName = models{i};
        means(i) = mean(results.(modelName).accuracy);
        stds(i) = std(results.(modelName).accuracy);
    end
    
    errorbar(1:numModels, means, stds, 'o', 'LineWidth', 2);
    title('Accuracy with Standard Deviation');
    ylabel('Accuracy');
    set(gca, 'XTick', 1:numModels);
    set(gca, 'XTickLabel', upper(models));
    grid on;
    rotateXLabels(gca, 45);
    
    % 5. Statistical significance
    subplot(2, 3, 5);
    
    if statisticalResults.significant
        % Create critical difference diagram
        plot(1:numModels, means, 'o-', 'LineWidth', 2);
        title('Statistical Significance (Friedman Test)');
        xlabel('Model Rank');
        ylabel('Mean Accuracy');
        grid on;
    else
        text(0.5, 0.5, 'No significant differences found', ...
            'HorizontalAlignment', 'center', 'FontSize', 12);
        title('Statistical Comparison');
    end
    
    % 6. Confusion matrix for best model
    subplot(2, 3, 6);
    
    % Find best model
    [~, bestIdx] = max(means);
    bestModel = models{bestIdx};
    
    % Average confusion matrix
    avgCM = mean(cat(3, results.(bestModel).confusionMatrix{:}), 3);
    confusionchart(avgCM, categories(unique(results.(bestModel).accuracy)));
    title(['Confusion Matrix - ' upper(bestModel)]);
    
    % Save figure
    saveas(gcf, fullfile('Results', 'model_comparison_results.png'));
end

function print_comparison_summary(results, models, bestModel, primaryMetric)
    % Print detailed comparison summary
    fprintf('\n=== MODEL COMPARISON SUMMARY ===\n');
    fprintf('%-15s %-10s %-10s %-10s %-10s %-10s\n', ...
        'Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC');
    fprintf('%s\n', repmat('-', 65, 1));
    
    for i = 1:length(models)
        modelName = models{i};
        fprintf('%-15s ', upper(modelName));
        fprintf('%-10.4f ', mean(results.(modelName).accuracy));
        fprintf('%-10.4f ', mean(results.(modelName).precision));
        fprintf('%-10.4f ', mean(results.(modelName).recall));
        fprintf('%-10.4f ', mean(results.(modelName).f1));
        fprintf('%-10.4f\n', mean(results.(modelName).auc));
    end
    
    fprintf('\nBest model: %s (%.4f %s)\n', ...
        upper(bestModel.name), bestModel.score, primaryMetric);
end

function spider_plot(P, varargin)
    % Simple spider plot implementation
    % This is a simplified version - consider using a proper spider plot function
    theta = linspace(0, 2*pi, size(P, 1) + 1);
    theta = theta(1:end-1);
    
    polarplot([theta, theta(1)], [P; P(1,:)], 'LineWidth', 2);
    thetaticklabels(gca, varargin{find(strcmp(varargin, 'AxesLabels'))+1});
end