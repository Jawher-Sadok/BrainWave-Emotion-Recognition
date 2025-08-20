function [results, models] = cross_validation(data, target, modelType, k)
    % Perform k-fold cross validation
    % Inputs:
    %   data - feature matrix
    %   target - labels
    %   modelType - 'svm', 'rf', 'cnn', etc.
    %   k - number of folds
    
    cv = cvpartition(target, 'KFold', k);
    results = struct();
    models = cell(k, 1);
    
    % Define evaluation metrics
    metrics = {'Accuracy', 'Precision', 'Recall', 'F1', 'AUC'};
    
    for i = 1:k
        fprintf('Processing fold %d/%d...\n', i, k);
        
        % Split data
        trainIdx = cv.training(i);
        testIdx = cv.test(i);
        
        % Train model
        switch lower(modelType)
            case 'svm'
                model = fitcecoc(data(trainIdx,:), target(trainIdx), ...
                    'Learners', templateSVM('KernelFunction', 'rbf'));
            case 'rf'
                model = fitcensemble(data(trainIdx,:), target(trainIdx), ...
                    'Method', 'Bag', 'NumLearningCycles', 100);
            case 'cnn'
                % CNN implementation would go here
                error('CNN not implemented in this example');
            otherwise
                error('Unknown model type');
        end
        
        % Predict
        pred = predict(model, data(testIdx,:));
        
        % Store model
        models{i} = model;
        
        % Calculate metrics
        results(i).Accuracy = sum(pred == target(testIdx)) / numel(testIdx);
        [results(i).Precision, results(i).Recall, results(i).F1] = ...
            calculate_metrics(pred, target(testIdx));
        
        % For multi-class AUC, we need to calculate per-class
        if ismember('AUC', metrics)
            [~, scores] = predict(model, data(testIdx,:));
            results(i).AUC = multiclass_auc(scores, target(testIdx));
        end
    end
    
    % Aggregate results
    avgResults = struct();
    for m = 1:length(metrics)
        metric = metrics{m};
        avgResults.(metric) = mean([results.(metric)]);
    end
    
    disp('Cross-validation completed.');
    disp('Average results:');
    disp(avgResults);
end

function [precision, recall, f1] = calculate_metrics(pred, target)
    % Calculate precision, recall, and F1 score
    C = confusionmat(target, pred);
    
    precision = diag(C) ./ sum(C, 1)';
    recall = diag(C) ./ sum(C, 2);
    f1 = 2 * (precision .* recall) ./ (precision + recall);
end

function auc = multiclass_auc(scores, target)
    % Calculate multiclass AUC
    [~,~,~,auc] = perfcurve(target, scores, 'ClassNames', unique(target));
end