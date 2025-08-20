function [metrics, cm] = performance_metrics(trueLabels, predLabels, predScores)
    % Calculate comprehensive performance metrics
    % Inputs:
    %   trueLabels - ground truth labels
    %   predLabels - predicted labels
    %   predScores - predicted scores/probabilities
    
    % Confusion matrix
    cm = confusionmat(trueLabels, predLabels);
    
    % Basic metrics
    accuracy = sum(diag(cm)) / sum(cm(:));
    
    % Per-class metrics
    numClasses = size(cm, 1);
    precision = zeros(numClasses, 1);
    recall = zeros(numClasses, 1);
    f1 = zeros(numClasses, 1);
    
    for i = 1:numClasses
        tp = cm(i,i);
        fp = sum(cm(:,i)) - tp;
        fn = sum(cm(i,:)) - tp;
        
        precision(i) = tp / (tp + fp);
        recall(i) = tp / (tp + fn);
        f1(i) = 2 * (precision(i) * recall(i)) / (precision(i) + recall(i));
    end
    
    % Multiclass AUC if scores are provided
    if nargin > 2 && ~isempty(predScores)
        [~,~,~,auc] = perfcurve(trueLabels, predScores, 'ClassNames', unique(trueLabels));
    else
        auc = NaN;
    end
    
    % Cohen's Kappa
    observedAccuracy = accuracy;
    expectedAccuracy = sum(sum(cm, 1) .* sum(cm, 2)') / sum(cm(:))^2;
    kappa = (observedAccuracy - expectedAccuracy) / (1 - expectedAccuracy);
    
    % Store all metrics
    metrics = struct(...
        'Accuracy', accuracy, ...
        'Precision', precision, ...
        'Recall', recall, ...
        'F1', f1, ...
        'AUC', auc, ...
        'Kappa', kappa, ...
        'ConfusionMatrix', cm);
    
    % Visualization
    figure;
    confusionchart(cm, categories(trueLabels));
    title('Confusion Matrix');
    
    % ROC Curve if scores available
    if nargin > 2 && ~isempty(predScores)
        figure;
        [X,Y,T,AUC] = perfcurve(trueLabels, predScores, 'ClassNames', unique(trueLabels));
        plot(X,Y);
        xlabel('False positive rate'); 
        ylabel('True positive rate');
        title(['ROC Curve (AUC = ' num2str(AUC) ')']);
        legend(categories(trueLabels));
    end
end