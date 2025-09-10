function [prediction, confidence, scores] = predict_emotion(model, new_data, featureNames, xaiResults)
    % PREDICT_EMOTION Predict emotion for new data with XAI support
    % Input: model - trained classification model
    %        new_data - new feature data
    %        featureNames - names of features for validation
    %        xaiResults - (optional) XAI results for insights
    % Output: prediction - predicted emotion (1=Relaxed, 2=Funny/Happy)
    %         confidence - confidence scores
    %         scores - raw prediction scores
    
    % Validate input dimensions
    if size(new_data, 2) ~= length(featureNames)
        error('Input data must have %d features. Got %d.', length(featureNames), size(new_data, 2));
    end
    
    % Normalize new data
    new_data = normalize(new_data);
    
    % Make prediction
    [prediction, scores] = predict(model, new_data);
    
    % Calculate confidence (max probability)
    confidence = max(scores, [], 2);
    
    % Convert numeric predictions to string labels
    emotion_labels = {'Relaxed', 'Funny/Happy'};
    prediction_str = emotion_labels(prediction)';
    
    % Display results
    for i = 1:size(new_data, 1)
        fprintf('Sample %d: Predicted %s (Confidence: %.2f%%)\n', ...
            i, prediction_str{i}, confidence(i) * 100);
        fprintf('  Score - Relaxed: %.3f, Funny/Happy: %.3f\n', ...
            scores(i, 1), scores(i, 2));
    end
    
    % Display XAI insights if provided
    if nargin >= 4 && ~isempty(xaiResults)
        display_xai_insights(xaiResults, new_data, featureNames, prediction, confidence);
    end
    
    % Visualize prediction if single sample
    if size(new_data, 1) == 1
        visualize_prediction(scores, confidence, prediction_str{1});
    end
end

function display_xai_insights(xaiResults, sampleData, featureNames, prediction, confidence)
    % DISPLAY_XAI_INSIGHTS Show XAI explanations for a specific prediction
    
    fprintf('\n=== XAI Insights ===\n');
    
    % Check if we have basic XAI data
    if ~isfield(xaiResults, 'featureImportance') || isempty(xaiResults.featureImportance)
        fprintf('Basic XAI insights not available.\n');
        return;
    end
    
    % Top contributing features
    [~, topIdx] = sort(xaiResults.featureImportance, 'descend');
    fprintf('Top 3 influential features:\n');
    for i = 1:min(3, length(topIdx))
        if topIdx(i) <= length(featureNames) && topIdx(i) <= length(sampleData)
            fprintf('  %s: importance=%.3f, value=%.3f\n', ...
                featureNames{topIdx(i)}, ...
                xaiResults.featureImportance(topIdx(i)), ...
                sampleData(topIdx(i)));
        end
    end
    
    % Confidence assessment
    if isfield(xaiResults, 'confidenceMetrics')
        fprintf('\nConfidence Analysis:\n');
        fprintf('  Your prediction confidence: %.1f%%\n', confidence * 100);
        fprintf('  Model calibration error: %.3f\n', xaiResults.confidenceMetrics.calibrationError);
        
        % Simple confidence interpretation
        if confidence > 0.8
            fprintf('  ✓ High confidence prediction\n');
        elseif confidence > 0.6
            fprintf('  ⚠ Moderate confidence prediction\n');
        else
            fprintf('  ⚠ Low confidence prediction\n');
        end
    end
    
    % Feature contributions for this specific sample
    fprintf('\nKey decision factors for this prediction:\n');
    if isfield(xaiResults, 'pdpData') && ~isempty(xaiResults.pdpData)
        % Simple contribution estimation
        contributions = zeros(1, length(featureNames));
        for f = 1:length(featureNames)
            % Find if this feature has PDP data
            for p = 1:length(xaiResults.pdpData)
                if strcmp(xaiResults.pdpData(p).featureName, featureNames{f})
                    avg_value = mean(xaiResults.pdpData(p).gridPoints);
                    deviation = sampleData(f) - avg_value;
                    contributions(f) = deviation * xaiResults.featureImportance(f);
                    break;
                end
            end
        end
        
        [~, contribOrder] = sort(abs(contributions), 'descend');
        for i = 1:min(3, length(contribOrder))
            featIdx = contribOrder(i);
            if contributions(featIdx) > 0
                direction = 'increased';
            else
                direction = 'decreased';
            end
            fprintf('  %s: %s probability of Funny/Happy\n', ...
                featureNames{featIdx}, direction);
        end
    end
    
    % Decision pattern
    if isfield(xaiResults, 'decisionRules') && ~isempty(xaiResults.decisionRules)
        fprintf('\nRelevant decision pattern:\n');
        fprintf('  %s\n', xaiResults.decisionRules{1});
    end
end

function visualize_prediction(scores, confidence, prediction)
    % VISUALIZE_PREDICTION Visualize prediction results for a single sample
    
    figure('Name', 'Emotion Prediction Results', 'Position', [200, 200, 800, 400]);
    
    % Bar chart of prediction scores
    subplot(1, 2, 1);
    emotions = {'Relaxed', 'Funny/Happy'};
    bar(scores, 'FaceColor', 'flat');
    colormap([0.2 0.6 1; 1 0.6 0.2]); % Blue and orange
    set(gca, 'XTickLabel', emotions);
    ylabel('Prediction Score');
    ylim([0 1]);
    title('Emotion Prediction Scores');
    grid on;
    
    % Confidence gauge
    subplot(1, 2, 2);
    polarplot([0 confidence*2*pi], [0 1], 'LineWidth', 3, 'Color', 'r');
    hold on;
    thetaticks(0:30:330);
    thetaticklabels({'0%', '', '20%', '', '40%', '', '60%', '', '80%', '', '100%'});
    rlim([0 1]);
    title(sprintf('Confidence: %.1f%%\nPredicted: %s', confidence*100, prediction));
    hold off;
end