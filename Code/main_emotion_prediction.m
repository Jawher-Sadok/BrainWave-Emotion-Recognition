function main_emotion_predictions()
    % MAIN_EMOTION_PREDICTION Main program for emotion prediction with XAI
    % Enhanced with Explainable AI (XAI) components
    
    fprintf('=== Emotion Prediction System with XAI ===\n');
    fprintf('Predicting: Relaxed vs Funny/Happy\n\n');
    
    % File path
    filename = 'emotion_data.csv';
    
    try
        % 1. Load and preprocess data
        [features, labels, featureNames] = load_and_preprocess_data(filename);
        
        % 2. Analyze features
        analyze_features(features, labels, featureNames);
        
        % 3. Train classifier with XAI
        fprintf('\n=== Training with Explainable AI ===\n');
        [model, cvAccuracy, featureImportance, xaiResults] = train_emotion_classifier(features, labels, featureNames);
        
        % 4. Demonstrate prediction on some training samples with XAI insights
        fprintf('\n=== Demonstration Predictions with XAI ===\n');
        
        % Select a few samples from each class
        relaxed_samples = features(labels == 1, :);
        funny_samples = features(labels == 2, :);
        
        % Predict on samples from each class with XAI
        if ~isempty(relaxed_samples)
            sample_idx = randi(size(relaxed_samples, 1));
            fprintf('\nPredicting Relaxed sample:\n');
            predict_emotion(model, relaxed_samples(sample_idx, :), featureNames, xaiResults);
        end
        
        if ~isempty(funny_samples)
            sample_idx = randi(size(funny_samples, 1));
            fprintf('\nPredicting Funny/Happy sample:\n');
            predict_emotion(model, funny_samples(sample_idx, :), featureNames, xaiResults);
        end
        
        % 5. Show XAI dashboard
        fprintf('\n=== XAI Dashboard ===\n');
        display_xai_dashboard(xaiResults, featureNames);
        
        % 6. Display summary
        fprintf('\n=== Summary ===\n');
        fprintf('Model trained successfully with XAI!\n');
        fprintf('Cross-validation accuracy: %.2f%%\n', cvAccuracy * 100);
        
        if isfield(xaiResults, 'confidenceMetrics')
            fprintf('Test accuracy: %.2f%%\n', xaiResults.confidenceMetrics.accuracy * 100);
        end
        
        fprintf('Most important feature: %s (score: %.3f)\n', ...
            featureNames{find(featureImportance == max(featureImportance), 1)}, ...
            max(featureImportance));
        
        if isfield(xaiResults, 'confidenceMetrics')
            fprintf('Model calibration error: %.3f\n', xaiResults.confidenceMetrics.calibrationError);
        end
        
        % 7. Interactive XAI exploration
        fprintf('\n=== Interactive XAI Exploration ===\n');
        interactive_xai_exploration(model, features, labels, featureNames, xaiResults);
        
    catch ME
        fprintf('Error: %s\n', ME.message);
        fprintf('Please ensure the CSV file exists and has the correct format.\n');
        rethrow(ME);
    end
end

function display_xai_dashboard(xaiResults, featureNames)
    % DISPLAY_XAI_DASHBOARD Display comprehensive XAI insights
    
    fprintf('\nXAI Dashboard - Model Transparency Report:\n');
    fprintf('==========================================\n\n');
    
    % Check if we have basic XAI data
    if ~isfield(xaiResults, 'featureImportance') || isempty(xaiResults.featureImportance)
        fprintf('XAI insights not available.\n');
        return;
    end
    
    % Feature Importance Summary
    fprintf('TOP 5 MOST IMPORTANT FEATURES:\n');
    [sortedImp, sortedIdx] = sort(xaiResults.featureImportance, 'descend');
    for i = 1:min(5, length(sortedIdx))
        if sortedIdx(i) <= length(featureNames)
            fprintf('%d. %s: %.3f\n', i, featureNames{sortedIdx(i)}, sortedImp(i));
        end
    end
    
    % Confidence Calibration
    if isfield(xaiResults, 'confidenceMetrics')
        fprintf('\nCONFIDENCE CALIBRATION:\n');
        fprintf('Average confidence: %.1f%%\n', xaiResults.confidenceMetrics.avgConfidence * 100);
        fprintf('Actual accuracy: %.1f%%\n', xaiResults.confidenceMetrics.accuracy * 100);
        fprintf('Calibration error: %.3f (lower is better)\n', xaiResults.confidenceMetrics.calibrationError);
        
        if xaiResults.confidenceMetrics.calibrationError < 0.05
            fprintf('✓ Model is well-calibrated\n');
        elseif xaiResults.confidenceMetrics.calibrationError < 0.1
            fprintf('⚠ Model is moderately calibrated\n');
        else
            fprintf('✗ Model calibration needs improvement\n');
        end
    end
    
    % Decision Patterns
    if isfield(xaiResults, 'decisionRules') && ~isempty(xaiResults.decisionRules)
        fprintf('\nKEY DECISION PATTERNS:\n');
        for i = 1:min(3, length(xaiResults.decisionRules))
            fprintf('• %s\n', xaiResults.decisionRules{i});
        end
    end
    
    % Model Trustworthiness
    fprintf('\nMODEL TRUSTWORTHINESS ASSESSMENT:\n');
    if isfield(xaiResults, 'confidenceMetrics') && xaiResults.confidenceMetrics.accuracy > 0.8
        fprintf('✓ High accuracy model\n');
    else
        fprintf('⚠ Moderate accuracy model\n');
    end
    
    if max(xaiResults.featureImportance) > 0.1
        fprintf('✓ Clear feature importance patterns\n');
    else
        fprintf('⚠ Diffuse feature importance\n');
    end
    
    % Recommendations
    fprintf('\nRECOMMENDATIONS:\n');
    if any(xaiResults.featureImportance < 0.005)
        lowImpFeatures = sum(xaiResults.featureImportance < 0.005);
        fprintf('• Consider removing %d low-importance features\n', lowImpFeatures);
    end
    
    if isfield(xaiResults, 'confidenceMetrics') && xaiResults.confidenceMetrics.calibrationError > 0.08
        fprintf('• Apply confidence calibration techniques\n');
    end
end

function interactive_xai_exploration(model, features, labels, featureNames, xaiResults)
    % INTERACTIVE_XAI_EXPLORATION Allow user to explore XAI features
    
    fprintf('\nInteractive XAI Exploration:\n');
    fprintf('1. View feature explanations\n');
    fprintf('2. Test what-if scenarios\n');
    fprintf('3. Analyze specific samples\n');
    fprintf('4. Exit\n');
    
    choice = input('Choose an option (1-4): ');
    
    switch choice
        case 1
            explore_feature_explanations(xaiResults, featureNames);
        case 2
            test_what_if_scenarios(model, features, labels, featureNames, xaiResults);
        case 3
            analyze_specific_samples(model, features, labels, featureNames, xaiResults);
        case 4
            fprintf('Exiting XAI exploration.\n');
        otherwise
            fprintf('Invalid choice. Exiting.\n');
    end
end

function explore_feature_explanations(xaiResults, featureNames)
    % EXPLORE_FEATURE_EXPLANATIONS Interactive feature exploration
    
    fprintf('\n=== Feature Explanation Explorer ===\n');
    
    if ~isfield(xaiResults, 'featureImportance') || isempty(xaiResults.featureImportance)
        fprintf('Feature importance data not available.\n');
        return;
    end
    
    [sortedImp, sortedIdx] = sort(xaiResults.featureImportance, 'descend');
    
    for i = 1:min(5, length(sortedIdx))
        featIdx = sortedIdx(i);
        if featIdx <= length(featureNames)
            featName = featureNames{featIdx};
            
            fprintf('\n%d. %s (Importance: %.3f)\n', i, featName, sortedImp(i));
            
            if isfield(xaiResults, 'pdpData') && ~isempty(xaiResults.pdpData)
                % Find which PDP contains this feature
                for j = 1:length(xaiResults.pdpData)
                    if strcmp(xaiResults.pdpData(j).featureName, featName)
                        midPoint = mean(xaiResults.pdpData(j).predictions(:, 2));
                        fprintf('   Average effect on Funny/Happy probability: %.1f%%\n', midPoint * 100);
                        break;
                    end
                end
            end
        end
    end
end

function test_what_if_scenarios(model, features, labels, featureNames, xaiResults)
    % TEST_WHAT_IF_SCENARIOS Interactive what-if analysis
    
    fprintf('\n=== What-If Scenario Testing ===\n');
    
    % Use a random sample as base
    sampleIdx = randi(size(features, 1));
    baseSample = features(sampleIdx, :);
    
    [pred, scores] = predict(model, baseSample);
    confidence = max(scores);
    
    fprintf('Base sample: Predicted %s with %.1f%% confidence\n', ...
        ifelse(pred == 1, 'Relaxed', 'Funny/Happy'), confidence * 100);
    
    % Show top features that could change prediction
    if isfield(xaiResults, 'featureImportance')
        [~, topIdx] = sort(xaiResults.featureImportance, 'descend');
        
        fprintf('\nTop features to modify:\n');
        for i = 1:min(3, length(topIdx))
            featIdx = topIdx(i);
            if featIdx <= length(featureNames) && featIdx <= length(baseSample)
                currentVal = baseSample(featIdx);
                fprintf('%d. %s (current: %.2f)\n', i, featureNames{featIdx}, currentVal);
            end
        end
    end
end

function analyze_specific_samples(model, features, labels, featureNames, xaiResults)
    % ANALYZE_SPECIFIC_SAMPLES Analyze user-selected samples
    
    fprintf('\n=== Sample Analysis ===\n');
    
    [allPred, allScores] = predict(model, features);
    allConfidence = max(allScores, [], 2);
    
    % Use a random sample for analysis
    sampleIdx = randi(size(features, 1));
    sample = features(sampleIdx, :);
    trueLabel = labels(sampleIdx);
    [pred, scores] = predict(model, sample);
    confidence = max(scores);
    
    fprintf('\nSample Analysis Results:\n');
    fprintf('True label: %s\n', ifelse(trueLabel == 1, 'Relaxed', 'Funny/Happy'));
    fprintf('Predicted: %s\n', ifelse(pred == 1, 'Relaxed', 'Funny/Happy'));
    fprintf('Confidence: %.1f%%\n', confidence * 100);
    fprintf('Correct: %s\n', ifelse(pred == trueLabel, '✓ Yes', '✗ No'));
end

function result = ifelse(condition, trueVal, falseVal)
    % IFELSE Ternary operator replacement
    if condition
        result = trueVal;
    else
        result = falseVal;
    end
end

% Run the main program
main_emotion_predictions();