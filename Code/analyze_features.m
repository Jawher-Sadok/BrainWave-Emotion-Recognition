function analyze_features(features, labels, featureNames)
    % ANALYZE_FEATURES Perform detailed analysis of features
    % Input: features - feature matrix
    %        labels - emotion labels
    %        featureNames - names of features
    
    fprintf('\n=== Feature Analysis ===\n');
    
    % Calculate correlation matrix
    corrMatrix = corr(features);
    
    % Calculate feature statistics by class
    relaxed_features = features(labels == 1, :);
    funny_features = features(labels == 2, :);
    
    fprintf('\nFeature Statistics by Class:\n');
    fprintf('%-20s %-15s %-15s %-15s\n', 'Feature', 'Relaxed Mean', 'Funny Mean', 'p-value');
    fprintf('%s\n', repmat('-', 65, 1));
    
    p_values = zeros(length(featureNames), 1);
    for i = 1:length(featureNames)
        [~, p] = ttest2(relaxed_features(:, i), funny_features(:, i));
        p_values(i) = p;
        fprintf('%-20s %-15.3f %-15.3f %-15.4f\n', ...
            featureNames{i}, ...
            mean(relaxed_features(:, i)), ...
            mean(funny_features(:, i)), ...
            p);
    end
    
    % Visualize analysis
    visualize_feature_analysis(features, labels, featureNames, corrMatrix, p_values);
end

function visualize_feature_analysis(features, labels, featureNames, corrMatrix, p_values)
    % Visualize feature analysis results
    
    figure('Name', 'Feature Analysis', 'Position', [100, 100, 1200, 800]);
    
    % Correlation heatmap
    subplot(2, 2, 1);
    imagesc(corrMatrix);
    colorbar;
    colormap(jet);
    set(gca, 'XTick', 1:length(featureNames), 'XTickLabel', featureNames, 'XTickLabelRotation', 45);
    set(gca, 'YTick', 1:length(featureNames), 'YTickLabel', featureNames);
    title('Feature Correlation Matrix');
    
    % PCA visualization
    subplot(2, 2, 2);
    [coeff, score] = pca(features);
    gscatter(score(:, 1), score(:, 2), labels, 'br', 'o', 8, 'off');
    xlabel('Principal Component 1');
    ylabel('Principal Component 2');
    legend('Relaxed', 'Funny_Happy');
    title('PCA Projection');
    grid on;
    
    % Feature means by class
    subplot(2, 2, 3);
    relaxed_means = mean(features(labels == 1, :));
    funny_means = mean(features(labels == 2, :));
    bar([relaxed_means; funny_means]');
    set(gca, 'XTickLabel', featureNames, 'XTickLabelRotation', 45);
    ylabel('Mean Value');
    legend('Relaxed', 'Funny/Happy');
    title('Feature Means by Class');
    grid on;
    
    % Statistical significance (-log10(p-values))
    subplot(2, 2, 4);
    bar(-log10(p_values));
    hold on;
    plot(xlim, [-log10(0.05) -log10(0.05)], 'r--', 'LineWidth', 2);
    hold off;
    set(gca, 'XTick', 1:length(featureNames), 'XTickLabel', featureNames, 'XTickLabelRotation', 45);
    ylabel('-log10(p-value)');
    title('Statistical Significance');
    legend('Features', 'p=0.05 threshold');
    grid on;
    
    sgtitle('Comprehensive Feature Analysis');
end