function [h, p] = statistical_tests(model1Perf, model2Perf, testType)
    % Perform statistical tests between two models
    % Inputs:
    %   model1Perf - performance metrics for model 1 (array)
    %   model2Perf - performance metrics for model 2 (array)
    %   testType - 't-test', 'wilcoxon', or 'anova'
    
    switch lower(testType)
        case 't-test'
            % Paired t-test
            [h, p] = ttest(model1Perf, model2Perf);
            disp('Paired t-test results:');
            disp(['p-value: ' num2str(p)]);
            disp(['Significant difference (α=0.05): ' num2str(h)]);
            
        case 'wilcoxon'
            % Wilcoxon signed-rank test
            [p, h] = signrank(model1Perf, model2Perf);
            disp('Wilcoxon signed-rank test results:');
            disp(['p-value: ' num2str(p)]);
            disp(['Significant difference (α=0.05): ' num2str(h)]);
            
        case 'anova'
            % One-way ANOVA
            [p, tbl] = anova1([model1Perf, model2Perf], {'Model1', 'Model2'}, 'off');
            h = p < 0.05;
            disp('ANOVA results:');
            disp(tbl);
            disp(['p-value: ' num2str(p)]);
            disp(['Significant difference (α=0.05): ' num2str(h)]);
            
        otherwise
            error('Unknown test type');
    end
    
    % Visualization
    figure;
    boxplot([model1Perf, model2Perf], 'Labels', {'Model 1', 'Model 2'});
    ylabel('Performance Metric');
    title(['Statistical Comparison (' testType ')']);
end