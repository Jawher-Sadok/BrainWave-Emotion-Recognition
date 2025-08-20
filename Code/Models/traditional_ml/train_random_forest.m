function model = train_random_forest(X_train, y_train)
    % Convert to table (RF works better with tables)
    tbl = array2table(X_train);
    tbl.Label = y_train;
    
    % Train ensemble
    template = templateTree('Reproducible', true, 'MaxNumSplits', 100);
    model = fitcensemble(tbl, 'Label', ...
              'Method', 'Bag', ...
              'NumLearningCycles', 200, ...
              'Learners', template);
    
    % Feature importance
    imp = predictorImportance(model);
    figure;
    bar(imp);
    title('Feature Importance');
    xlabel('Features');
    ylabel('Importance');
end