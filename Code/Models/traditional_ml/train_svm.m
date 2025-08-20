function model = train_svm(X_train, y_train)
    % Find dominant class for each sample
    [~, dominant_class] = max(y_train, [], 2);
    
    % Train multiclass SVM
    template = templateSVM('KernelFunction', 'rbf', ...
                          'Standardize', true, ...
                          'KernelScale', 'auto');
    
    model = fitcecoc(X_train, dominant_class, ...
                    'Learners', template, ...
                    'Coding', 'onevsone');
    
    % Cross-validation
    cvmodel = crossval(model, 'KFold', 5);
    loss = kfoldLoss(cvmodel);
    fprintf('Cross-validated accuracy: %.2f%%\n', (1-loss)*100);
end