function net = lstm_model(inputSize, numClasses)
    layers = [
        sequenceInputLayer(inputSize)
        
        bilstmLayer(128, 'OutputMode', 'sequence')
        batchNormalizationLayer
        reluLayer
        
        bilstmLayer(64, 'OutputMode', 'last')
        batchNormalizationLayer
        reluLayer
        
        fullyConnectedLayer(numClasses)
        softmaxLayer
        classificationLayer];
    
    options = trainingOptions('adam', ...
        'MaxEpochs', 25, ...
        'MiniBatchSize', 16, ...
        'SequenceLength', 'longest', ...
        'ValidationData', {X_val, y_val}, ...
        'Plots', 'training-progress', ...
        'ExecutionEnvironment', 'gpu');
    
    net = trainNetwork(X_train, y_train, layers, options);
end