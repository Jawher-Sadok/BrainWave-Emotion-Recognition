function net = cnn_model(inputSize, numClasses)
    layers = [
        imageInputLayer([inputSize 1 1], 'Normalization', 'zscore')
        
        convolution2dLayer(3, 32, 'Padding', 'same')
        batchNormalizationLayer
        reluLayer
        maxPooling2dLayer(2, 'Stride', 2)
        
        convolution2dLayer(3, 64, 'Padding', 'same')
        batchNormalizationLayer
        reluLayer
        maxPooling2dLayer(2, 'Stride', 2)
        
        convolution2dLayer(3, 128, 'Padding', 'same')
        batchNormalizationLayer
        reluLayer
        
        fullyConnectedLayer(256)
        reluLayer
        dropoutLayer(0.5)
        
        fullyConnectedLayer(numClasses)
        softmaxLayer
        classificationLayer];
    
    options = trainingOptions('adam', ...
        'MaxEpochs', 30, ...
        'MiniBatchSize', 32, ...
        'ValidationData', {X_val, y_val}, ...
        'Plots', 'training-progress', ...
        'ExecutionEnvironment', 'gpu');
    
    net = trainNetwork(X_train, y_train, layers, options);
end