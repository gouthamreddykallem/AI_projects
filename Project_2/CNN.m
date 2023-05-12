load('mnist_test.mat');

X = double(test_data)/255;
y = categorical(test_label);


% Reshaping the test data
X = reshape(X, [28, 28, 1, numel(y)]);


% Split the data into training and testing sets
trainingRatio = 0.8;
c = cvpartition(numel(y),'HoldOut',1-trainingRatio);

X_train = X(:,:,:,training(c));
y_train = y(training(c));
X_test = X(:,:,:,test(c));
y_test = y(test(c));


% Define the CNN architecture
layers = [    
    imageInputLayer([28 28 1])
    convolution2dLayer(5,20,'Padding',1)
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(5,50,'Padding',1)
    batchNormalizationLayer
    maxPooling2dLayer(2,'Stride',2)
    fullyConnectedLayer(500)
    reluLayer
    dropoutLayer(0.5)
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer
    ];

% Train the CNN
options = trainingOptions('sgdm', ...
    'MaxEpochs',3, ...
    'InitialLearnRate',0.01, ...
    'MiniBatchSize',128, ...
    'ExecutionEnvironment','auto', ...
    'Shuffle','every-epoch', ...
    'ValidationData',{X_test,y_test}, ...
    'ValidationFrequency',30, ...
    'Plots','training-progress');

net = trainNetwork(X_train,y_train,layers,options);

% Test the CNN
YPred = classify(net,X_test);
accuracy = sum(YPred == y_test)/numel(y_test);
fprintf('Test Accuracy: %.2f%%\n', accuracy*100);

% Plot the confusion matrix
figure;
plotconfusion(y_test, YPred);

