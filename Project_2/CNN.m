close all
clear
clc

load('mnist_test.mat');
x = test_data';
y = test_label;
% Set the ratio of data to use for training
trainingRatio = 0.8;

% Create a partition object for stratified random partitioning
c = cvpartition(y,'Holdout',1-trainingRatio);

% Split the data into training and testing sets
X_train = x(c.training,:);
y_train = categorical(y(c.training));
X_test = x(c.test,:);
y_test = categorical(y(c.test));

% Normalizing the input data
X_train = double(X_train)/255;
X_test = double(X_test)/255;


% Reshape the input data into a 28x28x1 image format
X_train = reshape(X_train, 28, 28, 1, []);
X_test = reshape(X_test, 28, 28, 1, []);

layers = [
    imageInputLayer([28 28 1])
    convolution2dLayer(5, 20, 'Padding', 2)
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    convolution2dLayer(5, 50, 'Padding', 2)
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    fullyConnectedLayer(500)
    reluLayer
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer
];

% Specify the training options
options = trainingOptions('sgdm', ...
    'MaxEpochs', 10, ...
    'InitialLearnRate', 0.0001, ...
    'Verbose', true, ...
    'Plots', 'training-progress');

% Train the CNN using the specified options
net = trainNetwork(X_train, y_train, layers, options);

% Evaluate the trained CNN on the test set

y_pred = classify(net, X_test);
accuracy = sum(y_pred == y_test) / numel(y_test);
fprintf('Test accuracy = %0.4f\n', accuracy*100);

% Plot the confusion matrix
figure;
plotconfusion(y_test, y_pred);

