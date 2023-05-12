% Load and preprocess the MNIST dataset
close all;
clear;
clc;

load('mnist_test.mat');
x = test_data';
y = test_label;
% Set the ratio of data to use for training
trainingRatio = 0.8;

% Create a partition object for stratified random partitioning
c = cvpartition(y,'Holdout',1-trainingRatio);

% Split the data into training and testing sets
X_train = double(x(c.training,:));
y_train = double(y(c.training));
X_test = double(x(c.test,:));
y_test = double(y(c.test));


train_data = (X_train)/255;
test_data = (X_test)/255;
% train_labels = y_train;
% test_labels = y_test;

% Define the MLP architecture
input_size = size(train_data, 2);
output_size = 10; % number of classes
hidden_sizes = [50, 100, 200]; % different numbers of hidden units
activation_fn = @sigmoid;
learning_rate = 0.1;
num_epochs = 5;
batch_size = 100;

% Initialize the weights and biases randomly
W1 = randn(input_size, hidden_sizes(1))/sqrt(input_size);
b1 = zeros(1, hidden_sizes(1));
W2 = randn(hidden_sizes(1), output_size)/sqrt(hidden_sizes(1));
b2 = zeros(1, output_size);

% Train the MLP using stochastic gradient descent
for h = 1:length(hidden_sizes)
    hidden_size = hidden_sizes(h);
    W1 = randn(input_size, hidden_size)/sqrt(input_size);
    b1 = zeros(1, hidden_size);
    W2 = randn(hidden_size, output_size)/sqrt(hidden_size);
    b2 = zeros(1, output_size);

    for epoch = 1:num_epochs
        % Shuffle the training data
        idx = randperm(size(train_data, 1));
        train_data = train_data(idx, :);
        y_train = y_train(idx, :);

        for batch = 1:floor(size(train_data, 1)/batch_size)
            % Select a mini-batch of training data
            batch_data = train_data((batch-1)*batch_size+1:batch*batch_size, :);
            batch_labels = y_train((batch-1)*batch_size+1:batch*batch_size, :);

            % Forward pass
            z1 = batch_data*W1 + b1;
            a1 = activation_fn(z1);
            z2 = a1*W2 + b2;
            output = softmax(z2);

            % Backward pass
            delta2 = output - batch_labels;
            delta1 = (delta2*W2').*activation_fn(z1).*(1-activation_fn(z1));

            % Update weights and biases
            W2 = W2 - learning_rate*a1'*delta2;
            b2 = b2 - learning_rate*sum(delta2, 1);
            W1 = W1 - learning_rate*batch_data'*delta1;
            b1 = b1 - learning_rate*sum(delta1, 1);
        end

        % Evaluate the performance on the test set
%         z1 = test_data*W1 + b1;
%         a1 = activation_fn(z1);
%         z2 = a1*W2 + b2;
%         output = softmax(z2);
%         [~, predicted_labels] = max(output, [], 2);
%         confusion = confusionmat(y_test, predicted_labels);
%         accuracy = sum(diag(confusion))/sum(confusion(:));
%         fprintf('Hidden units: %d, Epoch: %d, Accuracy: %.2f%%\n', hidden_size, epoch, accuracy*100);
    end


z3 = test_data*W1 + b1;
a1 = sigmoid(z3);
z4 = a1*W2 + b2;
y_pred = softmax(z4);
[ ~,y_pred] = max(y_pred, [], 2);
[y_test, ~] = max(y_test, [], 2);
% accuracy = sum(y_pred == test_labels)/length(test_labels);

accuracy = sum(y_pred == y_test) / numel(y_test);

fprintf('Accuracy with hidden unit %d is = %0.4f\n',  hidden_size,  accuracy*100);


end


figure;
plotconfusion(categorical(y_test), categorical(y_pred));

% Computing the confusion matrix
C = confusionmat(y_test, y_pred);
disp("Confusion Matrix:");
disp(C);

function s = sigmoid(x)
     s = 1 ./ (1 + exp(-x));
end