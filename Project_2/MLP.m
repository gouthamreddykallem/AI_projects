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
y_train = y(c.training);
X_test = x(c.test,:);
y_test = y(c.test);




% Preprocess the data
X_train = 255 - X_train;
X_test = 255 - X_test;

% Split the data into training and testing sets
num_train = size(X_train, 1);
num_test = size(X_test, 1);


% Initialize the MLP
num_hidden_units = 100;
out_dim = 10;
W1 = randn([784, num_hidden_units]);
b1 = zeros([ 1,num_hidden_units]);
W2 = randn([num_hidden_units, out_dim]);
b2 = zeros([ 1,out_dim]);

% Initialize the learning rate
eta = 0.01;

% Train the MLP
for epoch = 1:100

  % Forward propagation
  Z1 =  X_train * W1 + b1;
  A1 = sigmoid(Z1);
  Z2 =  A1*W2 + b2;
  A2 = softmax(Z2);

  % Backpropagation
  dA2 = (A2 - y_train) / num_train;
  dZ2 = dA2 * W2';
  dW2 = dZ2 * A1';
  db2 = sum(dZ2, 1);
  dA1 =  dZ2 * W2;
  dZ1 = dA1' * sigmoid(Z1).^2;
  dW1 = dZ1' * X_train;
  db1 = sum(dZ1, 1);

  % Update the weights and biases
  W1 = W1 - eta * dW1;
  b1 = b1 - eta * db1;
  W2 = W2 - eta * dW2;
  b2 = b2 - eta * db2;
end

% Test the MLP
y_pred = predict(W1, b1, W2, b2, X_test);

% Compute the confusion matrix and accuracy
conf_mat = confusionmat(y_test, y_pred);
accuracy = sum(diag(conf_mat)) / num_test;

% Plot the confusion matrix
figure;
plotconfusionmat(conf_mat);
title('Confusion Matrix');
xlabel('Predicted');
ylabel('Actual');

% Print the accuracy
fprintf('Accuracy: %f\n', accuracy);









% 
% % Normalizing the input data
% X_train = X_train./255;
% X_test = X_test./255;
% 
% 
% % Initialize the MLP
% input_units = size(X_train, 1);
% output_units = 10; % number of classes
% hidden_units = 100; % you can try different values of hidden units
% 
% % Randomly initialize the weights
% W1 = randn(input_units,hidden_units);
% b1 = zeros(1,hidden_units);
% W2 = randn(hidden_units,output_units);
% b2 = zeros(1,output_units);
% 
% 
% % Train the MLP using stochastic gradient descent
% learning_rate = 0.1;
% batch_size = 64;
% num_epochs = 10; % you can try different values of epochs
% for epoch = 1:num_epochs
%     % Shuffle the training data
%     shuffled_indices = randperm(size(X_train, 2));
%     X_train = X_train(:, shuffled_indices);
%     y_train = y_train(shuffled_indices);
%     
%     % Train on batches
%     for i = 1:batch_size:size(X_train, 2)
%         % Get the current batch
%         batch_start = i;
%         batch_end = min(i+batch_size-1, size(X_train, 2));
%         X_batch = X_train(batch_start:batch_end,:);
%         y_batch = y_train(batch_start:batch_end);
%         
%         % Forward pass
%         z1 = W1*X_batch + b1;
%         a1 = sigmoid(z1);
%         z2 = W2*a1 + b2;
%         a2 = softmax(z2);
%         
%         % Backward pass
%         delta2 = a2 - one_hot(y_batch, output_units);
%         delta1 = (W2'*delta2) .* sigmoid_gradient(z1);
%         
%         % Update the weights
%         W2 = W2 - learning_rate*delta2*a1';
%         b2 = b2 - learning_rate*sum(delta2, 2);
%         W1 = W1 - learning_rate*delta1*X_batch';
%         b1 = b1 - learning_rate*sum(delta1, 2);
%     end
% end
% 
% % Evaluate on the test set
% z1_test = W1*X_test + b1;
% a1_test = sigmoid(z1_test);
% z2_test = W2*a1_test + b2;
% a2_test = softmax(z2_test);
% loss = cross_entropy_loss(a2_test, y_test);
% accuracy = mean(y_test == predict(a2_test));
% 
% % Compute the confusion matrix
% confusion = zeros(output_units);
% for i = 1:size(X_test, 2)
%     y_pred = predict(a2_test(:, i));
%     confusion(y_test(i)+1, y_pred+1) = confusion(y_test(i)+1, y_pred+1) + 1;
% end
% 
% % Display the results
% fprintf('Loss: %f\n', loss);
% fprintf('Accuracy: %f\n', accuracy);
% fprintf('Confusion matrix:\n');
% disp(confusion);
% 
% 
% function g = sigmoid_gradient(z)
% % Compute the gradient of the sigmoid function
% % z: input to the sigmoid function
%     g = sigmoid(z) .* (1 - sigmoid(z));
% end
% 
% 
function s = sigmoid(x)
     s = 1 ./ (1 + exp(-x));
end
% 
% 
% function loss = cross_entropy_loss(y_pred, y_true)
% % Computes the cross-entropy loss between the predicted and true labels
% 
%     num_samples = size(y_pred, 2);
%     y_true_1based = y_true + 1;  % Add 1 to convert to 1-based indices
%     log_likelihood = log(y_pred(sub2ind(size(y_pred), y_true_1based', 1:num_samples)));
%     loss = -sum(log_likelihood) / num_samples;
%     
% end
% 
% 
% function y = one_hot(labels, num_classes)
% % Convert labels to one-hot encoding
% % labels: vector of integer labels
% % num_classes: number of classes
%     y = zeros(num_classes, length(labels));
%     for i = 1:length(labels)
%         y(labels(i)+1, i) = 1;
%     end
% end


