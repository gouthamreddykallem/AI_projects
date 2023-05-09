

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
X_train = x(c.training,:);
y_train = y(c.training);
X_test = x(c.test,:);
y_test = y(c.test);






train_data = (X_train)/255;
test_data = (X_test)/255;


k = 1;
[predictions, accuracy] = knn(train_data, y_train, test_data, y_test, k);
fprintf('Accuracy with k=%d: %f\n', k, accuracy);

% Test the k-nearest neighbors algorithm with k=5
k = 5;
[predictions, accuracy] = knn(train_data, y_train, test_data, y_test, k);
fprintf('Accuracy with k=%d: %f\n', k, accuracy);

% Compute confusion matrix
C = confusionmat(y_test, predictions);
disp('Confusion matrix:');
disp(C);

figure;
plotconfusion(categorical(y_test), categorical(predictions));



function [predictions, accuracy] = knn(train_data, train_labels, test_data, test_labels, k)
% K-Nearest Neighbors Algorithm
% train_data: n-by-d matrix of training data
% train_labels: n-by-1 vector of labels for training data
% test_data: m-by-d matrix of test data
% test_labels: m-by-1 vector of true labels for test data
% k: number of neighbors to consider
% predictions: m-by-1 vector of predicted labels for test data
% accuracy: scalar accuracy of predictions

  % Calculate pairwise distances between test and training data
  distances = pdist2(test_data, train_data);

  % Sort distances and get indices of k nearest neighbors
  [~, indices] = sort(distances, 2);
  indices = indices(:, 1:k);

  % Get labels of k nearest neighbors
  nearest_labels = train_labels(indices);

  % Make predictions based on majority vote
  predictions = mode(nearest_labels, 2);

  % Calculate accuracy
  accuracy = sum(predictions == test_labels) / numel(test_labels);

end
