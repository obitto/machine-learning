%1st dataset
D = 500; % set the dimensionality
% data for class 1
mu1 = 3*ones(1,D); Sigma1 = eye(D); N1 = 5000;
xdata1 = mvnrnd(mu1, Sigma1, N1);
% data for class 2
mu2 = 5*ones(1,D); Sigma2 = eye(D); N2 = 5000;
xdata2 = mvnrnd(mu2, Sigma2, N2);
xdata = [xdata1; xdata2];
labels = [ zeros(N1,1); ones(N2,1) ];
total = [xdata labels];
data = total(randperm(size(total,1)),:);
xdata = data(:,1:D);
labels = data(:,D+1);
%2nd dataset
%{
xdata = load('binary_features.txt'); 
labels = load('labels.txt');
total = [xdata labels];
total = repmat(total,5,1);
data = total(randperm(size(total,1)),:);
[N,D] =size(data);
xdata = data(:,1:D-1);
mu = zeros(1,D-1);
Sigma = eye(D-1);
xdata = xdata + mvnrnd(mu, Sigma, N);
labels = data(:,D);
%}
epsilon = 1e-5;
iterations = 1000;
M = 10;
t1 = cputime;
[w1 accuracy1 time1] = logistic_train(xdata,labels);
e1 = cputime - t1;
t2 = cputime;
[w2 accuracy2 time2] = logistic_train(xdata,labels,epsilon,iterations,1,M);
e2 = cputime - t2;
%h = sigmoid(data*w2);