function [weights] = logistic_train(traindata,trainlabels,lrate,maxiterations,newton);
% gradient (first-order) or Newton (second-order) training of a logistic regression model
%
% INPUT:
% traindata: n x d matrix of input training data
% trainlabels:  n x 1 vector of corresponding labels, 0 or 1
% lrate: learning rate (for gradient method - ignored by Hessian method)
% maxiterations: the maximum number of iterations to run  
% newton: set to 1 to run Newton's method, otherwise batch gradient updates will be used
%
% OUTPUT:
% weights: (d+1) x 1 vector of learned weights
%
%                                       Padhraic Smyth, CS 274A, Winter 2016

% should add some error checking here, e.g., that dimensions of labels and
% data agree.....

n = size(traindata,1);
data = [traindata ones(n,1)];  % add an extra column at the end for the intercept
d = size(data,2);
y = trainlabels; 

% initialize/set various variables
epsilon = 10^(-5);  % stop iterating when mean change in predictions (from one iteration to the next) per data point < epsilon
delta = 1;  % initial value for delta (which is the mean change in predictions from one iteration to the next) 
hessian_eps = 10^(-5);  % add this to the diagonal of the Hessian before inverting (for numerical stability)
lrate0 = lrate;  % initial learning rate (for gradient method)
lrate_flag = 0;  % set to 1 if we want the learning to be decreased at every iteration (for gradient method)

weights = zeros(d,1);   % initial weight values
predictions = ones(n,1)*0.5;   % initial n by 1 set of default predictions
accuracy = zeros(maxiterations,1);
logloss = nan(maxiterations,1);
logloss(1) = inf;
deltalogloss = -inf;

i = 1;
tstart = tic;
% continue to update weights as long as (a) number of iterations < maxiterations,
% and (b) change in predictions is < epsilon, and (c) logloss is decreasing
while( i < maxiterations && delta > epsilon &&  deltalogloss < 0)
    
    % Stor the predictions and logloss from the previous iteration
    oldpredictions = predictions; 
    oldlogloss = logloss;
    
    % Compute the gradient
    errors = (y - oldpredictions); % n by 1 vector of prediction errors
    deriv = -errors'*data;  % (1 by n) by (n by d) -> 1 by d vector of derivatives (the gradient vector)
    
    % Update the weights
    if newton==1  % use second-order Hessian
        V = oldpredictions .* (1-oldpredictions); 
        V = eye(n,n) .* repmat(V,[1,n]);  % n by n matrix with f(1-f) on diagonals
        H = data'*V*data + hessian_eps*eye(d);  % d x d matrix, with "hessian_eps" added to diagonal terms for stability
        weights = weights - inv(H)*deriv';  % d by 1 vector
    else  % use gradient
        weights = weights - lrate*deriv';  % d by 1 vector
    end
    i = i + 1;
    
    % Compute the prediction vector and the overall log-likelihood and accuracy
    predictions = 1./(1 + exp(-data*weights)); % n by 1 vector of predictions with the updated weights
    logloss(i) =   - y'*log(predictions) - (1-y)'*log(1-predictions);  % scalar log-loss (negative log-likelihood for binary y)
    accuracy(i) = 100*( 1 - mean(  abs( y - (predictions>0.5) ) ) );  % classification accuracy on training data (as percentage)
     
    % Compute convergence metrics    
    delta = mean(abs( predictions - oldpredictions ) );
    deltalogloss =  logloss(i) - logloss(i-1);
    
    if lrate_flag == 1 & newton==0  % decrease the learning rate for the gradient method?
        lrate = lrate0/(1+lrate0*i);  %  decrease learning rate slightly at each iteration (heuristic from Bottou 2012 paper)
    end
           
end

% Print various summary results to the screen
if newton==1
    fprintf('\nResults with second-order Newton method:\n');
    tstr = ['TRAINING WITH NEWTON METHOD'];
else
    fprintf('\nResults with first-order gradient method:\n'); 
    tstr = ['TRAINING WITH BATCH GRADIENT METHOD'];
end
fprintf('Iterations = %3d \t Mean logloss = %6.5f  \t Accuracy = %4.2f \n',i,logloss(i)/n,accuracy(i));
if newton==0
fprintf('Learning rate (initial) = %6.5f \t Learning rate (final) = %6.5f  \n',lrate0,lrate);
end
fprintf('Data Set: n = %d, d = %d\n',n,d);
% Report how long the algorithm took (since the 'tic' command)
toc(tstart)

% Display convergence results in a figure
figure; 
subplot(2,1,1); plot(1:i,logloss(1:i)/n,'.:','linewidth',2); xlabel('ITERATION'); ylabel('MEAN LOG LOSS'); title(tstr);
subplot(2,1,2); plot(1:i,accuracy(1:i),'.:','linewidth',2); xlabel('ITERATION'); ylabel('ACCURACY');

end

