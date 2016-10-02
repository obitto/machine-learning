function [weights] = logistic_train_stochastic(traindata,trainlabels,lrate0,minibatch, maxiterations);
% stochastic gradient training of the logistic regression model
% (ideally this should share code with logistic_train.m rather than being a
% separate function)
%
% INPUT:
% traindata: n x d matrix of input training data
% trainlabels:  n x 1 vector of corresponding labels, 0 or 1
% lrate0: initial learning rate
% minibatch: the size of minibatches used to compute the gradient
% maxiterations: the number of iterations to run (fixed for demo purposes)
%
% OUTPUT:
% weights: (d+1) x 1 vector of learned weights
%
%                                       Padhraic Smyth, CS 274A, Winter 2016

% should add some error checking here, e.g., that dimensions of labels and
% data agree.....


n = size(traindata,1);
index = randperm(n);  % shuffle the order of the data randomly
data = [traindata(index,:) ones(n,1)];  % add an extra column at the end for the intercept
d = size(data,2);
y = trainlabels(index);

% initialize/set various variables
m = minibatch;  % minibatch size    
lrate = lrate0;  % initial learning rate  
lrate_flag = 0;  % set to 1 if we want the learning to be decreased at every iteration 
 
weights = zeros(d,1);
predictions = ones(n,1)*0.5; 
delta = 1;
accuracy = zeros(maxiterations,1);
logloss = nan(maxiterations,1);
logloss(1) = inf;
deltalogloss = -inf;  
i = 1; e = 0; 

tstart = tic;
% continue to update weights for a fixed number of iterations
% (this is done for simplicity here - a better way to do this would be to
% compute the gradient over multiple minibatches every K iterations and
% check for convergence automatically)
while( i < maxiterations  )
    
    oldpredictions = predictions; 
    oldlogloss = logloss;
    
    s = e+1; e = s+m-1; % figure out the start (s) and end (e) indices for the minibatch
    if e>n  % go back to the first row if the end of the minibatch extends beyond the data (this is not the correct way to do this...but will be ok if m << N)
        s=1; e = m;
    end
    
    by = y(s:e);  % select the minibatch targets
    bdata = data(s:e,:);  % select the minibatch inputs
    bpredictions = 1./(1 + exp(-bdata*weights)); % m by 1 vector of minibatch predictions with the current weights
    
    errors = (by - bpredictions); % m by 1 vector of prediction errors on the minibatch data
    deriv = -errors'*bdata;  % (1 by m) by (m by d) -> 1 by d vector of derivatives (the stochastic gradient vector)
    weights = weights - lrate*deriv';  % d by 1 vector for the new weights
    
%   The Newton method is normally not used with stochastic methods (so its commented out here)
%   if newton==1  % use second-order Hessian (normall not used with
%         V = oldpredictions .* (1-oldpredictions); 
%         V = eye(n,n) .* repmat(V,[1,n]);  % n by n matrix with f(1-f) on diagonals
%         H = data'*V*data;  % d x d matrix
%         weights = weights - inv(H)*deriv';  % d by 1 vector 
%     end
    
    i = i + 1;
    % m by 1 vector of predictions for the minibatch with the updated weights
    bpredictions = 1./(1 + exp(-bdata*weights)); 
    
    % compute the log-loss on the minibatch (negative log-likelihood for binary y)
    logloss(i) =   - by'*log(bpredictions) - (1-by)'*log(1-bpredictions);  
    
    % compute classification accuracy on the minibatch (as percentage)
    accuracy(i) = 100*( 1 - mean(  abs( by - (bpredictions>0.5) ) ) );  
    
     if lrate_flag == 1 % decrease the learning rate?
        lrate = lrate0/(1+lrate0*i);  %  decrease learning rate slightly at each iteration (heuristic from Bottou 2012 paper)
     end
     
end  

% Summarize the results after training....
predictions = 1./(1 + exp(-data*weights)); % m by 1 vector of predictions with the updated weights
logloss(i+1) =   - y'*log(predictions) - (1-y)'*log(1-predictions);  % scalar log-loss (negative log-likelihood for binary y)
accuracy(i+1) = 100*( 1 - mean(  abs( y - (predictions>0.5) ) ) );  % classification accuracy on training data (as percentage)

fprintf('\nResults with stochastic gradient method:\n'); 
fprintf('Iterations = %3d \t Mean logloss = %6.5f  \t Accuracy = %4.2f \n',i,logloss(i+1)/n,accuracy(i+1));
fprintf('Learning rate (initial) = %6.5f \t Learning rate (final) = %6.5f \t Minibatch size = %4d \n',lrate0,lrate,m);
fprintf('Data Set: n = %d, d = %d\n',n,d);
toc(tstart)

figure;  % plot the progression of learning over iterations....
tstr = ['TRAINING WITH STOCHASTIC GRADIENT'];
subplot(2,1,1); plot(1:i,logloss(1:i)/m,'.:','linewidth',2); xlabel('ITERATION'); ylabel('MEAN LOG LOSS');title(tstr);
hold on; plot(1:i, logloss(i+1)*ones(i,1)/n,'r-','linewidth',4);  % plot the mean log loss on the full data set
subplot(2,1,2); plot(1:i,accuracy(1:i),'.:','linewidth',2); xlabel('ITERATION'); ylabel('ACCURACY');
hold on; plot(1:i, accuracy(i+1)*ones(i,1),'r-','linewidth',4);  % plot the mean accuracy on the full data set


