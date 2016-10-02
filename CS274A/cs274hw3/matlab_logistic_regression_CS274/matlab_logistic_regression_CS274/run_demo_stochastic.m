% demo of training logistic regression models, comparing different
% optimization methods
%
%                                      CS 274A, Winter 2016


%% SPAM EMAIL DATA SET
load binary_features.txt
load labels.txt
fprintf('\nRESULTS ON SPAM EMAIL DATA:');
% first train with Newton's method using the Hessian
newton = 1;
maxiterations = 2000;
learning_rate = 0.0001; %ignored for the Newton method
logistic_train(binary_features,labels,learning_rate,maxiterations,newton);
%Iterations =  11 	 Mean logloss = 0.15625  	 Accuracy = 94.50 
%Elapsed time is 1.395244 seconds.

% now train with first-order batch gradient
newton = 0; 
logistic_train(binary_features,labels,learning_rate,maxiterations,newton);
% Iterations = 1300 	 Mean logloss = 0.16283  	 Accuracy = 94.26 
% Elapsed time is 1.226829 seconds.

% now train with stochastic gradient
lrate0 = 10^(-2);  % initial learning rate  
minibatch = 200;
maxiterations = 500;
logistic_train_stochastic(binary_features,labels,lrate0,minibatch, maxiterations);

fprintf('\n hit any key to continue....\n');
pause

%% SIMULATED DATA, PART 1
% now run with simulated data in different dimensions
mu1 = [3 3]; Sigma1 = [3 0; 0 3]; N1 = 100000;
xdata1 = [mvnrnd(mu1, Sigma1, N1)];
% data for class 2
mu2 = [5 5]; Sigma2 = [1 0; 0 1]; N2 = 100000;
xdata2 = [mvnrnd(mu2, Sigma2, N2)];
simdata  = [xdata1; xdata2];
simlabels = [ zeros(N1,1); ones(N2,1) ];

fprintf('\nRESULTS ON SIMULATED DATA, n = %d, d = %d:',N1+N2, 2);
% stochastic gradient....
lrate0 = 10^(-3);  % initial learning rate  
minibatch = 200;
maxiterations = 500;
logistic_train_stochastic(simdata,simlabels,lrate0,minibatch, maxiterations);
% Results with stochastic gradient method:
% Iterations = 500 	 Mean logloss = 0.37059  	 Accuracy = 85.48 
% Learning rate (initial) = 0.00100 	 Learning rate (final) = 0.00067 	 Minibatch size =  200 
% Elapsed time is 0.274075 seconds.


% batch gradient...
newton = 0;
maxiterations = 2000;
learning_rate = 0.000001;  
logistic_train(simdata,simlabels,learning_rate,maxiterations,newton);
% Results with first-order gradient method:
% Iterations = 1690 	 Mean logloss = 0.35011  	 Accuracy = 85.49 
% Elapsed time is 18.894716 seconds.


% Newton's method, with one tenth of the data (may not generalize as well
% to test data given that it is only able to use 1/10th of the training data)
% (Note that this may still require a lot of memory)
N1 = 10000;N2=10000;
simdata  = [xdata1(1:N1,:); xdata2(1:N2,:)];
simlabels = [ zeros(N1,1); ones(N2,1) ];
newton = 1;
logistic_train(simdata,simlabels,learning_rate,maxiterations,newton);
%Results with second-order Newton method:
% Iterations =   7 	 Mean logloss = 0.34712  	 Accuracy = 85.49 
% Data Set: n = 20000, d = 3
% Elapsed time is 17.152520 seconds.
 
fprintf('\n hit any key to continue....\n');
pause

%% SIMULATED DATA, PART 2: now in higher dimensions
d = 50;  % go from 2 dimensions to 50

% generate data that is separated in the first 2 dimensions as before
% with d-2 additional dimensions that have the same mean for both
% classes (so these dimensions are essentially noise and the
% best attainable accuracy will remain at about 0.85 as before)
mu1 = [3 3 3*ones(1,d-2)];   Sigma1 = 3*eye(d); %Sigma1(1,1) = 3; Sigma1(2,2) = 3;
N1 = 100000;
xdata1 = [mvnrnd(mu1, Sigma1, N1)];
% data for class 2
mu2 = [5 5 3*ones(1,d-2)]; Sigma2 =  eye(d); %Sigma2(1,1) = 1; Sigma2(2,2) = 1;
N2 = 100000;
xdata2 = [mvnrnd(mu2, Sigma2, N2)];
simdata  = [xdata1; xdata2];
simlabels = [ zeros(N1,1); ones(N2,1) ];


fprintf('\nRESULTS ON SIMULATED DATA, n = %d, d = %d:',N1+N2, d);
% stochastic gradient....
lrate0 = 10^(-4);  % initial learning rate (smaller than before)
minibatch = 200;
maxiterations = 10000;  % run for longer...takes longer to converge on this data
logistic_train_stochastic(simdata,simlabels,lrate0,minibatch, maxiterations);
% Results with stochastic gradient method:
% Iterations = 10000 	 Mean logloss = 0.36300  	 Accuracy = 84.72 
% Learning rate (initial) = 0.00010 	 Learning rate (final) = 0.00010 	 Minibatch size =  200 
% Data Set: n = 200000, d = 51
% Elapsed time is 1.341190 seconds.

% Newton's method, with one tenth of the data (may not generalize as well
% to test data given that it is only able to use 1/10th of the training data)
% (Note that this may still require a lot of memory)
N1 = 10000;N2=10000;
simdata  = [xdata1(1:N1,:); xdata2(1:N2,:)];
simlabels = [ zeros(N1,1); ones(N2,1) ];
newton = 1;
logistic_train(simdata,simlabels,learning_rate,maxiterations,newton);
% Results with second-order Newton method: 
% Iterations =   7 	 Mean logloss = 0.34860  	 Accuracy = 85.27 
% Data Set: n = 20000, d = 51
% Elapsed time is 18.594812 seconds.

 



