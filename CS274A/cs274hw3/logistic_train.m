function [ weights accuracy time] = logistic_train( data,labels,epsilon,maxiterations,SGflag,M)
%LOGISTIC Summary of this function goes here
%   Detailed explanation goes here
if nargin < 3
    epsilon = 1e-5;
end
if nargin < 4
    maxiterations = 1000;
end
if nargin <5
    SGflag = 0;
end
if nargin <6
    M = 5;
end
[n,d] = size(data);
data = [ones(n,1),data];
iterations = zeros(maxiterations,1);
accuracy = zeros(maxiterations,1);
time = zeros(maxiterations,1);
cost = zeros(maxiterations,1);
diff = zeros(maxiterations,1);
weights = zeros(d+1,1);
lambda = 0.001;
alpha = 1e-6;
count = 1;
start = cputime;
next = cputime;
if SGflag ~= 1
    for i = 1:maxiterations
        h = sigmoid(data * weights);
        grad = (1/n) .* data' * (h-labels);
        Hessian = (1/n) .* data' * diag(h) *diag(1-h) * data;
        [p,q] = size(Hessian);
        weights = weights - (Hessian+alpha*eye(p,q))\grad;
        phat =sigmoid(data*weights);
        chat = phat>0.5;
        accuracy(i) = 100*sum(labels==chat)/length(labels);
        iterations(i) = i;
        cost(i) =(1/n)*sum(-labels.*log(h) - (1-labels).*log(1-h));
        %{
        e = cputime;
        if (e-next) >= 0.1
            accuracy(count) = 100*sum(labels==chat)/length(labels);
            time(count) = e - start;
            count = count + 1;
            next = e;
        end
        %}
        if(sum(abs(phat-h))< (n * epsilon))
            iter = i;
            %fprintf('%3d\n',accuracy(i));
            break;
        end
    end
else
    i = 1;
    iter = 0;
    while 1
        iter = iter +1;
        if (i+M-1) < n
            x = data(i:(i+M-1),:);
            y = labels(i:(i+M-1),:);
            i = i+M;
        else
            x = data(i:n,:);
            y = labels(i:n,:);
            i = 1;
        end
        h = sigmoid(x * weights);
        oldh = sigmoid(data * weights);
        grad = ((y-h)' * x);
        weights = weights + lambda * grad';
        phat = sigmoid(data*weights);
        chat = phat>0.5;
        accuracy(iter) = 100*sum(labels==chat)/length(labels);
        iterations(iter) = iter;
        cost(iter) =(1/n)*sum(-labels.*log(phat) - (1-labels).*log(1-phat));
        diff(iter) = sum(abs(phat-oldh));
        %{
        e = cputime;
        if (e-next) >= 0.1
            accuracy(count) = 100*sum(labels==chat)/length(labels);
            time(count) = e - start;
            count = count + 1;
            next = e;
        end
        %}
        if(sum(abs(phat-oldh))< (n * epsilon))|| (iter >= maxiterations)
            break;
        end
    end
end
%{
%plot accuracy/time
figure,plot(time(1:count-1),accuracy(1:count-1));
ylabel('accuracy/%');
xlabel('time/S');
%}

%plot accuracy/iteration
fprintf('%3d',iter);
figure,plot(iterations(1:iter),accuracy(1:iter));
ylabel('accuracy');
xlabel('iteraions');
figure,plot(iterations(1:iter),cost(1:iter));
ylabel('cost');
xlabel('iteraions');
end
