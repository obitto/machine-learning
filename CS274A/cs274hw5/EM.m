function [ gparams,weight,like ] = EM( data,K ,maxiterations)
%EM Summary of this function goes here
%   data is a n*d array, n sample with d features.
[m,n] = size(data);
smin = 0.001;
weight = rand([m,K]);
theta = zeros(K,n);
C = zeros(n,n,K);
normal = sum(weight,2);
newW = zeros(m,K);
newtheta = zeros(K,n);
newcenter = zeros(K,n);
iteration = zeros(100,1);
likelihood = zeros(100,1);
newcovariance = zeros(n,n,K);
for i = 1:K
    weight(:,i) = weight(:,i) ./ normal;
end
[~,center] = kmeans(data,K,100,0);
C(:,:,1) = cov(data);
C = repmat(C(:,:,1),1,1,K);
Q = zeros(m,K);
count = 0;
%{
for k = 1:K
    gparams(k) = struct('mean',center(k,:),'covariance',C(:,:,k));
end
figure;
hold on;
[~,indx] = max(weight,[],2);
for i = 1:k
    ind = find(indx == i);
    class = data(ind(:),:);
    plot(class(:,1),class(:,2),'.');
end
plot_gaussians(data,gparams,1,2,[],[],weight,'Kmeans');
%}
%calculate log likelihood
for iter = 1:maxiterations
%calculate new W
    count = count+1;
    for k = 1:K
        Q(:,k) = mvnpdf(data,center(k,:),C(:,:,k));
    end
    alpha = sum(weight,1)/m;
    like = Q * alpha';
    iteration(iter) = iter;
    likelihood(iter) = sum(log(like)); 
    for i = 1:m
        Q(i,:)=Q(i,:) .* alpha;
    end
    P = sum(Q,2);
    for k = 1:K
        newW(:,k) = Q(:,k) ./ P;
    end

%calculate new parameter
    N = sum(newW,1);
    for k = 1:K
        newcenter(k,:) =  newW(:,k)' * data ./ N(k);  
        temp = data-repmat(newcenter(k,:),m,1);
        newc = zeros(n,n);
        for i = 1:m
            newc = newc + newW (i,k)*temp(i,:)' * temp(i,:);
        end
        newcovariance(:,:,k) = newc /N(k);
    end
    if sum(sum(abs(newcenter-center))) < 10 ^ -6
        center = newcenter;
        weight = newW;
        C = newcovariance;
        break;
    end
    center = newcenter;
    weight = newW;
    C = newcovariance;
    
end
like = likelihood(iter);
for k = 1:K
    gparams(k) = struct('mean',center(k,:),'covariance',C(:,:,k));
end
%{
figure,
plot(iteration(1:count),likelihood(1:count));
%}
end

