function [ label,center ] = kmeans(data,K,maxiteration,print,r)
%KMEANS Summary of this function goes here
%   data is a n*d array,n samples with d features.
if nargin < 5
    r=1;
end
[m,n] = size(data);
Max = max(data);
Min = min(data);
flabel = zeros(m,r);
fcenter = zeros(K,n,r);
fssm = zeros(r,1);
for p = 1:r
    center = rand([K,1])*(Max-Min)+repmat(Min,K,1);
    label = zeros(1,m);
    dist = zeros(K,m);
    iter = 0;
    iterations = zeros(maxiteration,1);
    ssm = zeros(maxiteration);
    for i = 1:maxiteration
        for j = 1:m
            for k = 1:K
                v = data(j,:)-center(k,:);
                dist(k,j)= v *v';
            end
        end
        [mindist,label] = min(dist);
        iter = iter + 1;
        ssm(iter) = sum(mindist);
        iterations(iter) = iter;
        for  k = 1:K
            if length(find(label==k))>0
                center(k,:) = mean(data(find(label==k),:));
            end
        end
    end
    label = label';
    flabel(:,p) = label;
    fcenter(:,:,p) = center;
    fssm(p)  = ssm(iter);
end
[~,best] = min(fssm,[],1);
label = flabel(:,best);
center = fcenter(:,:,best);
if print == 1
    figure,plot(iterations,ssm);
end
end

