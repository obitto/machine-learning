function [data,label] = loadData(k)
%   Detailed explanation goes here
    d = csvread('train.csv',1,0);
    [m,n] = size(d);
    data = d(:,2:n);
    %data = reshape(data,m,28,28);
    %data = zero_pad(data,k);
    l = d(:,1);
    label = zeros(m,10);
    for i = 1:m
        label(i,l(i)+1) = 1;
    end      
end

