function [ data] = loadtest( k )
%LOADTEST Summary of this function goes here
%   Detailed explanation goes here
    data = csvread('test.csv',1,0);
    [m,n] = size(d);
    data = reshape(data,m,28,28);
    data = zero_pad(data,k);
end

