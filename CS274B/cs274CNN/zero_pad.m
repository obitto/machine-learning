function [ data] = zero_pad3(data,k)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    [l,m,n] = size(data);
    temp = zeros(l,m+2*k,n+2*k);
    temp(:,(k+1):(m+k),(k+1):(n+k)) = data;
    data = temp;
end

