function [w ] = softmax( x )
%SOFTMAX Summary of this function goes here
%   Detailed explanation goes here
    %display(max(x));
    %display(max(x));
    [d,n]=size(x);
    m = max(x,[],2);
    w = exp(x-repmat(m,1,n));
    w = w./repmat(sum(w,2),1,n);
end

