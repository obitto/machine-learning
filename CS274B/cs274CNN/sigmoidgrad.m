function [ result ] = sigmoidgrad( x )
%SIGMOIDGRAD Summary of this function goes here
%   Detailed explanation goes here
    [n,d] = size(x);
    result = zeros(n,d);
    for i = 1:n
        for j = 1:d
            xx = x(i,j);
            result(i,j) = xx*(1-xx);
        end
    end

end

