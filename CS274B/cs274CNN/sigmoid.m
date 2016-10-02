function [result] = sigmoid( wx )
%SIGMOID Summary of this function goes here
%   Detailed explanation goes here
    [m,n] = size(wx);
    result = zeros(m,n);
    for i = 1:m
        for j = 1:n
            result(i,j) = 1/(1+exp(-wx(i,j)));
        end
    end
end

