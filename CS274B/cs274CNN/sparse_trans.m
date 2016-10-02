function [z,ibu] = sparse_trans( shift,ibu)
%SPARSE Summary of this function goes here
%   Detailed explanation goes here
    [m,n] = size(shift);
    a = 0.015;
    b = 1.5;
    z = zeros(m,n);
    for i = 1:n
        ibu(i) = exp(b*shift(i)) + (1-a)*ibu(i)/a;
        z(i) = exp(b*shift(i))/ibu(i);
    end
end

