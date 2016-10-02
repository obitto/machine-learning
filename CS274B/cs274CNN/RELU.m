function [ result ] = RELU( data)
%RELU Summary of this function goes here
%   Detailed explanation goes here
    result = data .* (data>=0);

end

