function [ gradientw,gradientb,delta ] = back( input,weight,delta)
%BACK Summary of this function goes here
%   Detailed explanation goes here
    %display(size(input));
    %display(size(delta));
    gradientw = input'*delta;
    %display(gradientw);
    gradientb = sum(delta,1);
    delta = (delta*weight') .* sigmoidgrad(input);
end

