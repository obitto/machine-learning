function [ score ] = bicscore(like,k,n, d)
%BICSCORE Summary of this function goes here
%   Detailed explanation goes here
score = like - (k*d+k*d*d)*log(n)/2;

end

