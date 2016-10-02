function [ density] = uniform( a,b,x )
%UNIFORM Summary of this function goes here
%   Detailed explanation goes here
temp1 = x<=b;
temp2 = x>=a;
temp = (temp1 == temp2);
density = temp*(1/(b-a));
end

