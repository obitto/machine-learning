function [ output ] = reconstruct( filter_bank,shift,feature )
%RECONSTRUCT Summary of this function goes here
%   Detailed explanation goes here
    [n,n,d] = size(filter_bank);
    [h,w,~] = size(feature);
    output = zeros(h+n-1,w+n-1);
    for i =1:d
        output = output + conv2(feature(:,:,i),squeeze(filter_bank(:,:,i)),'full').*shift(i);
    end
end

