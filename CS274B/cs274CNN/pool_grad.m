function [newdelta] = pool_grad( delta,weight,filter_num,pool_dim,pool_size,img_dim)
%POOL_GRAD Summary of this function goes here
%   Detailed explanation goes here
    newdelta = zeros(img_dim,img_dim,filter_num);
    %display(size(weight));
    %display(size(delta));
    %display(size(weight));
    %display(size(error));
    error = reshape(delta,pool_size,pool_size,filter_num);
    %display(size(error));
    for i = 1:filter_num
        e = squeeze(error(:,:,i));
        newdelta(:,:,i) = (1/pool_dim^2) * kron(e,ones(pool_dim));
    end
end

