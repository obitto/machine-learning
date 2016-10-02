function [ pooledimage] = mean_pool( input,image_dim,filter_dim)
%MEAN_POOL Summary of this function goes here
%   Detailed explanation goes here
    [h,w,m] = size(input);   
    step = floor(image_dim/filter_dim);
    d = filter_dim;
    pooledimage = zeros(step,step,m);
    for i = 1:m
        for j  = 1:step
            for k = 1:step
                %display(input((j-1)*d+1:j*d,(k-1)*d+1:k*d,i));
                %display(sum(sum(input((j-1)*d+1:j*d,(k-1)*d+1:k*d),i)));
                pooledimage(j,k,i) = sum(sum(sum(input((j-1)*d+1:j*d,(k-1)*d+1:k*d),i)))/(d*d);
            end
        end
    end
end

