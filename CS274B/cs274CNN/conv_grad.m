function [newdelta,gradw,gradb] = conv_grad( delta,filter,input )
%CONV_GRAD Summary of this function goes here
%   Detailed explanation goes here
    [h,w,d] = size(input);
    [~,n,m] = size(filter);
    newdelta = zeros(h,w,m,d);
    gradw = zeros(n,n,m);
    gradb = zeros(m,1);
    for j = 1:m
        weight = squeeze(filter(:,:,j));
        for i = 1:d
            error = squeeze(delta(:,:,j,i));
            %display(size(error));
            %display(size(squeeze(input(:,:,d))));
            %display(conv2(squeeze(input(:,:,d)),rot90(error,2),'valid'));
            %display(error);
            %display(input(:,:,d));
            %display(error);
            gradw(:,:,j) = gradw(:,:,j)+conv2(squeeze(input(:,:,d)),rot90(error,2),'valid');
            %display(squeeze(gradw(:,:,j)));
            newdelta(:,:,j,i) = conv2(error,weight,'full');
        end
    end
    
    %display(error(1:10,1:10));
end

