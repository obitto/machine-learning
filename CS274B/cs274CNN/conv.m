function [result ] = conv( image,filter,step,bias )
%CONV Summary of this function goes here
%   Detailed explanation goes here
    [m,n,h] = size(image);
    [p,q,d] = size(filter);
    s = floor((m-p)/step);
    l = floor((n-q)/step);
    result = zeros(s+1,l+1,d);
    for o = 1:d
        for i = 0:s
            for j = 0:l
                a = step * i + p;
                b = step * j + q;
                %display(size(image((a- p + 1):a,(b - q + 1):b)));
                result(i+1,j+1,o) = sum(sum(filter(:,:,o) .* image((a- p + 1):a,(b - q + 1):b)))+bias(o);
            end
        end
    end
end

