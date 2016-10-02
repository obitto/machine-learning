function [fc_layer ,fc_bias] = backPropa(conv_layer,fc_layer,fc_bias,predict,label,learn)
%BACKPROPA Summary of this function goes here
%   Detailed explanation goes here
    [k,p,l,w] = size(fc_layer);
    for a = 1:k
        for b = 1:p
            for i = 1:l
               for j = 1:w
                   loss = predict(a)-label(a);
                   grad = learn * loss * predict(a)* (1-predict(a)) * conv_layer(i,j,b);
                   fc_layer(a,b,i,j) = fc_layer(a,b,i,j) - grad;
               end
            end
            fc_bias(a,b) = 0.9 * fc_bias(a,b)- learn * loss * sigmoidgrad(predict(a));
        end
    end
end

