function [ predict] = pre(conv_layer,fc_layer,fc_bias)
%PRE Summary of this function goes here
%   Detailed explanation goes here
    [m,n,d] = size(conv_layer);
    [k,p,l,w] = size(fc_layer);
    predict = zeros(1,k);
    for i = 1:k
        score = 0;
        for j = 1:p
            %display(size(squeeze(conv_layer(:,:,j))));
            %display(size(squeeze(fc_layer(i,j,:,:))));
            score = score + sum(sum(sum(squeeze(conv_layer(:,:,j)) .* squeeze(fc_layer(i,j,:,:)))));
        end
        %display(sigmoid(score+fc_bias(i,j)));
        predict(i) = sigmoid(score+fc_bias(i,j));
    end
    %display(size(predict));
end

