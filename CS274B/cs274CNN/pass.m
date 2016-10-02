function [ output ] = pass( input,fc_layer,fc_size,fc_bias )
%PASS Summary of this function goes here
%   Detailed explanation goes here
    [~,d] = size(input);
    %display(d);
    output = zeros(fc_size,1);
    %display(size(fc_layer));
    for i = 1:fc_size
        score = input * fc_layer(1:d,i)+fc_bias(i);
        %display(score);
        output(i) = sigmoid(score);
        %display(output(i));
        %output(i) = score;
    end
    %display(output);
    %output = softmax(score);
end

