function [ output ] = foward( input,fc_layer,fc_size,fc_bias )
%FOWARD Summary of this function goes here
%   Detailed explanation goes here
    [in_m,in_d]= size(input);
    [depth,~] = size(fc_size);
    width = max(fc_size);
    output = zeros(depth,in_m,width);
    for i = 1:depth
        for j = 1: in_m
            if i == 1 
                neuron_input = input(j,:);
            else
                neuron_input = reshape(output(i,j,1:fc_size(i-1)),1,[]);
            end
            output(i,j,1:fc_size(i)) = pass(neuron_input,squeeze(fc_layer(:,:,i)),fc_size(i),squeeze(fc_bias(i,:)));
        end
    end
    %display(output(1:10,1,1));
    %[~,idx] = max(output(depth,:,:),[],3);
    %display(idx-1);
end

