function [ label ] = predict( input,fc_layer,conv_filter,fc_bias,conv_bias,fc_size)
%PREDICT Summary of this function goes here
%   Detailed explanation goes here
   [d,m,n] = size(input);
   [depth,~] = size(fc_size);
   label = zeros(d,1);
   for i =1:d
       image = reshape(input(i,:,:),m,n);
       conv_layer = conv(image,conv_filter,1,conv_bias);
       conv_layer = reshape(conv_layer,1,[]);
       output = foward(conv_layer,fc_layer,fc_size,fc_bias);
       [~,idx] = max(output(depth,:,:),[],3);
       label(i) = idx-1;
   end
end

