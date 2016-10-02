function [ fc_layer,conv_filter1,conv_filter2,fc_bias,conv_bias1,conv_bias2,fc_size,fc_para] = CNN( data,label,maxiter)
%CNN Summary of this function goes here
%   Detailed explanation goes here
    [d,m,n] = size(data);
    conv_filter1 = rand(5,5,10);
    conv_filter2 = rand(5,5,8,16);
    %conv_filter(:,:,1) = zeros(3,3);
    conv_bias1 = zeros(10,1);
    conv_bias2 = zeros(16,1);
    fc_layer = rand(2000,200,2)/100;
    fc_size = [200;10];
    fc_para = [10*12*12;200];
    fc_bias = zeros(2,200);
    learn = 0.01;
    lambda = 0.01;
    for iter = 1:100
        for i = 1:1000
            %forward
            %input = data((i-1)*5+1:i*5,:);
            input = data(i,:);
            %convolve and pooling
            image1 = reshape(input,28,28);
            conv_layer1 = zeros(24,24,10);
            for k = 1:10
                conv_layer1(:,:,k) = conv2(image1,squeeze(conv_filter1(:,:,k)),'valid')+conv_bias1(k);
                %imshow(conv2(image1,rot90(squeeze(conv_filter1(:,:,k)),2),'valid'));
            end
            pool_layer1 = mean_pool(conv_layer1,24,2);
            nn_input = reshape(pool_layer1,1,[]);
            %forward
            w1 = squeeze(fc_layer(1:fc_para(1),1:fc_size(1),1));
            %hidden = sigmoid(input*w1+fc_bias(1,1:fc_size(1)));
            %hidden = sigmoid(nn_input*w1+fc_bias(1,1:fc_size(1)));
            hidden = nn_input*w1+fc_bias(1,1:fc_size(1));
            w2 = squeeze(fc_layer(1:fc_para(2),1:fc_size(2),2));
            pred = softmax(hidden*w2+fc_bias(2,1:fc_size(2)));
            %back
            trainlabel = label(i,:);
            delta = pred - trainlabel;
            %fprintf('iteration %d  cost : %f\n',i,delta*delta');
            %display(pred);
            %display(trainlabel);
            %delta = delta .* sigmoidgrad(pred);
            %display(delta);
            gradw2 = hidden' *delta;
            %display(gradw2);
            gradb2 = sum(delta,1);
            fc_layer(1:fc_para(2),1:fc_size(2),2)= 0.999*fc_layer(1:fc_para(2),1:fc_size(2),2) - 0.01*gradw2;
            fc_bias(2,1:fc_size(2)) = fc_bias(2,1:fc_size(2)) - 0.01*gradb2;
            %delta = (delta*w2') .* sigmoidgrad(hidden);
            delta = (delta*w2');
            gradw1 = nn_input'*delta;
            gradb1 = sum(delta,1);
            fc_layer(1:fc_para(1),1:fc_size(1),1)= 0.999*fc_layer(1:fc_para(1),1:fc_size(1),1) - 0.01*gradw1;
            fc_bias(1,1:fc_size(1)) = fc_bias(1,1:fc_size(1)) - 0.01*gradb1;
            delta = (delta*w1');
            
            %gradient for conv and pool
            delta = pool_grad(delta,conv_layer1,10,2,12,24);
            [delta,gradw,gradb] = conv_grad(delta,conv_filter1,image1);
            conv_filter1 = conv_filter1 - 0.01*gradw;
            conv_bias1 = conv_bias1 - gradb;
        end
    end
    
end

