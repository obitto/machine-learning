fc_layer = rand(3000,1000,3)/1000;
fc_size = [1000;10];
fc_para = [m*n*7*7;1000];
w1 = rand(fc_para(1),fc_size(1))/1000;
w2 = rand(fc_para(2),fc_size(2))/1000;
%w3 = rand(fc_para(3),fc_size(3))/1000;
fc_bias = zeros(3,1000);
learn = 0.00001;
lambda = 0.01;
iter = 0;
maxiterations = 4000;
[samples,dim] = size(data);
conv_layer1 = zeros(22,22,m*n);
s = 50;
final_input = zeros(7,7,m*n,s);
error = zeros(maxiterations,1);
while iter < maxiterations
    t = 0;
    for i= 1:s
        image = squeeze(data(i,:));
        image = rot90(fliplr(reshape(image,28,28)));
        for k = 1:m*n
            filter = squeeze(fil_bank(:,:,k));
            conv_layer1(:,:,k) = RELU(conv2(image,rot90(filter,2),'valid'));
        end
        input = max_pool(conv_layer1,22,3);
        nn_input = reshape(input,1,[]);
        %w1 = squeeze(fc_layer(1:fc_para(1),1:fc_size(1),1));
        hidden1 = nn_input*w1+fc_bias(1,1:fc_size(1));
        %w2 = squeeze(fc_layer(1:fc_para(2),1:fc_size(2),2));
        pred = softmax(hidden1*w2+fc_bias(2,1:fc_size(2)));
        %w3 = squeeze(fc_layer(1:fc_para(3),1:fc_size(3),3));
        %pred = softmax(hidden2*w3+fc_bias(3,1:fc_size(3)));
        trainlabel = label(i,:);
        delta = (pred - trainlabel) ;
        %display(pred);
        %display(trainlabel);
        %display(delta);
        %fprintf('error rate : %f \n',delta *delta');
        %t = t+ delta *delta';
        t = t - sum(log(pred) .* trainlabel);
        
        %gradw3 = hidden2' *delta;
        %gradb3 = sum(delta,1);
        %w3= (1-learn*lambda)*w3 - learn*gradw3;
        %fc_bias(3,1:fc_size(3)) = fc_bias(3,1:fc_size(3)) - learn*gradb3;
        %delta = delta*w3' ;
        
        gradw2 = hidden1'*delta;
        gradb2 = sum(delta,1);
        w2= (1-learn*lambda)*w2 - learn*gradw2;
        fc_bias(2,1:fc_size(2)) = fc_bias(2,1:fc_size(2)) - learn*gradb2;
        delta = delta*w2' ;
        
        gradw1 = nn_input'*delta;
        gradb1 = sum(delta,1);
        w1= (1-learn*lambda)*w1 - learn*gradw1;
        fc_bias(1,1:fc_size(1)) = fc_bias(1,1:fc_size(1)) - learn*gradb1;
    end
    iter = iter + 1;
    fprintf('error rate : %f \n',t/s);
    error(iter) = t/s;
end