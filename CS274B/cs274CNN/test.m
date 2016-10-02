actual = zeros(1000,1);
predict = zeros(1000,1);
conv_layer1 = zeros(22,22,m*n);
for i =1:1000
    %display(i);
    image = squeeze(data(i,:));
    image = rot90(fliplr(reshape(image,28,28)));
    for k = 1:15
        filter = squeeze(fil_bank(:,:,k));
        conv_layer1(:,:,k) = RELU(conv2(image,rot90(filter,2),'valid'));
    end
    pool_layer1 = max_pool(conv_layer1,22,3);
    nn_input = reshape(pool_layer1,1,[]);
    %w1 = squeeze(fc_layer(1:fc_para(1),1:fc_size(1),1));
    %hidden = sigmoid(nn_input*w1+fc_bias(1,1:fc_size(1)));
    hidden1 = nn_input*w1+fc_bias(1,1:fc_size(1));
    %w2 = squeeze(fc_layer(1:fc_para(2),1:fc_size(2),2));
    %hidden2 = hidden1*w2+fc_bias(2,1:fc_size(2));
    %w3 = squeeze(fc_layer(1:fc_para(3),1:fc_size(3),3));
    %output = sigmoid(hidden2*w3+fc_bias(3,1:fc_size(3)));
    pred = softmax(hidden1*w2+fc_bias(2,1:fc_size(2)));
    [~,idx] = max(output);
    %display(output(depth,:,1:10));
    predict(i) = idx-1;
    [~,idx] = max(label(i,:),[],2);
    actual(i) = idx - 1;
end
result = (predict == actual);
error = 1-sum(result)/1000;
display(error);