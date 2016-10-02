%[data,label] = loadData(0);
%data = data/255;
[fc_layer,conv_filter1,conv_filter2,fc_bias,conv_bias1,conv_bias2,fc_size,fc_para] = CNN(data,label,1);
%[testdata] = loadtest(1);
[d,m,n] = size(data);
pred = zeros(1000,1);
%testlabel = zeros(10000,1);
%display(fc_layer(1,1,:,:));
%pred = predict(data,fc_layer,conv_filter,fc_bias,conv_bias,fc_size);
[depth,~] = size(fc_size);
%figure;
actual = zeros(1000,1);
%figure;

for i =1:1000
    %display(i);
    input = data(i,:);
    image1 = reshape(input,28,28);
    conv_layer1 = zeros(24,24,10);
    for k = 1:10
        conv_layer1(:,:,k) = conv2(image1,rot90(squeeze(conv_filter1(:,:,k)),2),'valid')+conv_bias1(k);
        %subplot(2,5,k);
        %imshow(squeeze(conv_layer1(:,:,k)));
    end
    %pause(2);
    pool_layer1 = mean_pool(conv_layer1,24,2);
    nn_input = reshape(pool_layer1,1,[]);
    w1 = squeeze(fc_layer(1:fc_para(1),1:fc_size(1),1));
    %hidden = sigmoid(nn_input*w1+fc_bias(1,1:fc_size(1)));
    hidden = nn_input*w1+fc_bias(1,1:fc_size(1));
    w2 = squeeze(fc_layer(1:fc_para(2),1:fc_size(2),2));
    output = softmax(hidden*w2+fc_bias(2,1:fc_size(2)));
    %output = sigmoid(hidden*w2+fc_bias(2,1:fc_size(2)));
    [~,idx] = max(output);
    %display(output(depth,:,1:10));
    pred(i) = idx-1;
    [~,idx] = max(label(i,:),[],2);
    actual(i) = idx - 1;
end

result = (pred == actual);
error = 1-sum(result)/1000;
display(error);

