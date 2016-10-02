%{
final_input = zeros(20000,11*11*m*n);
conv_layer1 = zeros(22,22,m*n);
for i = 1:20000
    image = squeeze(data(i,:));
    image = rot90(fliplr(reshape(image,28,28)));
    for k = 1:m*n
        filter = squeeze(fil_bank(:,:,k));
        conv_layer1(:,:,k) = RELU(conv2(image,rot90(filter,2),'valid'));
    end
    final = max_pool(conv_layer1,22,2);
    final_input(i,:) = reshape(final,1,[]);
end
%}
hiddenLayerSize = 500;
net = fitnet(hiddenLayerSize,10);