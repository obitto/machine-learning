%[data,label] = loadData(0);
%data = data/255;
m = 1;
n = 15;
d = 7;
fil_bank = rand(d,d,m*n)*0.1-0.05;
figure;
ibu = ones(m*n,1)/2;
for it = 1:5
    for i = 1:2000
        input = squeeze(data(i,:));
        input = rot90(fliplr(reshape(input,28,28)));
        subplot(4,n,1),imshow(mat2gray(input));
        [out,fil_bank] = pretrain(input,fil_bank,ibu);
        for k = 1:m
            for j = 1:n
                subplot(4,n,(k-1)*n+j+3*n),imshow(mat2gray(squeeze(fil_bank(:,:,(k-1)*n+j))));
            end
        end
        pause(0.01);
    end
end