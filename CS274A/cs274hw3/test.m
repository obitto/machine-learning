data = load('binary_features.txt'); 
y = load('labels.txt');
[n d] = size(data);
%[w1 accuracy1]= logistic_train(data,y);
epsilon = 1e-5;
maxiteration = 1000;
data1 = data(1:2000,:);
testrow = [10,20,50,100,200,500,1000,2000];
accuracy = zeros(length(testrow),1);
data2 = data(2001:4601,:);
data2 = [ones(2601,1) data2];
Y2 = y(2001:4601,:);
%{
for i =1:length(testrow)
    X = data1(1:testrow(i),:);
    Y = y(1:testrow(i),:);
    w = logistic_train(X,Y);
    phat =sigmoid(data2*w);
    chat = phat>0.5;
    accuracy(i) = 100*sum(Y2 == chat)/length(Y2);
end
%}
[w2 accuracy2] = logistic_train(data,y,epsilon,maxiteration,1,10);