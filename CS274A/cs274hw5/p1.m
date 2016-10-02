data = load('dataset3.txt');
k=2;
[label,center]= kmeans(data,k,10,1,5);
figure;
hold on ;
for i = 1:k
    ind = find(label == i);
    class = data(ind(:),:);
    plot(class(:,1),class(:,2),'.');
end
plot(center(:,1),center(:,2),'k.','markersize',25);