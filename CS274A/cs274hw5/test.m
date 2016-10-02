data = load('dataset2.txt');
K=5;
[gparams,memberships,like,bic,k ] = selection( data,K);
figure,hold on;
a1 = plot(x,like,x,bic); 
M1 = 'Likelihood';
M2 = 'BIC';
legend(M1,M2);
hold off;
figure;
hold on;
[~,indx] = max(memberships{k},[],2);
for i = 1:k
    ind = find(indx == i);
    class = data(ind(:),:);
    plot(class(:,1),class(:,2),'.');
end
plot_gaussians(data,gparams{k},1,2,[],[],memberships{k},'random');
hold off;















