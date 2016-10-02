alpah = sum(memberships{2},1)/m;
Q = zeros(m,K);
center = gparams{2}.mean;
C = gparams{2}.covariance;
for k = 1:2
    center = gparams{i}.mean;
    C = gparams{i}.covariance;
    Q(:,k) = mvnpdf(data,center,C);
end
alpha = sum(memberships{2},1)/m;
like = Q * alpha';
