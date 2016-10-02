function [ gparams,memberships,like,bic,bestk ] = selection( data,K)
%SELECTION Summary of this function goes here
%   Detailed explanation goes here
like = zeros(K,1);
bic = zeros(K,1);
[m,n] = size(data);
gparams = cell(K,1);
memberships = cell(K,1);
x = 1:1:K;
%selection goes here
for k = 1:K
    [gparams{k},memberships{k},like(k)] = EM(data,k,100);
    %bic(k) = like(k)- (k*n+k*n*n)*log(m)/2;
    bic(k) = bicscore(like(k),k,m,n);
end
[~,bestk] = max(bic);

end

