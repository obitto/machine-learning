function  plot_gaussians(data,params,xaxis,yaxis,ll,iteration,memberships,initstr)
% 
%  MATLAB function to plot a 2 dimensional scatter plot of
%  sample data (using xaxis and yaxis as the column indices into
%  an N x d data matrix) and superpose the means of cluster centers
%  and the k "covariance ellipses"  on this data (the parameter k,
%  the number of clusters, is figured out from the size of the mean
%  matrix)
%
%  Note: this version has been modified so that it can be repeatedly
%  called within an EM algorithm to periodically display the parameters
%  of the mixture model as they are being learned
% 
%  Known bugs: the calculations of the ellipse orientations
%              can "blow up" for matrices with zeroes off the diagonals
%              - should be easy to fix
%
%  INPUTS:
%    data: N x d matrix of d-dimensional feature vectors
%   means: k x d matrix of d-dimensional cluster mean vectors
%  covars: d x (dxk) matrix of dxd covariance matrices: the elements of the
%         jth covariance matrix are between (1,1+(j-1)d) and (d,d+(j-1))
%         (i.e., the matrices are "lined up" from left to right - in
%         the absence of any real data structures in MATLAB)
%  xaxis: an integer between 1 and d indicating which of the features is 
%         to be used as the x axis
%  yaxis: another integer between 1 and d for the y axis
%     ll: an optional argument which is the likelihood of the data
%         given the model.
%  iteration: integer indicating which iteration of EM is being plotted
%  memberships:  an N x K matrix of membership probabilities
%  initstr: a string describing which initialization method was used

xaxis = 1;
yaxis = 2;
colors = ['bgrckmy'];

 xmax = max(data(:,xaxis));
 xmin = min(data(:,xaxis));
 xrange = xmax - xmin;
 ymax = max(data(:,yaxis));
 ymin = min(data(:,yaxis));
 yrange = ymax - ymin;

[tmp clusters] = (max(memberships));
K = size(params,2);
means = [];
for i=1:K
    means = [means; params(i).mean];
end
[k d] = size(means);

 

% Calculate contours for the 2d normals at Mahalanobis dist = constant
mhdist = 3;
for i = 1:k
 indexk = clusters==i;
 icolor = colors(mod(i,7)+1);
 cstr = [icolor,'.'];
 plot(data(indexk,xaxis),data(indexk,yaxis),cstr); 
 hold on;
 plot(means(i,xaxis),means(i,yaxis),'k.','Markersize',20);
% Extract the relevant dimensions from the ith component matrix
%  xi = xaxis + i*d;
%  yi = yaxis + i*d;
%  tmp = [covars(xaxis,xi) covars(xaxis,yi); covars(yaxis,xi) covars(yaxis,yi)];

if isfield(params,'variances')
tmp = [params(i).variances(1) 0; 0 params(i).variances(2)];
else
tmp = params(i).covariance(1:2,1:2);
end
    
% Use some results from standard geometry to figure out the ellipse
% equations from the covariance matrix. Probably other ways to
% do this, e.g., finding the principal component directions, etc.
% See Fraleigh, p.431 for details on rotating the ellipse, etc
 icov = inv(tmp);
 a = icov(1,1);
 c = icov(2,2);
 eps = 0.000001;
 b = icov(1,2)*2 + eps;

 theta = 0.5*acot( (a-c)/b);

 sc = sin(theta)*cos(theta);
 c2 = cos(theta)*cos(theta);
 s2 = sin(theta)*sin(theta);

 a1 = a*c2 + b*sc + c*s2;
 c1 = a*s2 - b*sc + c*c2;

 th= 0:2*pi/100:2*pi;

 x1 = sqrt(mhdist/a1)*cos(th);
 y1 = sqrt(mhdist/c1)*sin(th);
 
 x = x1*cos(theta) - y1*sin(theta) + means(i,xaxis);
 y = x1*sin(theta) + y1*cos(theta) + means(i,yaxis);
% plot the ellipse 
cstr2 = icolor;
 plot(x,y,cstr2,'linewidth',3);
 axis([xmin-xrange/10 xmax+xrange/10 ymin-yrange/10 ymax+yrange/10]);

end

sx = ['Dimension ',num2str(xaxis)];
sy = ['Dimension ',num2str(yaxis)];

xlabel(sx,'fontsize',18,'fontweight','bold');
ylabel(sy,'fontsize',18,'fontweight','bold'); 
str = ['PLOT OF FITTED PARAMETERS AND DATA, ITERATION ',num2str(iteration)];
title(str,'fontsize',18,'fontweight','bold'); 

% this part for printing the likelihood value on the plot has not
% been double-checked, not sure it always puts the value in the
% appropriate location
 if nargin>4
        s= ['Loglikelihood = ',num2str(ll)];
        %s= ['Iteration = ',num2str(ll)];
        xx = xmin + xrange*0.6;
        yy = ymax+0.05*yrange;
        text(xx,yy,s, 'fontsize',18,'fontweight','bold'); 
        
        s2 = ['Initialization:',initstr];
        xx = xmin;
        text(xx,yy,s2, 'fontsize',18,'fontweight','bold'); 
	
 end

% put in a small pause to slow down the plotting to the screen, otherwise
% the plots can change too quickly to be visible to the humann eye
pause(0.1);
hold off;