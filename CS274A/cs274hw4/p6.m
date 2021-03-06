a = 1;
b = 3;
lambda = 1;
x = 0:0.001:5;
p1= 0.9;
p2 = 0.1;
f1 = uniform(a,b,x(:))*p1;
f2 = lambda * exp(-lambda * x)*p2;
f1 = f1';
boundary = f1>f2;
figure,plot(x,f1);
hold on;
plot(x,f2');
plot(x,boundary);
f = @(x,lambda,p) lambda * exp(-lambda * x)*p;
e1 = integral(@(x)f(x,lambda,p2),0.695,2);
e= e1 + (1/(b-a))*(0.695);