clearvars
close all
clc

% softplus derivative
nsp = [imageInputLayer([1,1],'Normalization','none'); softplusLayer()];
nsp = dlnetwork(nsp);
sp  = @(x) log(1+exp(x));
dsp = @(x) exp(x) ./ (1+exp(x));

% check softplus
xv = -5:1e-2:5;
yv = sp(xv);
yv_ = yv*0;
for i = 1 : numel(xv)
  yv_(i) = extractdata(forward(nsp,dlarray(xv(i),'SBCS')));
end
figure(1)
subplot(211)
plot(xv,yv,'ob',xv,yv_,'.')
title('softplus')
ylabel('$s(x)$','Interpreter','latex','fontsize',14)
grid minor
% check softplus derivative
dy1 = diff(yv)./diff(xv);
dy2 = dsp(xv);
subplot(212)
plot(xv(1:end-1),dy1,'ob',xv,dy2,'.')
title('softplus derivative')
ylabel('$\frac{ds(x)}{dx}$','Interpreter','latex','fontsize',14)
grid minor

%% Check network

% generate network: 
% 2 inputs, 1 output, 3 neurons and 2 fully connected layers
nin   = 2;
nout  = 1;
nneurons = 3;
nlayers  = 2;
NN = initNetwork_DLTB(nin,nout,nneurons,nlayers);

% weights and biases
W1 = extractdata(NN.Learnables(1,3).Value{1});
b1 = extractdata(NN.Learnables(2,3).Value{1});

W2 = extractdata(NN.Learnables(3,3).Value{1}); 
b2 = extractdata(NN.Learnables(4,3).Value{1});

% The network computes the following:
%   y = W2*(softplus(W1*x + B1)) + B2
% 
% Its derivative wrt x is
%   dy/dx = W2*dsp(W1*x + B1)*W1

x = rand(nin,1);
dlX = dlarray(x(1),'SBCS');
dlT = dlarray(x(2),'SBCS');

dlY_  = W2*sp(W1*x+b1) + b2;
gdlY_ = W2*diag(dsp(W1*x + b1))*W1;
[dlY, gdlY] = dlfeval(@modelGradients_DLTB,NN,dlX,dlT);

% check
assert(norm(extractdata(dlY)-dlY_)<=0.01*norm(dlY_))
assert(norm(extractdata(gdlY{1})-gdlY_(1))<=0.01*norm(gdlY_(1)))
assert(norm(extractdata(gdlY{2})-gdlY_(2))<=0.01*norm(gdlY_(2)))

disp('All fine.')