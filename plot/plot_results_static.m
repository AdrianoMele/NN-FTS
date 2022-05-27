% Define accelerated model functions
accfun_grad = dlaccelerate(@modelGradients_static);
accfun_loss = dlaccelerate(@modelLoss_static);

% Define a grid of points
ee(1:2) = eig(R);
ee(3:4) = eig(G(t(1)));

xmax = max(real(1./sqrt(ee)));
xmin = -xmax;
[X1,X2] = meshgrid(linspace(xmin,xmax,30),linspace(xmin,xmax,30));
x1 = reshape(X1,[],1);
x2 = reshape(X2,[],1);
xx = [x1 x2];

% Restrict search to the points inside the (fixed) trajectory domain
idx = false(size(xx,1),1);
for j = 1 : numel(x1)
  idx(j) = xx(j,:)*G(t(1))*xx(j,:)'<1;
end
xx = xx(idx,:);
dlxx = dlarray(xx','CB');

% Verify condition
[~,~,solutionFound] = dlfeval(@modelLoss_static,parameters,dlxx,dlX0,dlXB,f,options);

if not(solutionFound)
  warning('Conditions not satisfied on the test set. Try increasing the number of collocation points.')
end

% Compute V and its derivatives
[V, Vgrad] = dlfeval(accfun_grad,parameters,dlxx);
V  = extractdata(V);
Vx = extractdata(Vgrad{1});

Vdot = V*0;
for j = 1 : size(xx,1)
  Vdot(j) = sum(Vx(:,j).*f([],xx(j,:)'));
end
Vdot = gather(Vdot);

% Plots
Vplot          = x1*0;
VdotPlot       = x1*0;

Vplot(idx)     = V;
VdotPlot(idx)  = Vdot;

Vplot(~idx)    = NaN;
VdotPlot(~idx) = NaN;

h = figure('Position',[180 250 550 450]);
mesh(X1,X2,reshape(Vplot,size(X1,1),size(X1,2)),'FaceColor','flat','FaceAlpha','0.5')
hold on
mesh(X1,X2,reshape(VdotPlot,size(X1,1),size(X1,2)),'FaceColor',[1 0 0],'FaceAlpha','0.5')

plot_ellipse(R,'r','LineWidth',2);
plot_ellipse(G(t(1)),'b','LineWidth',2);

title("Predicted V")
legend({'V','dV/dt','\Omega_0','\Omega_t'})
