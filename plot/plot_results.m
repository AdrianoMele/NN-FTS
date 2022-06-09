% Define accelerated model functions
accfun_grad = dlaccelerate(@modelGradients);
accfun_loss = dlaccelerate(@modelLoss);

% Define new time vector
t = linspace(t(1),t(end),15);

% Define a grid of points based on R and G
ee = zeros(2+numel(t),1);
ee(1:2) = eig(R);
for i = 2 : 2 : 2*numel(t)
  ee(1+i:2+i) = eig(G(t(i/2)));
end

xmax = max(real(1./sqrt(ee)));
xmin = -xmax;
[X1,X2] = meshgrid(linspace(xmin,xmax,50),linspace(xmin,xmax,50));
x1 = reshape(X1,[],1);
x2 = reshape(X2,[],1);

zmax = 0;
zmin = 0;
for i = 1 : numel(t)
  
  % Restrict search to the points inside the (fixed) trajectory domain
  xx = [x1 x2];
  idx = false(size(xx,1),1);
  for j = 1 : numel(x1)
    idx(j) = xx(j,:)*G(t(i))*xx(j,:)'<1;
  end
  xx = xx(idx,:);
  tt = t(i)*ones(size(xx,1),1);
  dlxx = dlarray(xx','CB');
  dltt = dlarray(tt','CB');
  
  % Verify condition
  [~,~,solutionFound] = dlfeval(@modelLoss,parameters,dlxx,dltt,dlX0,dlT0,dlXB,dlTB,f,options);
  
  if not(solutionFound)
    warning('Conditions not satisfied on the test set at t = %.2f s. Try increasing the number of collocation points.',t(i))
  end
  
  % Compute V and its derivatives
  [V, Vgrad] = dlfeval(accfun_grad,parameters,dlxx,dltt);
  V  = extractdata(V);
  Vx = extractdata(Vgrad{1});
  Vt = extractdata(Vgrad{2});
  V = gather(V);
  
  Vdot = V*0;
  for j = 1 : size(xx,1)
    Vdot(j) = Vt(j) + sum(Vx(:,j).*f(tt(j),xx(j,:)'));
  end
  Vdot = gather(Vdot);
  
  % Plots
  Vplot{i}          = x1*0;
  VdotPlot{i}       = x1*0;
  
  Vplot{i}(idx)     = V;
  VdotPlot{i}(idx)  = Vdot;
  
  Vplot{i}(~idx)    = NaN;
  VdotPlot{i}(~idx) = NaN;
  
  zmax = max([zmax,max(V),max(Vdot)]);
  zmin = min([zmin,min(V),min(Vdot)]);
end

%% Plot results
h = figure('Position',[180 250 550 450]);
for i = 1 : numel(t)
  mesh(X1,X2,reshape(Vplot{i},size(X1,1),size(X1,2)),'FaceColor','flat','FaceAlpha','0.5')
  hold on
  mesh(X1,X2,reshape(VdotPlot{i},size(X1,1),size(X1,2)),'FaceColor',[1 0 0],'FaceAlpha','0.5')
  
  plot_ellipse(R,'r','LineWidth',2);
  plot_ellipse(G(t(i)),'b','LineWidth',2);
  
  pr = ellipse(R,100);
  patch(pr(1,:),pr(2,:),'r','FaceAlpha',0.5);
  
  % pg = ellipse(G(t(1)),100);
  % patch(pg(1,:),pg(2,:),'b','FaceAlpha',0.3);
  hold off

  xlabel('x_1')
  ylabel('x_2')
  legend({'V','dV/dt','\Omega_0','\Omega_t',''})
  
  xlim([xmin xmax])
  ylim([xmin xmax])
  zlim([zmin zmax])
  
  frame = getframe(h);
  im = frame2im(frame);
  [im,cm] = rgb2ind(im,256);
  if i == 1
    imwrite(im,cm,filename,'gif', 'Loopcount',inf,'DelayTime',0.1);
  else
    imwrite(im,cm,filename,'gif','WriteMode','append','DelayTime',0.1);
  end
  
  pause(0.00001)
end


%% Plot snapshots

figure('Position',[180 100 1000 600]);
iplot = 1;
for i = [1, round(numel(t)/2), numel(t)]
  
  subplot(1,3,iplot)
  iplot = iplot+1;
  mesh(X1,X2,reshape(Vplot{i},size(X1,1),size(X1,2)),'FaceColor','flat','FaceAlpha','0.5')
%   mesh(X1,X2,reshape(Vplot{i}+abs(zmin),size(X1,1),size(X1,2)),'FaceColor','flat','FaceAlpha','0.5')
  hold on
  mesh(X1,X2,reshape(VdotPlot{i},size(X1,1),size(X1,2)),'FaceColor',[1 0 0],'FaceAlpha','0.5')
  
  plot_ellipse(R,'r','LineWidth',2);
  plot_ellipse(G(t(i)),'b','LineWidth',2);
  
  pr = ellipse(R,100);
  patch(pr(1,:),pr(2,:),'r','FaceAlpha',0.5);
  
  title(['t = ' num2str(t(i)) 's'], 'FontSize', 14)

  xlabel('x_1')
  ylabel('x_2')
  
  xlim([xmin xmax])
  ylim([xmin xmax])
  zlim([zmin zmax])
%   zlim([0.5*zmin zmax+abs(zmin)])
  
%   set(gca,'CameraPosition',[-10.2092 -13.3049 100.2452]) % E5
%   set(gca,'CameraPosition',[-115.5051 -123.3994 85.3582]) % E6
  
end
legend({'V','dV/dt','\Omega_0','\Omega_t',''}, 'FontSize', 14)