function plot_collocation_points(t,R,G,T0,X0,TB,XB,TC,XC,nx,f)

% Plot collocation points and time-varying domains in 3D
figure
plot_ellipse3D(t(1),R,'r');
hold on
plot3(T0,X0(:,1),X0(:,2),'.g');

for it = 1:numel(t)
  plot_ellipse3D(t(it),G(t(it)),'b');
end

plot3(TB,XB(:,1),XB(:,2),'.')
scatter3(TC,XC(:,1),XC(:,2),'MarkerFaceColor','b','SizeData',2,'MarkerFaceAlpha',.2);

xlabel('t');
ylabel('x_1');
zlabel('x_2');

% Plot a bunch of initial trajectories
NX0 = 100;
n = 0;
while n < NX0
  x0 = 5 * 1/min(eig(R)) * 2 * (rand(nx,1)-0.5);
  if x0'*R*x0<=1
    [tsim,xsim] = ode45(f,t,x0);
    plot3(tsim,xsim(:,1),xsim(:,2),'r', 'LineWidth', 2)
    n = n+1;
  end
end

NP = numel([TC;TB;T0]);
title("n. of points: " + NP)