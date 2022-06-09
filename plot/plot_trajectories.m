function plot_trajectories(t,R,G,nx,f)

figure('Position', [500 300 400 450])
plot_ellipse3D(t(1),R,'Color',[0.2 0.8 0.2],'Linewidth',2);
hold on
pr = ellipse(R,20);
fill3(t(1)*ones(1,20),pr(1,:),pr(2,:),'g','FaceAlpha',0.5);

for it = 1:numel(t)
  plot_ellipse3D(t(it),G(t(it)),'b','Linewidth',2);
end

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




end

