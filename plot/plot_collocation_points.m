function plot_collocation_points(t,R,G,T0,X0,TB,XB,TC,XC,xc)
% Plot collocation points and time-varying domains in 3D

if nargin<10,   xc = @(t)[0;0]; end
if isempty(xc), xc = @(t)[0;0]; end

figure
plot_ellipse3D(t(1),R,xc(0),'g');
hold on
plot3(T0,X0(:,1),X0(:,2),'.','Color',[0.2 0.8 0.2],'MarkerSize',15);

for it = 1:numel(t)
  plot_ellipse3D(t(it),G(t(it)),xc(t(it)),'b');
end

plot3(TB,XB(:,1),XB(:,2),'.','Color',[0.5 0.2 0.8],'MarkerSize',5)
scatter3(TC,XC(:,1),XC(:,2),'MarkerEdgeColor','r','MarkerFaceColor','r','SizeData',6,'MarkerFaceAlpha',.2);

xlabel('t');
ylabel('x_1');
zlabel('x_2');

NP = numel([TC;TB;T0]);
title("total n. of points: " + NP)