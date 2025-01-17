function pendulum_movie(tsim,xsim,decimation)
l = 0.5;
fa = figure('Position', [700, 200, 400, 400]);
theta = xsim(:,1);
% step = round(numel(tsim)/decimation);
for i = 1 : decimation : length(tsim)
  figure(fa)
  plot([0 -l*sin(theta(i))], [0 +l*cos(theta(i))], '-ob', 'LineWidth', 2) % Nonlinear
  hold on
  plot(-l*sin(theta(i)), +l*cos(theta(i)), '-ob', 'LineWidth', 2, 'MarkerSize', 20, 'MarkerFaceColor', 'b')
  hold off
  axis([-1.1*l, 1.1*l, -1.1*l, 1.1*l])
  title(sprintf('t = %1.4f s',tsim(i)),'FontSize',14)
  pause(1e-2)
end
end