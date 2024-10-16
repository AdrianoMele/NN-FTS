addpath ./functions
addpath ./model
addpath ./plot


%% Check condition
dlXB = dlarray(XB','SBCS');
dlTB = dlarray(TB','SBCS');
VB = model_DLTB(network,dlXB,dlTB);

dlX0 = dlarray(X0','SBCS');
dlT0 = dlarray(T0','SBCS');
V0 = model_DLTB(network,dlX0,dlT0);

assert(max(V0)<min(VB),'boundary condition not satisfied')

%% Simulation

% refine time vector
Ts = 1e-3;
t = t(1):Ts:t(end);

x0 = [.01;.02];
x(:,1) = x0;
Ts = t(2)-t(1);

u1 = t*0;
u2 = t*0;
h = waitbar(0,'progress...');
for it = 1 : numel(t)
  
  v(:,it) = controller_DLTB(network,f,g,t(it),x(:,it),Umax);
  u1(it) = sqrt(v(1,it)^2 + v(2,it)^2);
  u2(it) = atan2(v(2,it),v(1,it));
  
  % trajectory
%   x(1,it+1) = x(1,it)+Ts*u1(it)*cos(u2(it));
%   x(2,it+1) = x(2,it)+Ts*u1(it)*sin(u2(it));
  x(1,it+1) = x(1,it)+Ts*v(1,it);
  x(2,it+1) = x(2,it)+Ts*v(2,it);

  waitbar(it/numel(t),h)
end
x(:,end) = [];

%% Plot

theta = linspace(0,2*pi,50);
chr = 'rgbcmykr';
figure

subplot(121)
plot(x(1,:),x(2,:)), grid, hold on
k = 1;
for it = [1:1500:length(t) length(t)]
  xp = xc(t(it));
  plot(xp(1)+rho(t(it))*cos(theta), xp(2)+rho(t(it))*sin(theta), 'g', 'linewidth', 2), hold on
  text(x(1,it),x(2,it),num2str(k))
  plot(x(1,it),x(2,it),[chr(k) '*'], 'linewidth', 2)
  k = k+1;
end
title('Trajectory and FTS sets')
axis equal

subplot(122)
plot(t,u1,t,u2, 'linewidth', 2)
title('Control action')