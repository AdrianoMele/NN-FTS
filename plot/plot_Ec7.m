addpath ./functions
addpath ./model
addpath ./plot

%% Check bnd condition
dlXB = dlarray(XB','SBCS');
dlTB = dlarray(TB','SBCS');
VB = model_DLTB(network,dlXB,dlTB);

dlX0 = dlarray(X0','SBCS');
dlT0 = dlarray(T0','SBCS');
V0 = model_DLTB(network,dlX0,dlT0);

assert(max(V0)<min(VB),'boundary condition not satisfied')

%% Check derivative condition
dlXC = dlarray(XC','SBCS');
dlTC = dlarray(TC','SBCS');
[V, dV] = dlfeval(@modelGradients_DLTB,network,dlXC,dlTC);
Vx = dV{1};
Vt = dV{2};

Vx = extractdata(squeeze(Vx));
Vt = extractdata(squeeze(Vt));
dlXC = squeeze(dlXC);
dlTC = squeeze(dlTC);

% Lie derivatives
f_x = f(dlTC,dlXC);
g_x = g(dlTC,dlXC);
if isdlarray(f_x), f_x = extractdata(f_x); end
if isdlarray(g_x), g_x = extractdata(g_x); end

Vdot = TC*0;
for i = 1 : numel(TC)
  Vdot(i) = Vt(i) + Vx(:,i)'*f_x(:,i) + ...
    Vx(:,i)'*g_x(:,:,i)*controller_DLTB(network,f,g,TC(i),XC(i,:)',Umax);
end

%% Simulation

% refine time vector
Ts = 1e-2;
t = t(1):Ts:t(end);

x0 = [.01;.02];
x = zeros(nx,numel(t));
x(:,1) = x0;

u1 = t*0;
u2 = t*0;
h = waitbar(0,'progress...');
for it = 1 : numel(t)
  
  v(:,it) = controller_DLTB(network,f,g,t(it),x(:,it),Umax);

  fprintf('Time: %.3f | x(t): %.3f %.3f | Controller: %.5f %.5f \n', t(it),x(:,it),v(:,it))
  u1(it) = sqrt(v(1,it)^2 + v(2,it)^2);
  u2(it) = atan2(v(2,it),v(1,it));
  
  % trajectory
  x(1,it+1) = x(1,it)+Ts*u1(it)*cos(u2(it));
  x(2,it+1) = x(2,it)+Ts*u1(it)*sin(u2(it));
%   x(1,it+1) = x(1,it)+Ts*v(1,it);
%   x(2,it+1) = x(2,it)+Ts*v(2,it);

%   x(:,it+1) = x(:,it) + Ts*(f(t(it),x(:,it)) + g(t(it),x(:,it))*v(:,it));

  waitbar(it/numel(t),h)
end
close(h)
x(:,end) = [];

%% Plot

theta = linspace(0,2*pi,50);
chr = 'rgbcmykr';
figure

subplot(121)
plot(x(1,:),x(2,:)), grid, hold on
k = 1;
for it = [1:round(numel(t)/5):length(t) length(t)]
  xp = xc(t(it));
  plot(xp(1)+rho(t(it))*cos(theta), xp(2)+rho(t(it))*sin(theta), 'g', 'linewidth', 2), hold on
  text(x(1,it),x(2,it),num2str(k))
  plot(x(1,it),x(2,it),[chr(k) '*'], 'linewidth', 2)
  k = k+1;
end
title('Trajectory and FTS sets')
axis equal

subplot(122)
% plot(t,u1,t,u2, 'linewidth', 2)
plot(t,v, 'linewidth', 2)
title('Control action')