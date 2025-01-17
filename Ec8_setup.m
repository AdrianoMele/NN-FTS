%% Example #7
% Controlled system

%% MLP parameters
numLayers  = 3; % hidden layers + 2 (in/out)
numNeurons = 64;

%% Training options
% Epochs and minibatch size
numEpochs      = 100;
numMiniBatches = 100;

% Specify ADAM optimization options
initialLearnRate = 0.001;
decayRate        = 0.00001;

% Additional training parameters
options.wVdot     = 1e-3;  % weight on derivative condition
options.wVbound   = 1e-2;  % weight on boundary condition
options.tolVdot   = 1e-4;    % tolerance on derivative condition (can be 0 for FTS and should if domains are not centered in the origin)
options.tolVbound = 1e-4;  % tolerance on boundary condition
options.wVt       = 0;    % regularization on dV/dt
options.wV        = 1e-6; % regularization on V

% Collocation points
NPC = 20000;
NPB = 100; % for each time sample
NP0 = 500;
nt  = 30;

%% Define FTS problem
T = 1;

% Time vector
t = linspace(0,T,nt)';

% System
f = @ff; 
g = @gg;

% State dimension
nx = 2;

% maximum control action
Umax = 5;

% guiding center
xc = @(t) [0;0]; 

% ellipses
% gamma0 = 0.1;
% rho    = @(t) 3*gamma0.*exp(-t/5);
% 
% G = @(t)1/rho(t)^2 * eye(nx);
% R = 1/gamma0^2     * eye(nx);

xlimit = [pi,1.5*pi];
G = @(t)diag(1./(xlimit*exp(-2*0*t)).^2);
R = diag(1./(0.9*xlimit).^2);

% plot_ellipse(R,[],'r');
% hold on
% for i = 1 : nt
%   plot_ellipse(G(t(i)),xc(t(i)),'b');
%   drawnow
% end

%% Simulate system with 0 control
% dt = 1e-3;
% tsim = (0:dt:5); % small dt for simulation purposes
% x0 = 0.5*xlim;
% % % myctrl = @(x)(max(min(-x(1),Umax)-Umax))*0;
% % myctrl = @(x)max(min(-2*x(1,:)-0.5*x(2,:),Umax),-Umax);
% % [tsim,xsim] = ode45(@(t,x)of(t,x,myctrl), tsim, x0);
% [tsim,xsim] = ode45(@(t,x)odefun(t,x,@(x)0), tsim, x0);
% h = figure('Position',[250 300 1000 470]);
% % subplot(211)
% plot(tsim,xsim,'linewidth',2);
% % subplot(212)
% % plot(tsim,myctrl(xsim'),'linewidth',2)
% legend({'$\theta$','$\dot{\theta}$'},'Interpreter','Latex')

% pendulum_movie(tsim,xsim,30)

%-------------------------------------------------------------------------%
%                                                                         %
%                           Local functions                               %
%                                                                         %
% ------------------------------------------------------------------------%
function f_ = ff(t,x)
g = 9.81;
m = 0.15;
l = 0.5;
b = 0.1;
f_ = [x(2,:);
  g/l*sin(x(1,:))-b/(m*l^2)*x(2,:)];

if isdlarray(x), f_ = dlarray(f_,'SBCS'); end
end

function g_ = gg(t,x)
m = 0.15;
l = 0.5;
g_ = [zeros(1,size(x,2)); ones(1,size(x,2))/(m*l^2)];

if isdlarray(x) 
  g_ = dlarray(g_,'SSB'); 
end
end

function xdot = odefun(t,x,u)
fprintf('t = %.4fs | u = %.4f | x1 = %.4f | x2 = %.4f \n',t,u(x),x(1),x(2))
xdot = ff(t,x) + gg(t,x)*u(x);
end

function check_ctrb(m,l,b)
% Check controllability around 0
g = 9.81;
A = [0 1; g/l, b/m/l^2];
B = [0; 1/m/l^2];
disp(rank(ctrb(A,B)));
end



