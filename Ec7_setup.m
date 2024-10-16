%% Example #7
% Controlled system

%% MLP parameters
numLayers  = 5; % hidden layers + 2 (in/out)
numNeurons = 32;

%% Training options
% Epochs and minibatch size
numEpochs      = 40;
numMiniBatches = 500;

% Specify ADAM optimization options
initialLearnRate = 0.01;
decayRate        = 0.00001;

% Additional training parameters
options.wVdot     = 1e0;  % weight on derivative condition
options.wVbound   = 1e0;  % weight on boundary condition
options.tolVdot   = 0;    % tolerance on derivative condition (can be 0 for FTS and should if domains are not centered in the origin)
options.tolVbound = 1e-1;  % tolerance on boundary condition
options.wVt       = 0;    % regularization on dV/dt
options.wV        = 0e-5; % regularization on V

% Collocation points
NPC = 10000;
NPB = 100; % for each time sample
NP0 = 200;

%% Define FTS problem

% Time vector
t = (0:1e-2:3)';

% System
f = @ff; 
g = @gg;

% State dimension
nx = 2;

% maximum control action
Umax = [1;1]*10;

% guiding center
xc = @(t) [2*t + sin(2*pi*t/5);
           2*t + cos(2*pi*t/5) - 1]; 

gamma0 = 0.1;
rho    = @(t) 3*gamma0.*exp(-t/5);

% ellipses
G = @(t)1/rho(t)^2 * eye(nx);
R = 1/gamma0^2     * eye(nx);

% figure
% plot(t,rho(t)), grid
% title('Radius of the neighborhood of the final point')

% theth = linspace(0,2*pi,50);
% figure
% % initial domain
% plot(gamma0*cos(theth),gamma0*sin(theth),'r'), hold on
% % guiding center
% xc_ = xc(t');
% plot(xc_(1,:),xc_(2,:),'b','linewidth',2)
% % trajectory domains
% for i = 1:length(t)  
%   plot(xc_(1,i)+rho(t(i))*cos(theth), xc_(2,i)+rho(t(i))*sin(theth), 'g'), hold on
% end
% grid, axis equal, hold on
% legend('initial condition set', 'FTS sets')



%% Plant

function f_ = ff(t,x)
f_ = x*0;
end

function g_ = gg(t,x) % with v1 = u1*cos(u2) and v2 = u1*sin(u2)
g_ = eye(size(x,1));
g_ = repmat(g_,1,1,numel(t));

if isdlarray(x) 
  g_ = dlarray(g_,'SSB'); 
end
end

% function g_ = gg(t,x)
% g_ = [cos(x(3,:)), 0;
%       sin(x(3,:)), 0;
%       0,           1];
% end

function xdot = odefun(t,x,u)
fprintf('t = %.4fs | u = %.4f | x1 = %.4f | x2 = %.4f \n',t,u(x),x(1),x(2))
xdot = ff(x) + gg(x)*u(x);
end



