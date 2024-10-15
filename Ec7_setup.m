%% Example #7
% Controlled system

%% MLP parameters
numLayers  = 5; % hidden layers + 2 (in/out)
numNeurons = 32;

%% Training options
% Epochs and minibatch size
numEpochs      = 40;
numMiniBatches = 200;

% Specify ADAM optimization options
initialLearnRate = 0.008;
decayRate        = 0.00001;

% Additional training parameters
options.wVdot     = 5;    % weight on derivative condition
options.wVbound   = 1;    % weight on boundary condition
options.tolVbound = 3e0;  % tolerance on boundary condition
options.tolVdot   = 1e0;  % tolerance on derivative condition
options.wVt       = 0;    % regularization on dV/dt
options.wV        = 0;    % regularization on V

% Collocation points
NPC = 50000;
NPB = 50; % for each time sample
NP0 = 700;

%% Define FTS problem

% Time vector
t = (0:5e-2:5)';

% System
f = @ff; 
g = @gg;

% State dimension
nx = 2;

% guiding center
xc = @(t) [2*t + sin(2*pi*t/5);
           2*t + cos(2*pi*t/5) - 1]; 

gamma0 = 0.1;
rho    = @(t) 3*gamma0.*exp(-t/5);

% ellipses
G = @(t)1/rho(t)^2 * eye(nx);
R = 1/gamma0^2     * eye(nx);


figure
plot(t,rho(t)), grid
title('Radius of the neighborhood of the final point')


theth = linspace(0,2*pi,50);
figure
% initial domain
plot(gamma0*cos(theth),gamma0*sin(theth),'r'), hold on
% guiding center
xc_ = xc(t');
plot(xc_(1,:),xc_(2,:),'b','linewidth',2)
% trajectory domains
for i = 1:length(t)  
  plot(xc_(1,i)+rho(t(i))*cos(theth), xc_(2,i)+rho(t(i))*sin(theth), 'g'), hold on
end
grid, axis equal, hold on
legend('initial condition set', 'FTS sets')









% plot_trajectories(t,R,G,nx,f)






%% Plant

function f_ = ff(x)
f_ = x*0;
end

function g_ = gg(x)
g_ = [cos(x(3,:)), 0;
      sin(x(3,:)), 0;
      0,           1];
end

function xdot = odefun(t,x,u)
fprintf('t = %.4fs | u = %.4f | x1 = %.4f | x2 = %.4f \n',t,u(x),x(1),x(2))
xdot = ff(x) + gg(x)*u(x);
end


