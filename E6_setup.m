%% Example #6
% Lakshmikantham (1.1.5) with contracting trajectory domain.

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
NPB = 200; % for each time sample
NP0 = 700;

%% Define FTS problem

% Time vector
t = (0:1e-1:1)';

% System
f = @odefun; 

% State dimension
nx = 2;

% Ellipsoidal domains
r0 = 1; k = 0.1;
u = @(t) r0^2 + (1/k - r0)*exp(2*t);
G = @(t) (k*u(t))*0.8*eye(nx);
% G = @(t)0.8*eye(2)*exp(1.8*t);
R = eye(nx);

plot_trajectories(t,R,G,nx,f)






%% Plant
function xdot = odefun(~,x)
k = 0.1;

xdot = x*0;
xx = x(1,:);
yy = x(2,:);
xdot(1,:) = -xx-yy + k*(xx-yy).*(xx.^2+yy.^2);
xdot(2,:) = +xx-yy + k*(xx+yy).*(xx.^2+yy.^2);

end






