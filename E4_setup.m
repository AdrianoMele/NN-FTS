%% Example #4
% Pendulum with contracting trajectory domain.

%% MLP parameters
numLayers  = 5; % hidden layers + 2 (in/out)
numNeurons = 32;

%% Training options
% Epochs and minibatch size
numEpochs      = 40;
numMiniBatches = 100;

% Specify ADAM optimization options
initialLearnRate = 0.01;
decayRate        = 0.00001;

% Additional training parameters
options.wVdot     = 5;    % weight on derivative condition
options.wVbound   = 1;    % weight on boundary condition
options.tolVbound = 3e0;  % tolerance on boundary condition
options.tolVdot   = 1e-0; % tolerance on derivative condition
options.wVt       = 0;    % regularization on dV/dt
options.wV        = 0;    % regularization on V

% Collocation points
NPC = 10000;
NPB = 200; % for each time sample
NP0 = 700;

%% Define FTS problem

% Time vector
t = (0:1e-1:3)';

% System
f = @odefun; 

% State dimension
nx = 2;

% Ellipsoidal domains
G = @(t)[2/pi^2 0; 0 0.5/pi^2]*exp(0.4*t);
R = 4/pi^2*eye(2);

plot_trajectories(t,R,G,nx,f)






%% Plant
function xdot = odefun(~,x)
g = 9.81;
m = 0.15;
b = 0.1;
l = 0.5;

xdot = [x(2,:)
        -g/l*sin(x(1,:))-b/m^2*x(2,:)];
end







