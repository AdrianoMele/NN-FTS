%% Example #1
% Simple LTI system with contracting trajectory domain.

%% MLP parameters
numLayers  = 4; % hidden layers + 2 (in/out)
numNeurons = 16; 

%% Training options
% Epochs and minibatch size
numEpochs      = 40;
numMiniBatches = 50;

% Specify ADAM optimization options
initialLearnRate = 0.05;
decayRate        = 0.00001;

% Additional training parameters
options.wVdot     = 10;   % weight on derivative condition
options.wVbound   = 1;    % weight on boundary condition
options.tolVbound = 3e0;  % tolerance on boundary condition
options.tolVdot   = 1e-1; % tolerance on derivative condition
options.wVt       = 0;    % regularization on dV/dt
options.wV        = 1e-2; % regularization on V

% Collocation points
NPC = 10000;
NPB = 1000;
NP0 = 500;

%% Define FTS problem

% Time vector
t = (0:1e-1:1)';

% State dimension
nx = 2;

% System
f = @(t,x) -0.4*x; % FTS, linear

% Ellipsoidal domains
G = @(t)0.25*eye(2)*exp(0.2*t);
R = eye(2)*0.3;







