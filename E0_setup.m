%% Example #0
% Simple LTI system with constant trajectory domain.

%% MLP parameters
numLayers  = 3; % hidden layers + 2 (in/out)
numNeurons = 128;

%% Training options
% Epochs and minibatch size
numEpochs      = 40;
numMiniBatches = 100;

% Specify ADAM optimization options
initialLearnRate = 0.05;
decayRate        = 0.00001;

% Additional training parameters
options.wVdot     = 1;    % weight on derivative condition
options.wVbound   = 1;    % weight on boundary condition
options.tolVbound = 1e0;  % tolerance on boundary condition
options.tolVdot   = 1e-1; % tolerance on derivative condition
options.wVt       = 0;    % regularization on dV/dt
options.wV        = 0;    % regularization on V

% Collocation points
NPC = 5000;
NPB = 500; % for each time sample
NP0 = 200;

%% Define FTS problem

% Time vector
t = (0:1e-1:1)';

% System
f = @(t,x) -0.1*x; % FTS, linear

% State dimension
nx = 2;

% Ellipsoidal domains
G = @(t)0.25*eye(nx);
R = 0.3*eye(nx);








