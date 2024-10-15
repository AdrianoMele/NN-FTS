%% Example #2
% Simple NLTI system with contracting trajectory domain.

%% MLP parameters
numLayers = 4; % hidden layers + 2 (in/out)
numNeurons = 64; 

%% Define FTS problem

% Time vector
t = (0:5e-2:1)';

% System
f = @(t,x) [-0.2*x(1,:); -0.1*x(1,:).^2 - 1*x(2,:).^3]; % FTS, nonlinear

% State dimension
nx = 2;

% Ellipsoidal domains
G = @(t)0.25*[0.7 0; 0 0.9]*exp(0.6*t);
R = eye(2)*0.3;

%% Training Options
% Epochs and minibatch size
numEpochs      = 40;
numMiniBatches = 100;

% Specify ADAM optimization options
initialLearnRate = 0.01;
decayRate        = 0.00001;

% Additional training parameters
options.wVdot     = 10;
options.wVbound   = 1;
options.wVt       = 0;
options.wV        = 1e-3;
options.tolVbound = 1e1;
options.tolVdot   = 1e-0;

% Collocation points
NPC = 8000;
NPB = 100; % for each time sample
NP0 = 500;

