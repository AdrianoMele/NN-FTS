%% Example #3
% Simple LTI system with expanding trajectory domain.

%% MLP parameters
numLayers  = 4; % hidden layers + 2 (in/out)
numNeurons = 64;

%% Training options
% Epochs and minibatch size
numEpochs      = 40;
numMiniBatches = 100;

% Specify ADAM optimization options
initialLearnRate = 0.01;
decayRate        = 0.00001;

% Additional training parameters
options.wVdot     = 10;   % weight on derivative condition
options.wVbound   = 1;    % weight on boundary condition
options.tolVbound = 1e1;  % tolerance on boundary condition
options.tolVdot   = 1e-1; % tolerance on derivative condition
options.wVt       = 0;    % regularization on dV/dt
options.wV        = 0;    % regularization on V

% Collocation points
NPC = 8000;
NPB = 100; % for each time sample
NP0 = 500;

%% Define FTS problem

% Time vector
t = (0:5e-2:1)';

% System
f = @(t,x) 0.1*x; 

% State dimension
nx = 2;

% Ellipsoidal domains
G = @(t)0.25*eye(2)*exp(-0.2*t);
R = eye(2)*0.3;








