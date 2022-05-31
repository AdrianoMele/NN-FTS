%% Example #0
% Simple LTI system with constant trajectory domain.

clearvars
close all
clc

addpath ./functions
addpath ./model
addpath ./plot

rng(0) % for repeatability

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


%% Choose collocation points
NPC = 5000;
NPB = 500; % for each time sample
NP0 = 200;

% Total number of points
NP = NPC+NPB*numel(t)+NP0;

% Place collocation points
[TC,XC] = place_collocation_points_ellipse(t,G,NPC,nx);
[T0,X0] = place_initial_points_ellipse(t,R,NP0,nx);
[TB,XB] = place_boundary_points_ellipse(t,G,NPB,nx);

% Create datastore with collocation points (to be divided in minibatches)
ds = arrayDatastore([TC XC]);

% Convert the initial and boundary conditions to |dlarray|; all thee points 
% are used at each training iteration.
% Specify format with dimensions |'CB'| (channel, batch).
dlTB = dlarray(TB','CB');
dlXB = dlarray(XB','CB');

dlT0 = dlarray(T0','CB');
dlX0 = dlarray(X0','CB');


%% Define deep learning model
% Define a multilayer perceptron architecture.

numLayers  = 3; % hidden layers + 2 (in/out)
numNeurons = 128;
numIn      = nx;
numOut     = 1;

parameters = initNetwork(numIn,numOut,numNeurons,numLayers);


%% Specify Training Options

% Epochs and minibatch size
numEpochs = 40;
numMiniBatches = 100;
miniBatchSize = round(NP/numMiniBatches);

% Specify ADAM optimization options.
initialLearnRate = 0.01;
decayRate = 0.00001;

% To train on a GPU if one is available, specify the execution environment "auto". 
executionEnvironment = "auto";

% Additional training parameters
options.wVdot     = 1;
options.wVbound   = 1;
options.wVt       = 0;
options.tolVbound = 1e0;
options.tolVdot   = 1e-1;

% Debug information (0: text, 1: figure)
verbose = 0;

%% Train model
parameters = train_model_static(parameters,f, ...
  ds,dlX0,dlXB,...
  miniBatchSize,numEpochs,executionEnvironment,...
  initialLearnRate,decayRate,options,verbose);

%% Validate and plot results
plot_collocation_points(t,R,G,T0,X0,TB,XB,TC,XC)
plot_results_static;

% Verify matrix condition on computed P
A = -0.1*eye(nx); 
disp('AP + PA'' eigenvalues: ')
disp(num2str(eig(A*P + P*A')))

savefig(h,'E0_res');

rmpath ./functions
rmpath ./model
rmpath ./plot

