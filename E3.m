%% Example #3
% Simple LTI system with expanding trajectory domain.

clearvars
close all
clc

addpath ./functions
addpath ./model
addpath ./plot

rng(0) % for repeatability

%% Define FTS problem

% Time vector
t = (0:5e-2:1)';

% System
f = @(t,x) 0.1*x; % FTS, linear

% State dimension
nx = 2;

% Ellipsoidal domains
G = @(t)0.25*eye(2)*exp(-0.2*t);
R = eye(2)*0.3;


%% Choose collocation points
NPC = 8000;
NPB = 100; % for each time sample
NP0 = 500;

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
numLayers = 4; % hidden layers + 2 (in/out)
numNeurons = 64; 
numIn  = nx + 1; % state and time
numOut = 1;

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
options.wVdot     = 10;
options.wVbound   = 1;
options.wVt       = 0;
options.tolVbound = 10e0;
options.tolVdot   = 1e-1;

%% Train model
parameters = train_model(parameters,f, ...
  ds,dlT0,dlX0,dlTB,dlXB,...
  miniBatchSize,numEpochs,executionEnvironment,...
  initialLearnRate,decayRate,options,1);

%% Validate and plot results
plot_collocation_points(t,R,G,T0,X0,TB,XB,TC,XC,nx,f)

filename = 'E3_res.gif';
plot_results;

rmpath ./functions
rmpath ./model
rmpath ./plot


