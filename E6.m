%% Example #6
% Lakshmikantham (1.1.5) with contracting trajectory domain.

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
f = @odefun; % FTS, linear

% State dimension
nx = 2;

% Ellipsoidal domains
r0 = 1; k = 0.1;
u = @(t) r0^2 + (1/k - r0)*exp(2*t);
G = @(t) (k*u(t))*0.8*eye(nx);
% G = @(t)0.8*eye(2)*exp(1.8*t);
R = eye(nx);

% plot_trajectories(t,R,G,nx,f)
%% Choose collocation points
NPC = 50000;
NPB = 200; % for each time sample
NP0 = 700;

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
numLayers = 5; % # hidden layers + 2 (in/out)
numNeurons = 32;
numIn  = nx + 1; % state and time
numOut = 1;

parameters = initNetwork(numIn,numOut,numNeurons,numLayers);


%% Specify Training Options

% Epochs and minibatch size
numEpochs = 40;
numMiniBatches = 200;
miniBatchSize = round(NP/numMiniBatches);

% Specify ADAM optimization options.
initialLearnRate = 0.003;
decayRate = 0.00001;

% To train on a GPU if one is available, specify the execution environment "auto". 
executionEnvironment = "auto";

% Additional training parameters
options.wVdot     = 5;
options.wVbound   = 1;
options.wVt       = 0;
options.tolVbound = 3e0;
options.tolVdot   = 1e0;

% Debug options
verbose = 0;

%% Train model
parameters = train_model(parameters,f, ...
  ds,dlT0,dlX0,dlTB,dlXB,...
  miniBatchSize,numEpochs,executionEnvironment,...
  initialLearnRate,decayRate,options,verbose);

%% Validate and plot results
plot_collocation_points(t,R,G,T0,X0,TB,XB,TC,XC)

filename = 'E6_res.gif';
plot_results;

rmpath ./functions
rmpath ./model
rmpath ./plot



return



%% Plant
function xdot = odefun(~,x)
k = 0.1;

xdot = x*0;
xx = x(1,:);
yy = x(2,:);
xdot(1,:) = -xx-yy + k*(xx-yy).*(xx.^2+yy.^2);
xdot(2,:) = +xx-yy + k*(xx+yy).*(xx.^2+yy.^2);

end
