%% Preprocess collocation points
% Total number of points
NP = NPC+NPB*numel(t)+NP0;

% Minibatch size
miniBatchSize = round(NP/numMiniBatches);

% Place collocation points
[TC,XC] = place_collocation_points_ellipse(t,G,NPC,nx,xc);
[T0,X0] = place_initial_points_ellipse(t,R,NP0,nx,xc);
[TB,XB] = place_boundary_points_ellipse(t,G,NPB,nx,xc);

% Create datastore with collocation points (to be divided in minibatches)
ds = arrayDatastore([TC XC]);

% Convert the initial and boundary conditions to |dlarray|; all the points 
% are used at each training iteration.
% Specify format with dimensions |'SBCS'| (spatial, batch, channel, spatial).
% These format is due to the fact that we are using an imageInputLayer in
% the model.
dlTB = dlarray(TB','SBCS');
dlXB = dlarray(XB','SBCS');
dlT0 = dlarray(T0','SBCS');
dlX0 = dlarray(X0','SBCS');

%% Initialize deep learning model
numIn   = nx + 1; % state and time
numOut  = 1;      % always one - we are getting V directly from the MLP
network = initNetwork_DLTB(numIn,numOut,numNeurons,numLayers);
% parameters = initNetwork(numIn,numOut,numNeurons,numLayers);

