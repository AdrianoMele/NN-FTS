%% Preprocess collocation points
% Total number of points
NP = NPC+NPB*numel(t)+NP0;

% Minibatch size
miniBatchSize = round(NP/numMiniBatches);

% Place collocation points
[TC,XC] = place_collocation_points_ellipse(t,G,NPC,nx);
[T0,X0] = place_initial_points_ellipse(t,R,NP0,nx);
[TB,XB] = place_boundary_points_ellipse(t,G,NPB,nx);

% Create datastore with collocation points (to be divided in minibatches)
ds = arrayDatastore([TC XC]);

% Convert the initial and boundary conditions to |dlarray|; all the points 
% are used at each training iteration.
% Specify format with dimensions |'SBCS'| (spatial, batch, channel, spatial).
% These format is due to the fact that we are using an imageInputLayer in
% the model.
dlTB = dlarray(TB','CB');
dlXB = dlarray(XB','CB');
dlT0 = dlarray(T0','CB');
dlX0 = dlarray(X0','CB');

%% Initialize deep learning model
numIn   = nx + 1; % state and time
numOut  = 1;      % always one - we are getting V directly from the MLP
parameters = initNetwork(numIn,numOut,numNeurons,numLayers);

