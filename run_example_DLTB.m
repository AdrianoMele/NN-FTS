clearvars
close all
clc

addpath ./functions
addpath ./model
addpath ./plot

rng(0) % for repeatability

% Debug options
verbose = 1;

% To train on a GPU if one is available, specify the execution environment "auto". 
executionEnvironment = "cpu";

% for ellipses moving in time; might be overwritten in the setup
xc = [];

% dummy input term for stability analysis
g    = @(x)0;
Umax = 0;

%% Choose example
example = "Ec7"; 

setupscript = example + "_setup.m";
filename    = example + "_res.gif"; % used to save plot

%% setup problem
run(setupscript);

%% prepare solver
preprocess_DLTB;

%% train model
network = train_model_DLTB(network,f,g,Umax, ...
  ds,dlT0,dlX0,dlTB,dlXB,...
  miniBatchSize,numEpochs,executionEnvironment,...
  initialLearnRate,decayRate,options,verbose);

%% Validate and plot results
plot_collocation_points(t,R,G,T0,X0,TB,XB,TC,XC,xc)

plot_results_DLTB;

rmpath ./functions
rmpath ./model
rmpath ./plot


