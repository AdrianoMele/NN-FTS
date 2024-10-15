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

%% Choose example
example = "E6"; 

setupscript = example + "_setup.m";
filename    = example + "_res.gif"; % used to save plot

%% setup problem
run(setupscript);

%% prepare solver
preprocess;

%% train model
parameters = train_model(parameters,f, ...
  ds,dlT0,dlX0,dlTB,dlXB,...
  miniBatchSize,numEpochs,executionEnvironment,...
  initialLearnRate,decayRate,options,verbose);

%% Validate and plot results
plot_collocation_points(t,R,G,T0,X0,TB,XB,TC,XC)

plot_results;

rmpath ./functions
rmpath ./model
rmpath ./plot


