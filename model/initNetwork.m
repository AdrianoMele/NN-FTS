function parameters = initNetwork(numIn,numOut,numNeurons,numLayers)
% parameters = initNetwork(numIn,numOut,numNeurons,numLayers)
% 
% Initialize neural network parameters. The neural network has numIn inputs
% and numOut outputs, numNeurons nodes in each hidden layer and a total
% number of layers of numLayers (no. of hidden layers + 2).

parameters = struct;

sz = [numNeurons numIn];
parameters.fc1.Weights = initializeHe(sz,3);
parameters.fc1.Bias = initializeZeros([numNeurons 1]);

% Initialize the parameters for each of the remaining intermediate fully connect
% operations
sz = [numNeurons numNeurons];
numIn = numNeurons;
for layerNumber=2:numLayers-1
  name = "fc"+layerNumber;
  parameters.(name).Weights = initializeHe(sz,numIn);
  parameters.(name).Bias = initializeZeros([numNeurons 1]);
end

% Initialize last fully connected layer
sz = [numOut numNeurons];
name = "fc"+numLayers;
parameters.(name).Weights = initializeHe(sz,numIn);
parameters.(name).Bias = initializeZeros([numOut 1]);
end

