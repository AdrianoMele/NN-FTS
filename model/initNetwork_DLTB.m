function network = initNetwork_DLTB(numIn,numOut,numNeurons,numLayers)
% network = initNetwork(numIn,numOut,numNeurons,numLayers)
% 
% Initialize neural network parameters. The neural network has numIn inputs
% and numOut outputs, numNeurons nodes in each hidden layer and a total
% number of layers of numLayers (no. of hidden layers + 2).

% Generate network
layers = [ ...
            imageInputLayer([numIn 1],'Normalization','none')
            fullyConnectedLayer(numNeurons) ];
for i = 2 : numLayers-1
  layers = [layers 
            softplusLayer
            fullyConnectedLayer(numNeurons) ]; %#ok<AGROW> 
end
layers = [layers
          fullyConnectedLayer(numOut) ]; % last fully connected layer
network = dlnetwork(layers);




% network = struct;
% 
% sz = [numNeurons numIn];
% network.fc1.Weights = initializeHe(sz,3);
% network.fc1.Bias = initializeZeros([numNeurons 1]);
% 
% % Initialize the parameters for each of the remaining intermediate fully connect
% % operations
% sz = [numNeurons numNeurons];
% numIn = numNeurons;
% for layerNumber=2:numLayers-1
%   name = "fc"+layerNumber;
%   network.(name).Weights = initializeHe(sz,numIn);
%   network.(name).Bias = initializeZeros([numNeurons 1]);
% end
% 
% % Initialize last fully connected layer
% sz = [numOut numNeurons];
% name = "fc"+numLayers;
% network.(name).Weights = initializeHe(sz,numIn);
% network.(name).Bias = initializeZeros([numOut 1]);
% end

