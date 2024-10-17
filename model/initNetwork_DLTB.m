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
          softplusLayer
          fullyConnectedLayer(numOut) ]; % last fully connected layer
network = dlnetwork(layers);

% % use legacy initialization to compare results
% iw = find(network.Learnables(:,2).Parameter=="Weights");
% for i = 1 : numLayers
%   network.Learnables(iw(i),3).Value{1} = initializeHe(size(network.Learnables(iw(i),3).Value{1}),numIn);
% end


