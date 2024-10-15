function dlV = model_DLTB(network,dlX,dlT)

% Concatenate SBCS dlarrays along first dimension
dlXT = cat(1,dlX,dlT);

% Get model prediction
dlV = forward(network,dlXT);

% dlXT = [dlX;dlT];
% numLayers = numel(fieldnames(parameters));
% 
% % First fully connect operation
% weights = parameters.fc1.Weights;
% bias = parameters.fc1.Bias;
% dlV = fullyconnect(dlXT,weights,bias);
% 
% % tanh/relu and fully connect operations for remaining layers
% for i = 2:numLayers
%   name = "fc" + i;
%   dlV = log(1+exp(dlV)); % Softplus layer
%   weights = parameters.(name).Weights;
%   bias = parameters.(name).Bias;
%   dlV = fullyconnect(dlV, weights, bias);
% end
% end