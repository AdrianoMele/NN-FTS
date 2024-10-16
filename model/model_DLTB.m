function dlV = model_DLTB(network,dlX,dlT)

% Concatenate SBCS dlarrays along first dimension
dlXT = cat(1,dlX,dlT);

% Get model prediction
dlV = forward(network,dlXT);
