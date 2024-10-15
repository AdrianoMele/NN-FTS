function [V, gradientsV] = modelGradients_DLTB(network,dlX,dlT)

% Make predictions with the initial conditions.
V = model_DLTB(network,dlX,dlT);

% Calculate derivatives with respect to X and T.
gradientsV = dlgradient(sum(V,'all'),{dlX,dlT},'EnableHigherDerivatives',true);

end


