function [V, gradientsV] = modelGradients(parameters,dlX,dlT)

% Make predictions with the initial conditions.
V = model(parameters,dlX,dlT);

% Calculate derivatives with respect to X and T.
gradientsV = dlgradient(sum(V,'all'),{dlX,dlT});

end


