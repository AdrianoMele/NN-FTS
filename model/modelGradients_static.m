function [V, gradientsV] = modelGradients_static(parameters,dlX)

% Make predictions with the initial conditions.
V = model_static(parameters,dlX);

% Calculate derivatives with respect to X and T.
gradientsV = dlgradient(sum(V,'all'),{dlX});

end

