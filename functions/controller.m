function u = controller(parameters,f,g,x,Umax,delta)
% u = controller(parameters,f,g,x,Umax)
%   Computes the controller u according to Sontag's formula based on the
%   neural Lyapunov function represented by the NN with the specified
%   parameters.

arguments
  parameters
  f
  g
  x
  Umax
  delta (1,1) {mustBeNumeric} = 1e-5;
end

if isdlarray(x)
  dlX = x;
else
  dlX = dlarray(x,'CB');
end

[V,Vx] = dlfeval(@modelGradients_static,parameters,dlX,delta);

Vx = extractdata(Vx);
LfV = double(sum(Vx.*f(x)));
LgV = double(sum(Vx.*g(x)));

% Sontag #1
if size(LgV,1)==1
  u = -(LfV + sqrt(LfV.^2 + LgV.^4)) ./  LgV; 
else
  u = -LgV .* (LfV + sqrt(LfV.^2 + sum(LgV.^4))) ./ (sum(LgV.^2));
end

% u = -(LfV + sqrt(LfV.^2 + LgV.^4)) ./ (LgV.*(1+sqrt(1+LgV.^2))); % Sontag #2
% u = -(min(LfV,0) + sqrt(LfV.^2 + LgV.^4)) ./  LgV;% Sontag variation
% u = -LfV./LgV; % Barely stabilize
% u = -(max(LfV,0) + sqrt(max(LfV,0).^2 + abs(LgV.^2))) ./  LgV; % Sontag variation
% u = -(0*LfV + 10*sqrt(Vx2)) ./  LgV; % Variation inspired by Ibanez paper
% u = -(LfV + 3*V) ./  LgV; % exponential stability?

%% Mofied Sontag's formula for control with input saturation limits (STP)
% assumptions made:
% (1) all saturation limits of actuators are [-1,1];
%
% tunable control parameters
% mu = 1;
% lambda = 1;
%
% % modfied Sontag's formula 
% Lf1V = LfV + mu * (norm(x)^2/(norm(x) + lambda));
% Lf2V = LfV + mu * norm(x);
% alpha = (Lf1V + sqrt(Lf2V^2 + (Umax * norm(LgV'))^4)) / ( (norm(LgV'))^2 * (1 + sqrt(1 + (Umax * norm(LgV'))^2)) );
% u = - alpha * LgV';

%%
% tol = 5; % tolerance : variations of V along g with respect to V
% u(abs(repmat(V,size(LgV,1),1))./LgV>tol) = 0;

% u = double(gather(extractdata(u)));
u = max(min(u,Umax),-Umax);
% u(abs(LgV)<0.10*Umax) = 0;

% if not(isinf(Umax)), u(abs(LgV)<0.01*Umax) = 0; end
% u(abs(LgV)<0.01) = 0;
% u(LfV<-1e-2) = 0;

% Zero control action in x = 0
% u(sum(dlX.^2)<1e-2) = 0;

end

