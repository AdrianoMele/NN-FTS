function u = controller_DLTB(network,f,g,t,x,Umax)
% u = controller(parameters,f,g,t,x,Umax)
%   Computes the controller u according to Sontag's formula based on the
%   neural Lyapunov function represented by the NN with the specified
%   parameters.

if isdlarray(x)
  dlX = x;
else
  dlX = dlarray(x,'SBCS');
end

if isdlarray(t)
  dlT = t;
else
  dlT = dlarray(t,'SBCS');
end

[V, gradients_V] = dlfeval(@modelGradients_DLTB,network,dlX,dlT);
Vx = gradients_V{1};
Vt = gradients_V{2};

% remove unnecessary dimensions
Vx = extractdata(squeeze(Vx));
Vt = extractdata(squeeze(Vt));

% Lie derivatives
% dlX = squeeze(dlX);
% dlT = squeeze(dlT);
% 
% f_x = f(dlT,dlX);
% g_x = g(dlT,dlX);
% 
f_x = f(t,x);
g_x = g(t,x);
if isdlarray(f_x), f_x = extractdata(f_x); end
if isdlarray(g_x), g_x = extractdata(g_x); end

LfV = f_x'*Vx;
LgV = g_x'*Vx;

% Sontag #1
alpha = (Vt + LfV);
beta  = LgV;
u = - (beta * (alpha + sqrt(alpha.^2 + sum(beta.^2)^2))) / (sum(beta.^2));

% saturation
u = max(min(u,Umax),-Umax);

end

