function u = controller_DLTB(network,f,g,t,x,Umax)
% u = controller(parameters,f,g,x,Umax)
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

[~, gradients_V] = dlfeval(@modelGradients_DLTB,network,dlX,dlT);
Vx = gradients_V{1};
Vt = gradients_V{2};

% remove unnecessary dimensions
Vx = extractdata(squeeze(Vx));
Vt = extractdata(squeeze(Vt));
dlX = squeeze(dlX);
dlT = squeeze(dlT);

% Lie derivatives
f_x = f(dlT,dlX);
g_x = g(dlT,dlX);

if isdlarray(f_x), f_x = extractdata(f_x); end
if isdlarray(g_x), g_x = extractdata(g_x); end

LfV = f_x'*Vx;
LgV = g_x'*Vx;


% Sontag #1
u = -LgV .* (Vt + LfV + sqrt(LfV.^2 + sum(LgV.^4))) ./ (sum(LgV.^2));
% if size(LgV,1)==1
%   u = -(Vt + LfV + sqrt(LfV.^2 + LgV.^4)) ./  LgV; 
% else
%   u = -LgV .* (Vt + LfV + sqrt(LfV.^2 + sum(LgV.^4))) ./ (sum(LgV.^2));
% end

u = max(min(u,Umax),-Umax);

end

