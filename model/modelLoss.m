function [gradients,loss,solutionFound] = modelLoss(parameters,dlX,dlT,dlX0,dlT0,dlXB,dlTB,f,g,Umax,options)

% Extract options
tolVdot   = options.tolVdot;
tolVbound = options.tolVbound;
wVdot   = options.wVdot;
wVbound = options.wVbound;
wVt     = options.wVt;

% Run model and compute gradients of dV
[~, gradients_V] = modelGradients(parameters,dlX,dlT);
Vx = gradients_V{1};
Vt = gradients_V{2};

% Lie derivatives
f_x = f(dlT,dlX);
g_x = g(dlT,dlX);
LfV = sum(Vx.*f_x);

if numel(Umax)==1
  LgV = sum(Vx.*g_x);
elseif numel(Umax)>1
  gU = zeros(numel(Umax),size(g_x,numel(Umax)+1));
  for i = 1 : size(g_x,3)
    gU(:,i)  = squeeze(g_x(:,:,i))*Umax;
    LgV(:,i) = (extractdata(Vx(:,1))'*g_x(:,:,1))';
  end  
end
LgVU = abs(LgV).*Umax; 
Vdot = Vt + LfV - LgVU;
% LgV = sum(Vx.*g_x);
% Vdot_train = LfV - abs(LgV).*Umax;


% Calculate lossVdot: Vdot = Vt + Vx*f(dlT,dlX)
% Vdot = Vt + sum(Vx.*f(dlT,dlX));
Vdoterr = max(Vdot + tolVdot.*sum(dlX.^2),0);
zeroTarget = zeros(size(Vdoterr), 'like', Vdoterr);
lossVdot = mse(Vdoterr,zeroTarget);

% Calculate lossVB: for each point, lossVB = max(0, sup(V0)-inf(VB)) 
% where V0 is V on the initial domain \Omega_0, VB is V
% on the boundary points of \Omega_t at different times.
% We look for a V such that inf(VB) > sup(V0).
% (Note that sup/inf become max/min for compact domains and continuous
% functions).
V0 = model(parameters,dlX0,dlT0);
VB = model(parameters,dlXB,dlTB);
V0max = max(V0);
VBerr = max(V0max-VB + tolVbound, 0);

zeroTarget = zeros(size(VBerr), 'like', VBerr);
lossVB = mse(VBerr,zeroTarget);

% Regularize by slightly penalizing Vt (Currently not used)
zeroTarget = zeros(size(Vt), 'like', Vt);
Vterr = Vt;
lossVt = mse(Vterr,zeroTarget);

% Combine losses
loss = wVdot*lossVdot + wVbound*lossVB + wVt*lossVt;

% Calculate gradients with respect to the learnable parameters.
gradients = dlgradient(loss,parameters,'EnableHigherDerivatives',true);

% Check termination condition: derivative must be nonpositive everywhere,
% distance between inf and sup must be positive
stopFlagVB   = not(any(V0max-VB>=0));
stopFlagVdot = not(any(Vdot>0)) | (options.wVdot==0); % if not optimizing for Vdot, just loook at the other condition;
solutionFound = stopFlagVB && stopFlagVdot;

end
