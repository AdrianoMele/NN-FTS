function [gradients,loss,solutionFound] = modelLoss_DLTB(network,dlX,dlT,dlX0,dlT0,dlXB,dlTB,f,g,Umax,options)

% Extract options
tolVdot   = options.tolVdot;
tolVbound = options.tolVbound;
wVdot     = options.wVdot;
wVbound   = options.wVbound;
wVt       = options.wVt;
wV        = options.wV;

% Run model and compute gradients of dV
[V, gradients_V] = modelGradients_DLTB(network,dlX,dlT);
Vx = gradients_V{1};
Vt = gradients_V{2};

% remove unnecessary dimensions
Vx = squeeze(Vx);
Vt = squeeze(Vt);

dlX = squeeze(dlX);
dlT = squeeze(dlT);

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
    LgV(:,i) = (extractdata(Vx(:,i))'*g_x(:,:,i))';
  end  
end
LgVU = sum(abs(LgV).*Umax); 
Vdot = Vt + LfV - LgVU;
% LgV = sum(Vx.*g_x);
% Vdot_train = LfV - abs(LgV).*Umax;

% Calculate lossVdot: Vdot = Vt + Vx*f(dlT,dlX)
% Vdot = Vt + sum(Vx.*f(dlT,dlX));
Vdoterr = max(Vdot + tolVdot.*sum(dlX.^2),0); % ~ Zubov PDE
zeroTarget = zeros(size(Vdoterr), 'like', Vdoterr);
lossVdot = mse(Vdoterr,zeroTarget);

% Calculate lossVB: for each point, lossVB = max(0, sup(V0)-inf(VB)) 
% where V0 is V on the initial domain \Omega_0, VB is V
% on the boundary points of \Omega_t at different times.
% We look for a V such that inf(VB) > sup(V0).
% (Note that sup/inf become max/min for compact domains and continuous
% functions).
V0 = model_DLTB(network,dlX0,dlT0);
VB = model_DLTB(network,dlXB,dlTB);
V0max = max(V0);
VBerr = max(V0max-VB + tolVbound, 0);

zeroTarget = zeros(size(VBerr), 'like', VBerr);
lossVB = mse(VBerr,zeroTarget);

% Regularize by slightly penalizing Vt (Currently not used)
zeroTarget = zeros(size(Vt), 'like', Vt);
Vterr = Vt;
lossVt = mse(Vterr,zeroTarget);

% Weight on V for regularization
zeroTarget = zeros(size(V), 'like', V);
lossV = mse(V,zeroTarget);

% Combine losses
loss = wVdot*lossVdot + wVbound*lossVB + wVt*lossVt + wV*lossV;

% Calculate gradients with respect to the learnable parameters.
gradients = dlgradient(loss,network.Learnables,'EnableHigherDerivatives',true);

% Check termination condition: derivative must be nonpositive everywhere,
% distance between inf and sup must be positive
stopFlagVB    = not(any(V0max-VB>=0));
stopFlagVdot  = not(any(Vdot>0)) | (options.wVdot==0); % if not optimizing for Vdot, just loook at the other condition
solutionFound = stopFlagVB & stopFlagVdot;

end
