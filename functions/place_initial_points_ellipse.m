function [T0,X0] = place_initial_points_ellipse(t,R,NP0,nx,xc)
% Places NP0 points in the set {x | x'Rx<1}

if nargin<5, xc = @(t)[0;0]; end

% Choose even NP0
if mod(NP0,2)==1, NP0 = NP0+1; end

% Place points inside the domain
T0 = t(1)*ones(NP0/2,1);
X0 = zeros(NP0/2,nx);
n = 1;
while n < NP0/2+1
  xt = (rand(nx,1)-[0.5;0.5])*2*max(real(1./sqrt(eig(R))));
  if xt'*R*xt<=1
    X0(n,:) = xt + xc(t(1));
    n = n+1;
  end
end

% Explicitly add boundary
EE = ellipse(R,NP0/2);
X0 = [X0; EE'];
T0 = [T0; T0];

end
