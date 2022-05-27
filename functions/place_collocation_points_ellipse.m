function [TC,XC] = place_collocation_points_ellipse(t,G,NPC,nx)
% Places NPC collocation points in the set 
% [t_0, t_0+T] x {x | x'G(t)x<1}

%%% Alternative: 3D random points
TC = t(1) + rand(NPC,1)*(t(end)-t(1));
XC = zeros(NPC,nx);
for it = 1 : NPC
flag = false;
  while not(flag)
    xt = (rand(nx,1)-0.5*ones(nx,1))*2*max(real(1./sqrt(eig(G(TC(it))))));
    if xt'*G(TC(it))*xt<1
      XC(it,:) = xt;
      flag = true;
    end
  end
end
end

