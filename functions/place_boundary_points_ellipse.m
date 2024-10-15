function [TB,XB] = place_boundary_points_ellipse(t,G,NPB,nx,xc)

if nargin<5, xc = @(t)[0,0]; end

TB = zeros(numel(t)*NPB,1);
XB = zeros(numel(t)*NPB,nx);

for it = 1 : numel(t)
  EE = ellipse(G(t(it)),NPB,xc(t(it)));
  
  TB((it-1)*NPB+1:it*NPB) = t(it)*ones(1,NPB);
  XB((it-1)*NPB+1:it*NPB,:) = EE';
end

end

