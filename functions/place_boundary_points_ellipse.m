function [TB,XB] = place_boundary_points_ellipse(t,G,NPB,nx)

TB = zeros(numel(t)*NPB,1);
XB = zeros(numel(t)*NPB,nx);

for it = 1 : numel(t)
  EE = ellipse(G(t(it)),NPB);
  
  TB((it-1)*NPB+1:it*NPB) = t(it)*ones(1,NPB);
  XB((it-1)*NPB+1:it*NPB,:) = EE';
end

end

