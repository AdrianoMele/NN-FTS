function plot_ellipse3D(x,E,varargin)
 % plots an ellipse of the form xEx = 1
 R = chol(E);
 t = linspace(0, 2*pi, 100); % or any high number to make curve smooth
 z = [cos(t); sin(t)];
 ellipse = inv(R) * z;
 x = x + t*0;
 plot3(x,ellipse(1,:), ellipse(2,:), varargin{:})
end 