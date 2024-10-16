function [ellipse] = plot_ellipse(E,xc,varargin)
 % plots an ellipse of the form xEx = 1

 if isempty(xc), xc = [0;0]; end

 R = chol(E);
 t = linspace(0, 2*pi, 100); % or any high number to make curve smooth
 z = [cos(t); sin(t)];
 ellipse = inv(R) * z;
 plot(ellipse(1,:)+xc(1), ellipse(2,:)+xc(2), varargin{:})
end 