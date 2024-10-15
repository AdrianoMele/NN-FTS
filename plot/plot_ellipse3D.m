function plot_ellipse3D(t,E,xc,varargin)
 % plots an ellipse of the form xEx = 1

 ellipse_ = ellipse(E,30,xc);
 t = t + zeros(1,size(ellipse_,2));
 plot3(t,ellipse_(1,:), ellipse_(2,:), varargin{:})
end 