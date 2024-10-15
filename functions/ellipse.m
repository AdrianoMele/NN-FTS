function ellipse = ellipse(E,NP,xc)
% Returns points of a 2D ellipse of the form xEx = 1.
% if the third argument is specified, the center of the ellipse is shifted
% by xc.
if nargin<3,    xc = [0;0]; end 
if isempty(xc), xc = [0;0]; end 
if isrow(xc),   xc = xc';   end

R = chol(E);
t = linspace(0, 2*pi, NP) + rand(1); % or any high number to make curve smooth; add random number to shuffle points at different times
z = [cos(t); sin(t)];
ellipse = inv(R) * z + xc;
end 