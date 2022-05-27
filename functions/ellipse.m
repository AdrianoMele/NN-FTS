function ellipse = ellipse(E,NP)
 % Returns points of an ellipse of the form xEx = 1
 R = chol(E);
 t = linspace(0, 2*pi, NP) + rand(1); % or any high number to make curve smooth; add random number to shuffle points at different times
 z = [cos(t); sin(t)];
 ellipse = inv(R) * z;
end 