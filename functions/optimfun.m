function J = optimfun(P,V,x)
% Function used in the quadratic fit.

J = 0;
for i = 1 : size(x,1)
  J = J + (x(i,:)*(P'*P)*x(i,:)' - V(i))^2;
end

end

