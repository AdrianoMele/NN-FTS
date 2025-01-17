t = rand(1);
x = rand(2,1);
dt  = 1e-4;
dx1 = 1e-4;
dx2 = 1e-4;

dlX = dlarray(x,'SBCS');
dlT = dlarray(t,'SBCS');
[V, gradients_V] = dlfeval(@modelGradients_DLTB,network,dlX,dlT);
V = extractdata(V);
Vx = extractdata(gradients_V{1});
Vt = extractdata(gradients_V{2});

t2 = t+dt;
dlT2 = dlarray(t2,'SBCS');
[V2, gradients_V2] = dlfeval(@modelGradients_DLTB,network,dlX,dlT2);
V2 = extractdata(V2);
[(V2-V)/dt Vt]

x2 = x + [0;dx2];
dlX2 = dlarray(x2,'SBCS');
[V2, gradients_V2] = dlfeval(@modelGradients_DLTB,network,dlX2,dlT);
V2 = extractdata(V2);
[(V2-V)/dx2 Vx(2)]

x2 = x + [dx1;0];
dlX2 = dlarray(x2,'SBCS');
[V2, gradients_V2] = dlfeval(@modelGradients_DLTB,network,dlX2,dlT);
V2 = extractdata(V2);
[(V2-V)/dx1 Vx(1)]
