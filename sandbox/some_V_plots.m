% initial point over time

Ts = 1e-3;
t = t(1):Ts:t(end);

x0 = [.01;.02];
x = zeros(nx,numel(t));
x(:,1) = x0;

for i = 1 : numel(t)
  x(:,i) = x0;
end

dlt = dlarray(t,'SBCS');
dlx = dlarray(x,'SBCS');
V = model_DLTB(network,dlx,dlt);

figure
plot(V)

%%

% domain center at time t
for i = 1 : numel(t)
  x(:,i) = xc(t(i));
end

dlt = dlarray(t,'SBCS');
dlx = dlarray(x,'SBCS');
V = model_DLTB(network,dlx,dlt);

figure
plot(V)