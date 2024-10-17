function [network] = train_model_DLTB(network,f,g,Umax, ...
  ds,dlT0,dlX0,dlTB,dlXB,...
  miniBatchSize,numEpochs,executionEnvironment,...
  initialLearnRate,decayRate,options,verbose)

% If training using a GPU, convert the initial and conditions to |gpuArray|.
if (executionEnvironment == "auto" && canUseGPU) || (executionEnvironment == "gpu")
  dlT0 = gpuArray(dlT0);
  dlX0 = gpuArray(dlX0);
  dlTB = gpuArray(dlTB);
  dlXB = gpuArray(dlXB);
end

% % Shuffle dataset
% ds = shuffle(ds);

% Used to verify termination condition
TXC  = cell2mat(ds.readall);
dlTC = dlarray(TXC(:,1)',    'SBCS');
dlXC = dlarray(TXC(:,2:end)','SBCS');

% Arrange points in a minibatch
mbq = minibatchqueue(ds, ...
  'MiniBatchSize',miniBatchSize, ...
  'OutputAsDlarray',false,... % otherwise fiddle with MiniBatchFormat...
  'OutputEnvironment',executionEnvironment);

% For each iteration:
% * Read a mini-batch of data from the mini-batch queue
% * Evaluate the model gradients and loss using the accelerated model gradients
% and |dlfeval| functions.
% * Update the learning rate.
% * Update the learnable parameters using the |adamupdate| function.
% At the end of each epoch, update the training plot with the loss values.

% Initialize the parameters for the Adam solver.
averageGrad   = [];
averageSqGrad = [];

% Accelerate the model gradients function using the |dlaccelerate| function.
% accfun_loss = dlaccelerate(@modelLoss_DLTB);
accfun_loss = @modelLoss_DLTB; % no acceleration :(

% Initialize the training progress plot.
if verbose
  ht = figure('Position',[250 300 850 470]);
  C = colororder;
  lineLoss = animatedline('Color',C(2,:),'LineWidth',2);
  ylim([0 inf])
  xlabel("Iteration")
  ylabel("Loss")
  grid on
end

start = tic;
iteration = 0;
for epoch = 1:numEpochs

  % Reset mbq and stop condition
  reset(mbq);

  while hasdata(mbq)

    % Update iteration number
    iteration = iteration + 1;

    % Extract next minibatch
    next_sample = next(mbq);
    T = next_sample(:,1);
    X = next_sample(:,2:end);

    % To be checked
    dlX = dlarray(X','SBCS');
    dlT = dlarray(T','SBCS');

    % Evaluate the model gradients and loss using dlfeval
    [gradients,loss,~] = dlfeval(accfun_loss,network,dlX,dlT,dlX0,dlT0,dlXB,dlTB,f,g,Umax,options);

    % Update learning rate
    learningRate = initialLearnRate / (1+decayRate*iteration);

    % Update the network parameters using the adamupdate function
    [network,averageGrad,averageSqGrad] = adamupdate(network,gradients,averageGrad, ...
      averageSqGrad,iteration,learningRate);

    % Check for NaNs
    if any(isnan(network.Learnables.Value{1}))
      error('Some parameters are NaN, try decreasing the learning rate.')
    end

    % Diagnostics
    loss = double(gather(extractdata(loss)));
    D = duration(0,0,toc(start),'Format','hh:mm:ss');
    if verbose
      % Plot training progress
      addpoints(lineLoss,iteration, loss);
      figure(ht)
      title(sprintf("Epoch: %d | Elapsed: %s | Learning rate: %.6f | Loss: %.5f", epoch, string(D), learningRate, loss))
      drawnow
    else
      fprintf("Epoch: %d | Elapsed: %s | Learning rate: %.6f | Loss: %.5f \n", epoch, string(D), learningRate, loss)
    end
  end

  % Break the cycle if a solution is found
  [~,~,solutionFound] = dlfeval(@modelLoss_DLTB,network,dlXC,dlTC,dlX0,dlT0,dlXB,dlTB,f,g,Umax,options); 
  if solutionFound
    disp('Solution found!')
    break
  end
end

end

