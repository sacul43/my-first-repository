function [ Input_weights, Input_biases, Sorting_weights, Batch_bounds, Output_weights] = RealTimeELMtrain( Inputs, Targets, Nb_hidden_neurons, Nb_batch )
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here

N = size(Inputs,1);
L = Nb_batch;
K = ceil(N/L);

% Sorting the data
t1 = toc;
Sorting_weights = rand(size(Inputs,2),1);
[~,I] = sort(Inputs*Sorting_weights);
Inputs = Inputs(I,:);
Targets = Targets(I,:);
t2 = toc;

disp(['     Dataset sorted in ' num2str(t2-t1) 's']);

% Batching inputs and targets
Input_batches = zeros(L, K, size(Inputs,2));
for i = 1:L-1
    Input_batches(i,:,:) = Inputs((i-1)*K+1:i*K,:);
end
Input_batches(L,1:N-(L-1)*K,:) = Inputs((L-1)*K+1:end,:);

Target_batches = zeros(L, K, size(Targets,2));
for i = 1:L-1
    Target_batches(i,:,:) = Targets((i-1)*K+1:i*K,:);
end
Target_batches(L,1:N-(L-1)*K,:) = Targets((L-1)*K+1:end,:);


Batch_bounds = zeros(L,2);
for i = 1:L
    Batch_bounds(i,1) = squeeze(Input_batches(i,1,:))'*Sorting_weights;
    Batch_bounds(i,2) = squeeze(Input_batches(i,end,:))'*Sorting_weights;
end;

% Start building the network
Input_weights = rand(size(Inputs,2), Nb_hidden_neurons) - 0.5;
Input_biases = rand(1, Nb_hidden_neurons) - 0.5;

% Building the L sub-networks
Output_weights = zeros(L,Nb_hidden_neurons, size(Targets,2));
for i = 1:L
    disp(['     Training on batch number ' num2str(i) ' ...']);
    Output_weights(i,:,:) = CalculateOutputWeights( squeeze(Input_batches(i,:,:)), Input_weights, Input_biases, squeeze(Target_batches(i,:,:)) );
end

% Tying the L sub-networks


end

