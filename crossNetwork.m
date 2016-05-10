function [ Outputs ] = crossNetwork( Inputs, Network_weights )
%CROSSNETWORK Summary of this function goes here
%   Detailed explanation goes here

tmp = Inputs;

for layer = 1:size(Network_weights,2)
    
    tmp = crossLayer( tmp, Network_weights{layer});
    
end

Outputs = tmp;

end

