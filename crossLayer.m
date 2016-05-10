function [ Outputs ] = crossLayer( Inputs, Input_weights )
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
H = Inputs*Input_weights;
Outputs = 1./(1 + exp(H(:,:)));
end

