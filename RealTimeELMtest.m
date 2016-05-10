function [ Outputs ] = RealTimeELMtest( Inputs, Input_weights, Input_biases, Sorting_weights, Batch_bounds, Output_weights )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
Outputs = zeros(size(Inputs,1), size(Output_weights,3));
t = 1;

for i = 1:size(Inputs,1)
    Required_sub_network = 1;
    while Inputs(i,:)*Sorting_weights > Batch_bounds(Required_sub_network,2)
        Required_sub_network = Required_sub_network + 1;
    end
    
    Outputs(i,:) = ClassicELMtest(Inputs(i,:), Input_weights, Input_biases, squeeze(Output_weights(Required_sub_network,:,:)));
    
    if (i >= t*size(Inputs,1)/4)
        disp([ '     test ' num2str(t*25) '% achieved ...']);
        t = t+1;
    end
    
end

end



