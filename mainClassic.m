%% Loading the dataset
tic
Inputs = loadMNISTImages('E:/Deep Learning/ELM Code Matlab/train-images.idx3-ubyte');
Inputs = Inputs';
Ot = loadMNISTLabels('E:/Deep Learning/ELM Code Matlab/train-labels.idx1-ubyte');
Targets = zeros(60000, 10);
for i = 1:60000
    Targets(i,Ot(i)+1) = 1;
end
clear Ot
disp(['Dataset loaded in ' num2str(toc) 's']);

%% Training the network
tic
disp('Training the network ...');

[ Input_weights, Input_biases, Output_weights] = ClassicELMtrain( Inputs, Targets, 100 );
disp(['Network trained in ' num2str(toc) 's']);

%% Training error
Outputs = ClassicELMtest( Inputs, Input_weights, Input_biases, Output_weights );
disp(['training accuracy is ' num2str(100*mean(Single_compare(Outputs, Targets))) '%']);

%% Testing the network
Inputs = loadMNISTImages('E:/Deep Learning/ELM Code Matlab/t10k-images.idx3-ubyte');
Inputs = Inputs';

Ot = loadMNISTLabels('E:/Deep Learning/ELM Code Matlab/t10k-labels.idx1-ubyte');
Targets = zeros(10000, 10);
for i = 1:10000
    Targets(i,Ot(i)+1) = 1;
end
clear Ot

Outputs = ClassicELMtest( Inputs, Input_weights, Input_biases, Output_weights );

disp(['test accuracy is ' num2str(100*mean(Single_compare(Outputs, Targets))) '%']);
