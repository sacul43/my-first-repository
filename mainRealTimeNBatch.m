%% Loading the dataset
tic
Inputs_data = loadMNISTImages('E:/Deep Learning/ELM Code Matlab/train-images.idx3-ubyte');
Inputs_data = Inputs_data';
Ot = loadMNISTLabels('E:/Deep Learning/ELM Code Matlab/train-labels.idx1-ubyte');
Targets_data = zeros(60000, 10);
for i = 1:60000
    Targets_data(i,Ot(i)+1) = 1;
end
clear Ot
disp(['Dataset loaded in ' num2str(toc) 's']);
% Batches = [10,20];
% Batches = [1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50, 75, 100];
Batches = [1,2, 3, 10, 50, 100, 300];
Train_error_rates = zeros(1,size(Batches,2));
Test_error_rates = zeros(1,size(Batches,2));

Nneurones = 50;
Nloop = 1;
for Niter = 1:Nloop
for loop = 1:size(Batches,2)
Nbatch = Batches(loop);
%% Training the network
tic
disp('Start training the network ...');
[Input_weights, Input_biases, Sorting_weights, Batch_bounds, Output_weights] = RealTimeELMtrain( Inputs_data, Targets_data, Nneurones, Nbatch );
disp(['Network trained in ' num2str(toc) 's']);

%% Training error
Outputs = RealTimeELMtest( Inputs_data, Input_weights, Input_biases, Sorting_weights, Batch_bounds, Output_weights );
disp(['training accuracy is ' num2str(100*mean(Single_compare(Outputs, Targets_data))) '%']);

Test_error_rates(loop) = Test_error_rates(loop) + 1-mean(Single_compare(Outputs, Targets_data));

%% Testing the network on test dataset
Inputstest = loadMNISTImages('E:/Deep Learning/ELM Code Matlab/t10k-images.idx3-ubyte');
Inputstest = Inputstest';

Ot = loadMNISTLabels('E:/Deep Learning/ELM Code Matlab/t10k-labels.idx1-ubyte');
Targetstest = zeros(10000, 10);
for i = 1:10000
    Targetstest(i,Ot(i)+1) = 1;
end
clear Ot

Outputs = RealTimeELMtest( Inputstest, Input_weights, Input_biases, Sorting_weights, Batch_bounds, Output_weights );

disp(['test accuracy is ' num2str(100*mean(Single_compare(Outputs, Targetstest))) '%']);
Train_error_rates(loop) = Train_error_rates(loop) + 1-mean(Single_compare(Outputs, Targetstest));

end
end

Train_error_rates = Train_error_rates/Nloop;
Test_error_rates = Test_error_rates/Nloop;

figure

plot(Batches, Test_error_rates, Batches, Train_error_rates, '--')

title(['Erreur en test et en train (--) pour ' num2str(Nneurones) ' neurones par batch, en fonction du nombre de batch (moyenne sur ' num2str(Nloop) ' réalisations)'])
xlabel('Nombre de batch')
ylabel('Taux d''erreur')
hold on
disp([num2str(toc)]);