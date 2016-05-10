%% Loading the dataset
tic
Input_data = loadMNISTImages('E:/Deep Learning/ELM Code Matlab/train-images.idx3-ubyte');
Input_data = Input_data';
Ot = loadMNISTLabels('E:/Deep Learning/ELM Code Matlab/train-labels.idx1-ubyte');
Targets_data = zeros(60000, 10);
for i = 1:60000
    Targets_data(i,Ot(i)+1) = 1;
end
clear Ot
disp(['Dataset loaded in ' num2str(toc) 's']);

%%
for loop = 1:10
    
    T = find(Targets_data(:,loop));
    Inputs = Input_data(T,:);
    
    Targets = Targets_data(T,:);
    Nsample = 4*8;
    I = zeros(1,Nsample);
    for i = 1:size(I,2)
        I(i) = randi(size(T,1));
    end
    
    figure
    
    for j = 1:Nsample
        subplot(4,8,j)
        imagesc(reshape(Inputs(I(j),:),28,28));
        colormap(gray)
    end
end