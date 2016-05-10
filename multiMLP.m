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
    Neurons = [ 200 200 200 200 200 200];
    Nlayer = size(Neurons,2);
    Nsample = 5;
    I = zeros(1,Nsample);
    for i = 1:size(I,2)
        I(i) = randi(size(T,1));
    end
    
    figure
    
    for j = 1:size(I,2)
        subplot(size(I,2),Nlayer+1,(Nlayer+1)*(j-1)+1)
        imagesc(reshape(Inputs(I(j),:),28,28));
        colormap(gray)
        
    end
    %
    % i = 1;
    % tic
    % [ Input_weights, Input_biases, Output_weights] = ClassicELMtrain( Inputs, Inputs, 100 );
    % toc
    % tic
    % Inputs = ClassicELMtest( Inputs, Input_weights, Input_biases, Output_weights );
    % Inputss = ClassicELMtest( Inputss, Input_weights, Input_biases, Output_weights );
    %
    % toc
    %
    % for j = 1:size(I,2)
    %
    %     subplot(2*size(I,2),Nlayer+1,(Nlayer+1)*2*(j-1)+i)
    %     imagesc(reshape(Inputs(I(j),:),28,28));
    %     colormap(gray)
    %
    %     subplot(2*size(I,2),Nlayer+1,(Nlayer+1)*2*(j-1/2)+i)
    %     imagesc(reshape(Inputss(I(j),:),28,28));
    %     colormap(gray)
    %
    % end
    
    %%
    for i = 1:Nlayer
        
        tic
        [ Input_weights, Input_biases, Output_weights] = ClassicELMtrain( Inputs, Inputs, Neurons(i) );
        toc
        tic
        Inputs = ClassicELMtest( Inputs, Input_weights, Input_biases, Output_weights );        
        toc
        
        for j = 1:size(I,2)
            
            subplot(size(I,2),Nlayer+1,(Nlayer+1)*(j-1)+i+1)
            imagesc(reshape(Inputs(I(j),:),28,28));
            colormap(gray)
            
        end
        
    end
end

%% mean

figure
for i = 1:10
    T = find(Targets_data(:,i));
    Inputs = Input_data(T,:);
    subplot(2,5,i)
    imagesc(reshape(mean(Inputs(:,:)),28,28));
    colormap(gray)
end