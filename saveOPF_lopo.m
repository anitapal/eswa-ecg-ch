% Classificação de Arritmias
%
% type : 'AAMI' ou 'AAMI2'
%
% featureSet: conjunto de características
% Pode assimir os valores:
%
% '2005Chazal', '2005Guler', 2005Song, 2006Yu, 2007Yu, 2010YeCoimbra, VCGComplexNet
%
% classifier: classificador a ser utilizado
% Pode assumir os valores:
%
% 'SVM', 'MLP', 'PNN', 'LD'
%
% Autor: Eduardo Luz
%
%

function saveOPF_lopo(featureSet)


% Inicializa os registros
%registers = {'232';'101';'106';'108';'109';'112';'114';'115';'116';'118';'119';'122';'124';'201';'203';'205';'207';'208';'209';'215';'220';'223';'230';...
%    '100';'103';'105';'111';'113';'117';'121';'123';'200';'202';'210';'212';'213';'214';'219';'221';'222';'228';'231';'233';'234'};   

%apenas registros de DS1
registers = {'101';'106';'108';'109';'112';'114';'115';'116';'118';'119';'122';'124';'201';'203';'205';'207';'208';'209';'215';'220';'223';'230'};

for k=1:size(registers,1) % numero de registros

    train_ds = [];
    test_ds = str2double(registers(k));
    count=1;
    
    for j=1:size(registers,1)
        if j ~= k
            train_ds(count) = str2double(registers(j));
            count = count +1;
        end
    end
    
    feat_folder = ['features\' featureSet '\'];
  
    [p1d p1t p2d p2t] = loadDataAAMI(0,feat_folder,train_ds,test_ds);
    
    if strcmp(featureSet,'2007Yu')
        fsel =[1 2 3 4 5 6 7 11 12]; % selecionado com busca para frente
        p1d = p1d(:,fsel) ; 
        p2d = p2d(:,fsel) ; 
    end
    
    [p1d, scale_factor] = mapminmax(p1d');   
    p2d = mapminmax('apply',p2d',scale_factor);
    
    p1d = p1d';
    p2d = p2d';
  
    %% PREPARA PARA OPF

numSamples = size(p1d,1);
numClasses = size(p1t,2);
numFeatures = size(p1d,2);

numSamples2 = size(p2d,1);
numClasses2 = size(p2t,2);
numFeatures2 = size(p2d,2);
    %% TREINO
targetTrain = zeros(size(p1t,1),2); % inicializa

for i=1:size(p1t,1)
targetTrain(i,1) = i-1; % iD para OPF file
end

%transforma as matrizes em um vetor de Ìndices
for i=2:size(p1t,2)
    p1t(find(p1t(:,i)==1),i)=i;
end
targetTrainSum = sum(p1t'); 
targetTrain(:,2) = targetTrainSum';

% cria matriz para OPF
trainOPFMatrix = [targetTrain(:,:) p1d(:,:)];

%% TESTE
targetTest = zeros(size(p2t,1),2); % inicializa

for i=1:size(p2t,1)
targetTest(i,1) = i-1; % iD para OPF file
end

%transforma as matrizes em um vetor de Ìndices
for i=2:size(p2t,2)
    p2t(find(p2t(:,i)==1),i)=i;
end
targetTestSum = sum(p2t'); 
targetTest(:,2) = targetTestSum';

%cria matriz para OPF
testOPFMatrix = [targetTest(:,:) p2d(:,:)];

%OPF file format for datasets
%============================

%The original dataset and its parts training, evaluation and test sets
%must be in the following BINARY file format:
%
%<# of samples> <# of labels> <# of features> 
%<0> <label> <feature 1 from element 0> <feature 2 from element 0> ...
%<1> <label> <feature 1 from element 1> <feature 2 from element 1> ...
%.
%.
%<i> <label> <feature 1 from element i> <feature 2 from element i> ...
%<i+1> <label> <feature 1 from element i+1> <feature 2 from element i+1> ...
%.
%.
%<n-1> <label> <feature 1 from element n-1> <feature 2 from element n-1> ... 

%The first number of each line, <0>, <1>, ... <n-1>, is a sample
%identifier (for n samples in the dataset), which is used in the case
%of precomputed distances. However, the identifier must be specified
%anyway. For unlabeled datasets, please use label 0 for all samples
%(unsupervised OPF).

%Example: Suppose that you have a dataset with 5 samples, distributed
%into 3 classes, with 2 elements from label 1, 2 elements from label 2
%and 1 element from label 3. Each sample is represented by a feature
%vector of size 2. So, the OPF file format should look like as below:

%5	3	2
%0	1	0.21	0.45
%1	1	0.22	0.43
%2	2	0.67	1.12
%3	2	0.60	1.11
%4	3	0.79	0.04


%%--- escreve no arquivo em formato ASCII
opf_folder_train = ['.\OPF\' featureSet '\DS1_train_' char(registers(k)) '.txt'];
arq = fopen(opf_folder_train,'w');
fprintf(arq,'%d %d %d', numSamples, numClasses, numFeatures);
fprintf(arq,'\n');
for p=1:numSamples
    for j=1:numFeatures+2
        if(j<=2)
            fprintf(arq,'%d ', trainOPFMatrix(p,j));
        else
            fprintf(arq,'%f ', trainOPFMatrix(p,j));
        end
    end
    fprintf(arq,'\n');
end
fclose(arq);

opf_folder_test = ['.\OPF\' featureSet '\DS1_test_' char(registers(k)) '.txt'];
arq = fopen(opf_folder_test,'w');
fprintf(arq,'%d %d %d', numSamples2, numClasses2, numFeatures2);
fprintf(arq,'\n');
for p=1:numSamples2
    for j=1:numFeatures+2
        if(j<=2)
            fprintf(arq,'%d ', testOPFMatrix(p,j));
        else
            fprintf(arq,'%f ', testOPFMatrix(p,j));
        end
    end
    fprintf(arq,'\n');
end
fclose(arq);

end % for t

end