% Autores : Eduardo Luz 
% Função : Caregar as características extaídas pelo "featureExtraction.m"
% e coloca-las em 2 partições conforme norma da AAMI.
%
% pca = 1 => faz o PCA com 99.32% (DE ACORDO COM O PAPER)
% pca = 0 ignora PCA
%
% type = Cpntrole das características dinâmicas (informação de RR)
% type = 1 : caract. dinâmicas originais do paper 
% type = 2 : remoção de todas caract. dinâmicas 
% type = 3 : caract. dinâmicas RELATIVIZADAS 
%
function ARFFtoOPF(trainMatrix, testMatrix)

featureSet = 'ARFF'; % nome do arquivo
featSel = [];
smote_S_V = 0;

%% PREPARA PARA OPF

numSamples = size(trainMatrix,1);
numClasses = max(trainMatrix(:,end))+1;
numFeatures = size(trainMatrix,2)-1; % o label não entra

numSamples2 = size(testMatrix,1);
numClasses2 = max(testMatrix(:,end))+1;
numFeatures2 = size(testMatrix,2)-1; % o label não entra

%% TREINO
targetTrain = zeros(size(trainMatrix,1),2); % inicializa

for i=1:size(trainMatrix,1)
targetTrain(i,1) = i-1; % iD para OPF file
end

%transforma as matrizes em um vetor de Ìndices
targetTrain(:,2) = trainMatrix(:,end) + 1;

% cria matriz para OPF
trainOPFMatrix = [targetTrain(:,:) trainMatrix(:,1:end-1)];

%% TESTE
targetTest = zeros(size(testMatrix,1),2); % inicializa

for i=1:size(testMatrix,1)
targetTest(i,1) = i-1; % iD para OPF file
end

targetTest(:,2) = testMatrix(:,end) + 1;

%cria matriz para OPF
testOPFMatrix = [targetTest(:,:) testMatrix(:,1:end-1)];

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

%fwrite(fileID, A, precision, skip, machineformat)
%precision = double
%skip =0
%Windows systems use little-endian ordering, and most UNIX systems use big-endian ordering, for both bytes and bits.
%'l' or 'ieee-le' Little-endian ordering
%'s' or 'ieee-be' Big-endian ordering
%machineformat = 'ieee-be.l64'

%fid = fopen('trainOPF.dat', 'w');
%fwrite(fid, [numSamples numClasses numFeatures], 'double',0,'ieee-be'); % primeira linha
%fwrite(fid, trainOPFMatrix, 'double',0,'ieee-be');
%fwrite(fid, [numSamples numClasses numFeatures], 'int32','0','ieee-be'); % primeira linha
%fwrite(fid, trainOPFMatrix, 'int32','0','ieee-be');
%fclose(fid);

%fid = fopen('testOPF.dat', 'w');
%fwrite(fid, [numSamples2 numClasses2 numFeatures2], 'double',0,'ieee-be'); % primeira linha
%fwrite(fid, testOPFMatrix, 'double',0,'ieee-be');
%fwrite(fid, [numSamples numClasses numFeatures], 'int32','0','ieee-be'); % primeira linha
%fwrite(fid, testOPFMatrix, 'int32','0','ieee-be');
%fclose(fid);

%%--- escreve no arquivo em formato ASCII
if smote_S_V == 1
    opf_folder_train = ['.\OPF\' featureSet '\DS1_smote.txt'];
else
    if (~isempty(featSel))
        numfs = size(featSel,2);
        opf_folder_train = ['.\OPF\' featureSet '\DS1' '_FS' num2str(numfs) '_' '.txt'];
    else
        opf_folder_train = ['.\OPF\' featureSet '\DS1.txt'];
    end
end
arq = fopen(opf_folder_train,'w');
fprintf(arq,'%d %d %d', numSamples, numClasses, numFeatures);
fprintf(arq,'\n');
for k=1:numSamples
    for j=1:numFeatures+2
        if(j<=2)
            fprintf(arq,'%d ', trainOPFMatrix(k,j));
        else
            fprintf(arq,'%f ', trainOPFMatrix(k,j));
        end
    end
    fprintf(arq,'\n');
end
fclose(arq);

if (~isempty(featSel))
    numfs = size(featSel,2);
    opf_folder_test = ['.\OPF\' featureSet '\DS2' '_FS' num2str(numfs) '_' '.txt'];
else
    opf_folder_test = ['.\OPF\' featureSet '\DS2.txt'];
end
arq = fopen(opf_folder_test,'w');
fprintf(arq,'%d %d %d', numSamples2, numClasses2, numFeatures2);
fprintf(arq,'\n');
for k=1:numSamples2
    for j=1:numFeatures+2
        if(j<=2)
            fprintf(arq,'%d ', testOPFMatrix(k,j));
        else
            fprintf(arq,'%f ', testOPFMatrix(k,j));
        end
    end
    fprintf(arq,'\n');
end
fclose(arq);
    
end



