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
function [p1d p1t p2d p2t] = saveOPFfile_SMOTE(featureSet)

%if nargin < 2
   smote_S_V = 1;
%end

% datasets recomendades pela AAMI : sem os pacientes com mrca-passo
dataset1 = {'101';'106';'108';'109';'112';'114';'115';'116';'118';'119';'122';'124';'201';'203';'205';'207';'208';'209';'215';'220';'223';'230'};
dataset2 = {'100';'103';'105';'111';'113';'117';'121';'123';'200';'202';'210';'212';'213';'214';'219';'221';'222';'228';'231';'232';'233';'234'};

train_ds = [];
for j=1:size(dataset1,1)
    train_ds(j) = str2double(dataset1(j));
end
test_ds = [];
for j=1:size(dataset2,1)
    test_ds(j) = str2double(dataset2(j));
end

%----------------- inicializacao das variaveis
dados1  = [];
rotulo1 = [];
dados2  = [];
rotulo2 = [];
%---------
feat_folder = ['features\' featureSet '\'];
[p1d p1t p2d p2t] = loadDataAAMI(0,feat_folder,train_ds,test_ds);

%if strcmp(featureSet,'2007Yu')
%    fsel =[1 2 3 4 5 6 7 11 12]; % selecionado com busca para frente
%    p1d = p1d(:,fsel) ; 
%    p2d = p2d(:,fsel) ; 
%end
    
if smote_S_V == 1
    [p1d p1t] = SMOTE(p1d, p1t, 2, 1200, 3);  
    [p1d p1t] = SMOTE(p1d, p1t, 3, 1000, 3); 
    [p1d p1t] = SMOTE(p1d, p1t, 4, 2000, 3); 
    [p1d p1t] = SMOTE(p1d, p1t, 5, 32000, 3); 
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
    opf_folder_train = ['.\OPF\' featureSet '\DS1_smote_aami.txt'];
else
    opf_folder_train = ['.\OPF\' featureSet '\DS1.txt'];
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

opf_folder_test = ['.\OPF\' featureSet '\DS2.txt'];
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

%-------------------- Função de separação dos tipos
% os rótulos estão em ascii code
% Seguindo o padrão AAMI heartbeat classes, Normal (N), supraventricular
% ectopic beat (SVEB), Ventricular ectopic beat(VEB), fusion beat(F) e
% unknown beat (Q)
% 
function [data target]=loadarq(arq)
amostra = load(arq);
tam = size(amostra,2);

pos = find(amostra(:,tam)==78); % N -> N
N = amostra(pos,1:tam-1);
sz = size(N,1);
tN = repmat([1 0 0 0 0],sz,1); 

pos = find(amostra(:,tam)==76); % L -> N
L = amostra(pos,1:tam-1);
sz = size(L,1);
tL = repmat([1 0 0 0 0],sz,1); 

pos = find(amostra(:,tam)==82);  % R -> N
R  = amostra(pos,1:tam-1);
sz = size(R,1);
tR = repmat([1 0 0 0 0],sz,1); 

pos = find(amostra(:,tam)==65); % A -> SVEB
A  = amostra(pos,1:tam-1);
sz = size(A,1);
tA = repmat([0 1 0 0 0],sz,1); 

pos = find(amostra(:,tam)==86);  % V -> VEB
V  = amostra(pos,1:tam-1);
sz = size(V,1);
tV = repmat([0 0 1 0 0],sz,1); 

pos = find(amostra(:,tam)==47);  % paced (/) -> Q
P  = amostra(pos,1:tam-1);
sz = size(P,1);
tP = repmat([0 0 0 0 1],sz,1); 

pos = find(amostra(:,tam)==97);  % a -> SVEB
a  = amostra(pos,1:tam-1);
sz = size(a,1);
ta = repmat([0 1 0 0 0],sz,1); 

pos = find(amostra(:,tam)==70);  % F -> F
F = amostra(pos,1:tam-1);
sz = size(F,1);
tF = repmat([0 0 0 1 0 ],sz,1); 

pos = find(amostra(:,tam)==106);  % j -> N
j  = amostra(pos,1:tam-1);
sz = size(j,1);
tj = repmat([1 0 0 0 0],sz,1); 

pos = find(amostra(:,tam)==102);  % f -> Q
f = amostra(pos,1:tam-1);
sz = size(f,1);
tf = repmat([0 0 0 0 1],sz,1); 

pos = find(amostra(:,tam)==69);  % E -> VEB
E  = amostra(pos,1:tam-1);
sz = size(E,1);
tE = repmat([0 0 1 0 0],sz,1); 

pos = find(amostra(:,tam)==74);  % J -> SVEB
J  = amostra(pos,1:tam-1);
sz = size(J,1);
tJ = repmat([0 1 0 0 0],sz,1); 

pos = find(amostra(:,tam)==101);  % e -> N
e  = amostra(pos,1:tam-1);
sz = size(e,1);
te = repmat([1 0 0 0 0],sz,1); 

pos = find(amostra(:,tam)==83);  % S -> SVEB
S  = amostra(pos,1:tam-1);
sz = size(S,1);
tS = repmat([0 1 0 0 0],sz,1); 

pos = find(amostra(:,tam)==81);  % Q -> Q
Q  = amostra(pos,1:tam-1);
sz = size(Q,1);
tQ = repmat([0 0 0 0 1],sz,1);


data = [N;L;R;A;V;P;a;F;S;j;f;E;J;e;Q];
target = [tN;tL;tR;tA;tV;tP;ta;tF;tS;tj;tf;tE;tJ;te;tQ];
end

