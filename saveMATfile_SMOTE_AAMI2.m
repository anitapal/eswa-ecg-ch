
function [p1d p1t p2d p2t] = saveMATfile_SMOTE_AAMI2(featureSet)
% Salva matrix de características para lopo em arquivo .MAT
% As inst6ancias estão organizadas por linha e a última coluna é o atributo
% classe
%
% Autor:Eduardo Luz
%

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
[p1d p1t p2d p2t] = loadDataAAMI2(0,feat_folder,train_ds,test_ds);

[p1d p1t] = SMOTE(p1d, p1t, 2, 1200, 3);  
[p1d p1t] = SMOTE(p1d, p1t, 3, 1000, 3); 
    
%[p1d, scale_factor] = mapminmax(p1d');   
%p2d = mapminmax('apply',p2d',scale_factor);

%p1d = p1d';
%p2d = p2d';
    
    %% TREINO
targetTrain = zeros(size(p1t,1),1); % inicializa

%transforma as matrizes em um vetor de Ìndices
for i=2:size(p1t,2)
    p1t(find(p1t(:,i)==1),i)=i;
end
targetTrainSum = sum(p1t'); 
targetTrain(:,1) = targetTrainSum';

% cria matriz para arquivo .MAT
trainMatrix = [p1d(:,:) targetTrain(:,:) ];

%% TESTE
targetTest = zeros(size(p2t,1),1); % inicializa

%transforma as matrizes em um vetor de Ìndices
for i=2:size(p2t,2)
    p2t(find(p2t(:,i)==1),i)=i;
end
targetTestSum = sum(p2t'); 
targetTest(:,1) = targetTestSum';

%cria matriz para arquivo .MAT
testMatrix = [p2d(:,:) targetTest(:,:)];

%% salva as matrizes
train_file = ['.\MAT\' featureSet '\DS1-train-SMOTE-aami2.mat'];
save(train_file, 'trainMatrix');
test_file = ['.\MAT\' featureSet '\DS2-valid-SMOTE-aami2.mat'];
save(test_file, 'testMatrix');
    
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

