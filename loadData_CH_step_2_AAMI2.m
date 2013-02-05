% Autores : Eduardo Luz 
% Fun��o : Caregar as caracter�sticas exta�das pelo "featureExtraction.m"
% e coloca-las em 2 parti��es conforme norma da AAMI.
%
% pca = 1 => faz o PCA com 99.32% (DE ACORDO COM O PAPER)
% pca = 0 ignora PCA
%
% type = Cpntrole das caracter�sticas din�micas (informa��o de RR)
% type = 1 : caract. din�micas originais do paper 
% type = 2 : remo��o de todas caract. din�micas 
% type = 3 : caract. din�micas RELATIVIZADAS 
%
function [p1d p1t p2d p2t] = loadData_CH_step_2_AAMI2(pca, file_path, trainset, testset)

if nargin < 4
   % pca = 0;
   dataset1 = {'101';'106';'108';'109';'112';'114';'115';'116';'118';'119';'122';'124';'201';'203';'205';'207';'208';'209';'215';'220';'223';'230'};
   dataset2 = {'100';'103';'105';'111';'113';'117';'121';'123';'200';'202';'210';'212';'213';'214';'219';'221';'222';'228';'231';'232';'233';'234'};
   % dataset1 = str2double(dataset1(t));
else
    dataset1 = trainset;
    dataset2 = testset;
end

%----------------- inicializacao das variaveis
dados1  = [];
rotulo1 = [];
dados2  = [];
rotulo2 = [];
%-------------------------------

%carrega dos arquivos e separa apenas os batimentos desejados
for i=1:size(dataset1,2)
    if nargin < 4
        dataset1(i) = str2double(dataset1(i));
    end
    filename = strcat(num2str(dataset1(i)),'.txt');
    filename = strcat(file_path,filename);
    [data target] = loadarq(char(filename));
    dados1 = [dados1;data];
    rotulo1 = [rotulo1;target];    
end

for i=1:size(dataset2,2)
    if nargin < 4
        dataset2(i) = str2double(dataset2(i));
    end
    filename = strcat(num2str(dataset2(i)),'.txt');
    filename = strcat(file_path,filename);
    [data target] = loadarq(char(filename));
    dados2 = [dados2;data];
    rotulo2 = [rotulo2;target];    
end

%normaliza: X - med / std
%[dados1 dados2 rotulo1 rotulo2] = normaliza(dados1, dados2, rotulo1, rotulo2);

p1d = dados1; 
p1t = rotulo1;

p2d = dados2;
p2t = rotulo2;

if(pca == 1)  
        %aplica PCA para reduzir as caracter�sticas
        %[coeff index1]=applyPCA(p1d, 99.32);
        [coeff index1]=applyPCA(p1d, 99);

        p1d = p1d * coeff(:,1:index1);
        p2d = p2d * coeff(:,1:index1);
end 
   
end

%-------------------- Fun��o de separa��o dos tipos
% os r�tulos est�o em ascii code
% Seguindo o padr�o AAMI heartbeat classes, Normal (N), supraventricular
% ectopic beat (SVEB), Ventricular ectopic beat(VEB), fusion beat(F) e
% unknown beat (Q)
% 
function [data target]=loadarq(arq)
amostra = load(arq);
tam = size(amostra,2);

pos = find(amostra(:,tam)==65); % A -> SVEB
A  = amostra(pos,1:tam-1);
sz = size(A,1);
tA = repmat([1 0],sz,1); 

pos = find(amostra(:,tam)==86);  % V -> VEB
V  = amostra(pos,1:tam-1);
sz = size(V,1);
tV = repmat([0 1],sz,1); 

pos = find(amostra(:,tam)==97);  % a -> SVEB
a  = amostra(pos,1:tam-1);
sz = size(a,1);
ta = repmat([1 0],sz,1); 

pos = find(amostra(:,tam)==70);  % F -> F _> VEB
F = amostra(pos,1:tam-1);
sz = size(F,1);
tF = repmat([0 1],sz,1); 

pos = find(amostra(:,tam)==69);  % E -> VEB
E  = amostra(pos,1:tam-1);
sz = size(E,1);
tE = repmat([0 1],sz,1); 

pos = find(amostra(:,tam)==74);  % J -> SVEB
J  = amostra(pos,1:tam-1);
sz = size(J,1);
tJ = repmat([1 0],sz,1); 


pos = find(amostra(:,tam)==83);  % S -> SVEB
S  = amostra(pos,1:tam-1);
sz = size(S,1);
tS = repmat([1 0],sz,1); 

data = [A;V;a;F;S;E;J];
target = [tA;tV;ta;tF;tS;tE;tJ;];
end

