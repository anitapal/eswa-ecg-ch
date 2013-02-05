function saveMAT_lopo(featureSet)
% Salva matrix de características para lopo em arquivo .MAT
% As inst6ancias estão organizadas por linha e a última coluna é o atributo
% classe
% 
%Autor:Eduardo Luz
%

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
train_file = ['.\MAT\' featureSet '\DS1-' char(registers(k)) '-train.mat'];
save(train_file, 'trainMatrix');
test_file = ['.\MAT\' featureSet '\DS1-' char(registers(k)) '-test.mat'];
save(test_file, 'testMatrix');

end % for t

end