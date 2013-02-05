% Autores : Eduardo Luz 
% Função : salava as caracteristicas em formato ARFF do weka. Esta função
% salva 22 particoes, referentes aos pacientes de DS1, para fazer o lopo.
%
function [p1d p1t p2d p2t] = saveARFF_lopo(featureSet, featSel, smote_S_V)

if nargin < 2
   smote_S_V = 0;
   featSel=[];
elseif nargin < 3
   smote_S_V = 0;
end

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

    %----------------- inicializacao das variaveis
    dados1  = [];
    rotulo1 = [];
    dados2  = [];
    rotulo2 = [];
    %---------
    feat_folder = ['features\' featureSet '\'];
    [p1d p1t p2d p2t] = loadDataAAMI(0,feat_folder,train_ds,test_ds);

    if ~isempty(featSel)
        fsel = find(featSel==1);

        p1d = p1d(:,fsel) ; 
        p2d = p2d(:,fsel) ; 
    end

    if smote_S_V == 1
        [p1d p1t] = SMOTE(p1d, p1t, 2, 1000, 5);  
        [p1d p1t] = SMOTE(p1d, p1t, 3, 1000, 5);  
    end

    [p1d, scale_factor] = mapminmax(p1d');   
    p2d = mapminmax('apply',p2d',scale_factor);

    p1d = p1d';
    p2d = p2d';

    %% PREPARA PARA ARFF
    WEKA_HOME = 'C:\Arquivos de programas\Weka-3-6';

    javaaddpath([WEKA_HOME '\weka.jar']);

   %% Prepara DS1
    data_train = p1d;
    targets = p1t;

    % formata para uso no weka
    for t=2:size(targets,2)
        targets(find(targets(:,t)==1),t)=t;     
    end
    target_train = sum(targets');

    data = [data_train target_train'];

    %# attribute names
    numAttr = size(data,2);
    attributeNames=[];
    for i=1:numAttr
        str = int2str(i);
        str = strcat('attrib',str);
        attributeNames{i} = str;
        %attributeNames(i)= strcat('attrib',str);
    end
    %attributeNames = arrayfun(@(k) 'attr'+char(0:numAttr-1), 0:numAttr-1, 'Uni',false);

    wekaOBJ = matlab2weka('ecgTrain', attributeNames, data);
    if (~isempty(featSel))
        %numfs = size(featSel,2);
        numfs = size(find(featSel==1),2);
        folder_ds1 = ['.\WEKA\' featureSet '_FS' num2str(numfs) '_' char(registers(k)) '_AAMI2_train.arff'];
    else
        folder_ds1 = ['.\WEKA\' featureSet '_' char(registers(k)) '_AAMI2_train.arff'];
    end
    saveARFF(folder_ds1,wekaOBJ);

    %% prepara DS2
    data = [];
    data_test = p2d;
    targets_t = p2t;

    % formata para uso no weka
    for t=2:size(targets_t,2)
        targets_t(find(targets_t(:,t)==1),t)=t;     
    end
    target_test = sum(targets_t');

    data = [data_test target_test'];

    wekaOBJ = matlab2weka('ecgTest', attributeNames, data);
    if (~isempty(featSel))
        %numfs = size(featSel,2);
        numfs = size(find(featSel==1),2);
        folder_ds2 = ['.\WEKA\' featureSet '_FS' num2str(numfs) '_' char(registers(k)) '_AAMI2_test.arff'];

    else
        folder_ds2 = ['.\WEKA\' featureSet '_' char(registers(k)) '_AAMI2_test.arff'];
    end
    saveARFF(folder_ds2',wekaOBJ);

    end

end

