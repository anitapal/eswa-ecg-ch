%% Demo de uso do código para classificação com SVM
% Autor: Eduardo Luz
%
% ** As matrizes nos arquivos .mat (trainMatrix e testMatrix) estão organizados por linha. 
%    Cada linha é uma instância e na última coluna se encontra o atributo classe.
%
% ** Obs.: Para usar o svmtrain e o svmpredict da libSVM, é preciso ter os
% arquivos .mexw32 no diretório ou no PATH.

% Atenção : Usar saveMAT(featureFolderName) para gerar as matrizes!!!

%% Carregue os arquivos referentes ao método
fprintf('\nCarregando os arquivos...\n');

load('.\MAT\2007Yu\DS1-train.mat');
load('.\MAT\2007Yu\DS2-valid.mat');

% * ATENÇÃO : As características NÃO estão normalizadas!!! Use a normalização
% que julgar melhor

% normalização 1 : X - mean(X) / std(X)
%[dados1 dados2] = normaliza(dados1, dados2);

% normalização 2 : mapmaxmin - coloca tudo entre -1 e 1 (melhores resultados para SVM!)
[train, scale_factor] = mapminmax(trainMatrix(:,1:end-1)');   
test = mapminmax('apply',testMatrix(:,1:end-1)',scale_factor);

trainMatrix(:,1:end-1) = train';
testMatrix(:,1:end-1) = test';

%% utilze algum classificador, por exemplo o SVM

% parâmetros para o SVM
%-s svm_type : set type of SVM (default 0)
%	0 -- C-SVC%
%	1 -- nu-SVC
%	2 -- one-class SVM
%	3 -- epsilon-SVR
%	4 -- nu-SVR
% -t kernel_type : set type of kernel function (default 2)
%	0 -- linear: u'*v
%	1 -- polynomial: (gamma*u'*v + coef0)^degree%
%	2 -- radial basis function: exp(-gamma*|u-v|^2)%
%	3 -- sigmoid: tanh(gamma*u'*v + coef0)
%-d degree : set degree in kernel function (default 3)
%-g gamma : set gamma in kernel function (default 1/num_features)
%-r coef0 : set coef0 in kernel function (default 0)
%-c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)
%-n nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)
%-p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)
%-m cachesize : set cache memory size in MB (default 100)
%-e epsilon : set tolerance of termination criterion (default 0.001)
%-h shrinking: whether to use the shrinking heuristics, 0 or 1 (default 1)
%-b probability_estimates: whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)
%-wi weight: set the parameter C of class i to weight*C, for C-SVC (default 1)
%
%The k in the -g option means the number of attributes in the input data.

% Estamos utilizando kernel RBF, estamos usando valores diferentes de C
% para cada classe (-wi weight), on as classes minoritárias estão recebendo
% valores de C maiores...isso significa uma margem menor (modelo menos
% genérico). Os valores dos pesos usados aqui são mais ou menos proporcionais ao número de
% inst6ancias de cada classe.
fprintf('\n Treinando o classificador, aguarde...\n');

C = 0.5;
gamma = 1/(8*size(trainMatrix(:,1:end-1),2));

str = sprintf('-w1 1 -w2 50 -w3 10 -w4 100 -c %d -t 2 -g %f -d 4',C,gamma);

cd('svm')
model = svmtrain(trainMatrix(:,end), trainMatrix(:,1:end-1), str);
cd ..

%% faz a predição e constroi a matriz de  confuzão
[predict_label, accuracy, dec_values] = svmpredict(testMatrix(:,end), testMatrix(:,1:end-1), model); % test the training data
cm = confusionmat(testMatrix(:,end),predict_label);
cm

%% Calcula estatíSticas
   
   acc_num=0;
   acc_den=0;
   den1=0;
   den2=0;
   num=0;
   t=0;
   
    if(size(cm,1)>=1)
       t = 1;
       
       num = cm(t,t);
       den1 = sum(cm(t,:));
       den2 = sum(cm(:,t));
        
       TN = sum(sum(cm(:,:))) - den1 - den2 + cm(t,t);
       FP = den2 - cm(t,t);
       
        if(den1~=0)
            sensitivityN = (num/den1) * 100;
        else
            sensitivityN = -1;
        end
        
        if(den2~=0)
            specificityN = (num/den2) * 100;
        else
            specificityN = 0;
        end
        
        if(TN + FP > 0)
            FPR_N = 100*FP/(TN+FP);
        else
            FPR_N=0;
        end
        
        acc_num = acc_num + num;
        acc_den = acc_den + den1;
        
    else
        sensitivityN = -1;
        specificityN = -1;
        FPR_N=-1;
    end
   
    % caso especial para LD classifier
    %size(cm,1)==5
    
    if(size(cm,1)>=2)
       t = 2;
     
       num = cm(t,t);
        den1 = sum(cm(t,:));
        den2 = sum(cm(:,t));
        
        TN = sum(sum(cm(:,:))) - den1 - den2 + cm(t,t);
        FP = den2 - cm(t,t);
        
        if(den1~=0)
            sensitivitySVEB = (num/den1) * 100;
        else
            sensitivitySVEB = -1;
        end
        
        if(den2~=0)
            specificitySVEB = (num/den2) * 100;
        else
            specificitySVEB = -1;
        end
        
        if(TN + FP > 0)
            FPR_SVEB = 100*FP/(TN+FP);
        else
            FPR_SVEB=-1;
        end
        
        acc_num = acc_num + num;
        acc_den = acc_den + den1;
        
    else
        sensitivitySVEB = -1;
        specificitySVEB = -1;
        FPR_SVEB=-1;
    end
   
    if(size(cm,1)>=3)
       t = 3;
        num = cm(t,t);
        den1 = sum(cm(t,:));
        den2 = sum(cm(:,t));
        
        TN = sum(sum(cm(:,:))) - den1 - den2 + cm(t,t);
        FP = den2 - cm(t,t);
        
        if(den1~=0)
            sensitivityVEB = (num/den1) * 100;
        else
            sensitivityVEB = -1;
        end
        
        if(den2~=0)
            specificityVEB = (num/den2) * 100;
        else
            specificityVEB = -1;
        end
        
        if(TN + FP > 0)
            FPR_VEB = 100*FP/(TN+FP);
        else
            FPR_VEB=-1;
        end
        
        acc_num = acc_num + num;
        acc_den = acc_den + den1;
        
    else
        sensitivityVEB = -1;
        specificityVEB = -1;
        FPR_VEB=-1;
    end
   
    if(size(cm,1)>=4)
       t = 4;
       
       num = cm(t,t);
        den1 = sum(cm(t,:));
        den2 = sum(cm(:,t));
        
        TN = sum(sum(cm(:,:))) - den1 - den2 + cm(t,t);
        FP = den2 - cm(t,t);
        
        if(den1~=0)
            sensitivityF = (num/den1) * 100;
        else
            sensitivityF = -1;
        end
        if(den2~=0)
        specificityF = (num/den2) * 100;
        else
            specificityF = -1;
        end
        
        if(TN + FP > 0)
            FPR_F = 100*FP/(TN+FP);
        else
            FPR_F=-1;
        end
        
        acc_num = acc_num + num;
        acc_den = acc_den + den1;
        
    else
        sensitivityF = -1;
        specificityF = -1;
        FPR_F=-1;
    end
   
    if(size(cm,1)>=5)
       t = 5;
       
        num = cm(t,t);
        den1 = sum(cm(t,:));
        den2 = sum(cm(:,t));
        
        TN = sum(sum(cm(:,:))) - den1 - den2 + cm(t,t);
        FP = den2 - cm(t,t);
        
        if(den1~=0)
            sensitivityQ = (num/den1) * 100;
        else
            sensitivityQ = -1;
        end
        
        if(den2~=0)
            specificityQ = (num/den2) * 100;
        else
            specificityQ = -1;
        end
        
        if(TN + FP > 0)
            FPR_Q = 100*FP/(TN+FP);
        else
            FPR_Q=-1;
        end
        
        acc_num = acc_num + num;
        acc_den = acc_den + den1;
        
    else
        sensitivityQ = -1;
        specificityQ = -1;
        FPR_Q=-1;
    end
    
    fprintf('\n---- RESULTADO -----\n');
    fprintf('Teste em DS2 | Acc = %6.1f | N = %6.1f/%6.1f/%6.1f | SVEB = %6.1f/%6.1f/%6.1f | VEB = %6.1f/%6.1f/%6.1f & F = %6.1f/%6.1f/%6.1f \n',...
        100*acc_num/acc_den, sensitivityN,specificityN,FPR_N,sensitivitySVEB,specificitySVEB,FPR_SVEB,sensitivityVEB,specificityVEB,FPR_VEB,...
        sensitivityF,specificityF,FPR_F);         
    fprintf('\n');
    
