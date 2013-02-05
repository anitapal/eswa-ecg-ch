%% Demo de uso do c�digo para classifica��o de arritmias
% Autor: Eduardo Luz
%
% Aten��o: classify_lopo_ds1(..) pode demorar algumas horas para terminar!
% Para fazer um teste r�pido, utilize apenas classify_test_ds2(..)
%
% As fun��es classify_lopo_ds1(..) e classify_test_ds2(..) cont�m todos os
% scripts para gerar as estat�sticas! O c�digo est� bem comentado e os
% arquivos .tex v�o ser colocados na paste results, por exemplo : 'YuCN_SVM__test_ds2_results_FS_33'
%

%% Sele��o de caracter�sticas
% Rodar o m�todo para sele��o de caracter�sticas, por exemplo:
%fs=sequencialfsSVM('YuCN'); 
%fs = [0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1] ; % as duas primeiras caracter�sticas v�o ser exclu�das

% O retorno da fun��o deve ser um vetor, onde cada posi��o corresponde a
% uma caracter�stica. Zero para ignorar a caracter�stica e 1 para usa-la.

%fs =  [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1] ; % sele��o de todas as caracter�sticas

fs =  [1 1 1 1 1 1 1 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0] ; % apenas as features do Yu + RRrelativo

%% Classifica e gera os arquivos .tex no diret�rio -> results

%classify_lopo_ds1(featureSet, featSel, classifier, C, gamma)
% Classifica��o de Arritmia com valida��o cruzada 22-folds, por registro.
%
% >> featureSet: conjunto de caracter�sticas
% Pode assumir os valores:
% 'YuCN','YuFourier', '2005Chazal', '2005Guler', 2005Song, 2006Yu, 2007Yu, 2010YeCoimbra, VCGComplexNet
%
% >> featSel: vetor para sele��o de caracter�sticas
% Deve ser no formato: fs = [1 1 0 0 0 1 0 0 1], da dimens�o das
% caracter�sticas onde:
% 1 - usar a caracter�stica
% 0 - ignorar a caracter�stica
% passe um vetor vazio para utilizar todas as caracter�sticas
%
% >> classifier: classificador a ser utilizado
% Pode assumir os valores:
%
% 'SVM', 'LD', 'MLP'
%
% C: par�metro para SVM
% 0: escolhe default (0.05)
% -1: utiliza script para grid selection

% gamma: par�metro para SVM
% 0: escolhe default (1/8*num features)

% EXEMPLO: entre com a pasta onde as caracter�sticas foram extra�das (dentro do diretorio features\), 
% o vetor de sele��o de caracter�sticas (fs) e o classificador (SVM , LD ou MLP)

%classify_lopo_ds1('YuCN', fs, 'SVM')

% Par�metro para SVM 
% para default -> C=0 e gamma=0
% para rodar script de grid selection -> C=-1
C = 0;
gamma = 0;

% EXEMPLO: 
classify_test_ds2('YuCN', fs, 'SVM', C, gamma);
