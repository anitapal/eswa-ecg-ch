%% Demo de uso do código para classificação de arritmias
% Autor: Eduardo Luz
%
% Atenção: classify_lopo_ds1(..) pode demorar algumas horas para terminar!
% Para fazer um teste rápido, utilize apenas classify_test_ds2(..)
%
% As funções classify_lopo_ds1(..) e classify_test_ds2(..) contém todos os
% scripts para gerar as estatísticas! O código está bem comentado e os
% arquivos .tex vão ser colocados na paste results, por exemplo : 'YuCN_SVM__test_ds2_results_FS_33'
%

%% Seleção de características
% Rodar o método para seleção de características, por exemplo:
%fs=sequencialfsSVM('YuCN'); 
%fs = [0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1] ; % as duas primeiras características vão ser excluídas

% O retorno da função deve ser um vetor, onde cada posição corresponde a
% uma característica. Zero para ignorar a característica e 1 para usa-la.

%fs =  [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1] ; % seleção de todas as características

fs =  [1 1 1 1 1 1 1 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0] ; % apenas as features do Yu + RRrelativo

%% Classifica e gera os arquivos .tex no diretório -> results

%classify_lopo_ds1(featureSet, featSel, classifier, C, gamma)
% Classificação de Arritmia com validação cruzada 22-folds, por registro.
%
% >> featureSet: conjunto de características
% Pode assumir os valores:
% 'YuCN','YuFourier', '2005Chazal', '2005Guler', 2005Song, 2006Yu, 2007Yu, 2010YeCoimbra, VCGComplexNet
%
% >> featSel: vetor para seleção de características
% Deve ser no formato: fs = [1 1 0 0 0 1 0 0 1], da dimensão das
% características onde:
% 1 - usar a característica
% 0 - ignorar a característica
% passe um vetor vazio para utilizar todas as características
%
% >> classifier: classificador a ser utilizado
% Pode assumir os valores:
%
% 'SVM', 'LD', 'MLP'
%
% C: parâmetro para SVM
% 0: escolhe default (0.05)
% -1: utiliza script para grid selection

% gamma: parâmetro para SVM
% 0: escolhe default (1/8*num features)

% EXEMPLO: entre com a pasta onde as características foram extraídas (dentro do diretorio features\), 
% o vetor de seleção de características (fs) e o classificador (SVM , LD ou MLP)

%classify_lopo_ds1('YuCN', fs, 'SVM')

% Parâmetro para SVM 
% para default -> C=0 e gamma=0
% para rodar script de grid selection -> C=-1
C = 0;
gamma = 0;

% EXEMPLO: 
classify_test_ds2('YuCN', fs, 'SVM', C, gamma);
