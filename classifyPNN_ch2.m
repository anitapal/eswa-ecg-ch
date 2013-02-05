% Autor : Eduardo Luz
% Classifica utilizando Redes Neuronais probabil�sticas
% % Baseado no paper Yu e Chen,2007
%
% numTests : N�mero de testes a ser executado. 
% (Testa numTests vezes , faz m�dia e desvio padr�o)
%
% P : dados de entrada, para treino.
% T : R�tulo do grupo de cada padr�o, para treino
% testIn : dados de entrada para teste
% testIn : r�tulos dos grupos para teste
% spread : Spread of radial basis functions (default = 0.1)

function [cm net predict_label testTarget] = classifyPNN_ch2(P,T,testIn,testTarget,spread)

% Faz um shuffle dos dados 9aparentemente o shuffle n�o faz diferen�a alguma
% a fun��o pnn(.) deve implementar este passo
%[P T] = shuffleVectors(P,T);
%[testIn testTarget] = shuffleVectors(testIn,testTarget);

%transforma a matriz em um vetor de �ndices
for t=2:size(T,2)
    T(find(T(:,t)==1),t)=t;     
end
tipos = sum(T');

% Normaliza��o das carcter�sticas - ver p�gina 1144 paper Yu e Chen,2007
% Obs.: Fizemos um teste sem normalizar as caracter�sticas e o resultado fica
% muito ruim. Esta rede � muito sens�vel a isto, diferentemento das MLP
% (usando newff ou newpr)
for(i=1:length(P(:,1)))
    for(j=1:length(P(1,:)))
        P(i,j) = tansig((P(i,j)-mean(P(i,:)))/std(P(i,:)));     
    end
end

for(i=1:length(testIn(:,1)))
    for(j=1:length(testIn(1,:)))
        %normaliza tb a matriz de teste
        testIn(i,j) = tansig((testIn(i,j)-mean(testIn(i,:)))/std(testIn(i,:)));
    end
end

%cria a rede
tipos = ind2vec(tipos');
net = newpnn(P',tipos,spread);

%simula com os dados para treino
%Y1 = sim(net,P');

%simula com os dados para teste
predict_label = sim(net,testIn');

% Treino ----------------------------------------------------------
%transforma a matriz em um vetor de �ndices, para aplicar confusionmat(..)
%Y1 = vec2ind(Y1);
%cria a matriz de confus�o
%[Ctrain,ordert] = confusionmat(tipos',Y1);
%totalAccuracytrain=0;

%for i=1:size(Ctrain,2) % PARA TODAS AS CLASSES
%    totalAccuracytrain = totalAccuracytrain + Ctrain(i,i);
%end
%totalAccuracytrain = (totalAccuracytrain * 100) / size(P,1);

% Teste -----------------------------------------------------------
%transforma a matriz em um vetor de �ndices, para aplicar confusionmat(..)
%predict_label = vec2ind(predict_label);

%[C,CM,IND,PER] = CONFUSION(TARGETS,OUTPUTS)
[c cm] = confusion(testTarget', predict_label);

if size(cm,1)<4
    cm_N = zeros(4,4);
    for i=1:4
        for j=1:4
            if i <= size(cm,1) && j <= size(cm,2)
                cm_N(i,j) = cm(i,j);
            end
        end
    end
    
    cm = cm_N;
end

end
