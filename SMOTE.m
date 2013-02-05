% Implementação do SMOTE
% Synthetic Minority Over-sampling Technique
%
% Input: Minority class index; Amount of SMOTE N%; 
% Number of nearest neighbors k; 
%
% Output: (N/100) * T synthetic minority class samples
%
% Autor: Eduardo Luz

function [newData newTarget] = SMOTE(origData, originalTarget, classIndex, N, k)

index = find(originalTarget(:,classIndex)==1);
T = origData(index,:);
Tlabel = originalTarget(index,:);

%(? If N is less than 100%, randomize the minority class samples as only a random percent of them will be SMOTEd. ?)
if(N<=100)
    z = randperm(round(size(T,1))); 
    T = T(z,:);

    randIndx = round(size(T,1)*(N/100));
    T = T(1:randIndx,:);
    
    N = 1;
else
    v = ceil(N/100);
    
    newT = [];
    for i=1:v
        newT = [newT;T];
    end
    
    z = randperm(round(size(newT,1))); 
    newT = newT(z,:);
    randIndx = round(size(newT,1)*N/100/v);
    T = newT(1:randIndx,:);
    
    N = 1;
end


%newindex=0; % keeps a count of number of synthetic samples generated, initialized to 0

%Synthetic=zeros(size(T,1)*N, size(T,2)); % array for synthetic samples

Synthetic = [];
%(? Compute k nearest neighbors for each minority class sample only. ?)
for i=1:size(T,1)
    %Compute k nearest neighbors for i, and save the indices in the nnarray
    nnarray = knnsearch(T(i,:),T,k);
    Synthetic(i,:) = populate(N, i, nnarray, T,k);
end

index = find(originalTarget(:,classIndex)==1);
labelForSyntectic = originalTarget(index(1),:);% pega o formato do target
t = size(Synthetic,1);
synteticTarget = repmat(labelForSyntectic,t,1);

originalTarget = [originalTarget;synteticTarget];
newData = [origData;Synthetic];

% embaralha os dados
zr = randperm(round(size(newData,1)/1)); 
newData = newData(zr,:);
newTarget = originalTarget(zr,:);

end

function [newIndividual] = populate(N, newIndex, nnarray, T,k)

Synthetic=[]; % array for synthetic samples
numattrs=size(T,2); % Number of attributes

while N > 0
    %Choose a random number between 1 and k, call it nn. This step chooses one of the k nearest neighbors of i.
    nn = randi(k,1);
    
    for attr=1:numattrs
        %Compute: dif = Sample[nnarray[nn]][attr] - Sample[i][attr] 
        dif = T(nnarray(nn),attr) - T(newIndex,attr);
        %Compute: gap = random number between 0 and 1 
        gap = rand(1,1);
        %Synthetic[newindex][attr] = Sample[i][attr] + gap ? dif
        %Synthetic(newIndex,attr) = T(newIndex,attr) + gap*dif;
        newIndividual(1,attr) = T(newIndex,attr) + gap*dif;
    end
    
    N = N -1;
end

end
