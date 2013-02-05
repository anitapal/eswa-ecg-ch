% Normaliza conjunto de treino e conjunto de teste com baso no de treino

function [train_norm test_norm] = normaliza(trainset, testset)
% normalizacao
med_test = mean(trainset,1);
std_test = std(trainset,0,1);;
std_test(find(std_test==0))=1;
mean_trainset = mean(trainset,1);
std_trainset = std(trainset,0,1);
std_trainset(find(std_trainset==0))=1;
mean_trainset = repmat(mean_trainset,[size(trainset,1) 1]);
std_trainset = repmat(std_trainset,[size(trainset,1) 1]);
train_norm = trainset - mean_trainset;
train_norm = trainset./std_trainset;

train_norm(find(trainset==0))=1;

mean_testset = repmat(med_test,[size(testset,1) 1]);
std_testset = repmat(std_test,[size(testset,1) 1]);
test_norm = testset - mean_testset;
test_norm = testset./std_testset;

test_norm(find(testset==0))=1;

end
