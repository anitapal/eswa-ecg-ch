% faz um undersampling dos dados
% Aplica o a funao para todas as classes
% percent : percentual para oversampling
% Autor: Eduardo Luz
function [newData newTarget] = undersampling_all_class(percent, data, target)

if(percent>1)
    percent=percent/100;

newData=data;
newTarget=target;

for t=1:size(target,2)
% separa classe t
index_class = find(target(:,t) ~=0);
classData = data(index_class,:);
labelForDataClass = target(index_class,:);

z = randperm(round(size(classData,1)/1)); 
randClassData = classData(z,:);


%separa outras classes
index_other_class = find(target(:,1)~=1);
otherClassData = data(index_other_class,:);
labelForDataOtherClass = target(index_other_class,:);

%if(percent<1)
newSize = round(size(randClassData,1)*percent);
newData = randClassData(1:newSize,:);
newData = [otherClassData;newData];
newTarget = [labelForDataOtherClass;labelForDataClass(1:newSize,:)];
end

% embaralha os dados
zr = randperm(round(size(newData,1)/1)); 
newData = newData(zr,:);
newTarget = newTarget(zr,:);

end




