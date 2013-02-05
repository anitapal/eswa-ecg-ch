function [fs,history]=sequencialfsSVM_yu(featureSet, modelo)

path(path,'svm')

s = char(featureSet);
fileNamed = ['results\',s,'_FeatureSelection.tex'];
arq = fopen(fileNamed,'w');

feat_folder = ['features\' featureSet '\'];

DS1 = {'101';'106';'108';'109';'112';'114';'115';'116';'118';'119';'122';'124';'201';'203';'205';'207';'208';'209';'215';'220';'223';'230'};
DS2 = {'100';'103';'105';'111';'113';'117';'121';'123';'200';'202';'210';'212';'213';'214';'219';'221';'222';'228';'231';'232';'233';'234'};   

for j=1:size(DS1,1)
    train_ds1(j) = str2double(DS1(j));
end

for j=1:size(DS1,1)
    test_ds2(j) = str2double(DS2(j));
end


registers = {'101';'106';'108';'109';'112';'114';'115';'116';'118';'119';'122';'124';'201';'203';'205';'207';'208';'209';'215';'220';'223';'230'};
%registers = {'108';'109';'112';'114';};

% imprime a tabela de registros em arquvo .tex
%printRegisterTable(['features\' featureSet '\']);

fprintf('\n Gerando os modelos... \n')
tic
mdl=[];
scale_fac = [];

for k=1:size(registers,1) % numero de registros
    
    fprintf('\n Gera modelo %d ... \n',k)
    
    test_ds = [];
    train_ds = [];
    test_ds = str2double(registers(k));
    count=1;
    
    for j=1:size(registers,1)
        if j ~= k
            train_ds(count) = str2double(registers(j));
            count = count +1;
        end
    end
    
    %carrega os dados
    %p1d = instancias
    %p1t = label do treino
    [p1d p1t p2d p2t] = loadDataAAMI(0,feat_folder,train_ds,test_ds);
    %[p1d p1t] = undersampling_all_class(10, p1d,p1t);

    [p1d, scale_factor] = mapminmax(p1d');   
    p2d = mapminmax('apply',p2d',scale_factor);
    p1d = p1d';
    p2d = p2d';
    
    scale_fac{k}=scale_factor;
    
    for t=2:size(p1t,2)
        p1t(find(p1t(:,t)==1),t)=t;     
    end
    p1t = sum(p1t');
    p1t = p1t';
    
    if nargin < 2  % apenas gera modelos se nenhum foi passado por parâmetro
         cd svm
        % cria os modelos (1 pra cada paciente)
        % parametros para SVM
        param_c=0.05;
        param_g=1/(8 * size(p2t,2));
        cmd = ['-w1 1 -w2 50 -w3 10 -c ',num2str(param_c),' -g ',num2str(param_g)]; 
        mdl{k} = svmtrain(p1t,p1d,cmd);
        cd ..
    end
end

%{
% modelo de DS1 para ser aplicado a DS2
[p1d p1t p2d p2t] = loadDataAAMI(0,feat_folder,train_ds,test_ds);
%[p1d p1t] = undersampling_all_class(10, p1d,p1t);

[p1d, scale_factor] = mapminmax(p1d');   
p2d = mapminmax('apply',p2d',scale_factor);
p1d = p1d';
p2d = p2d';
    
for t=2:size(p1t,2)
   p1t(find(p1t(:,t)==1),t)=t;     
end
p1t = sum(p1t');
p1t = p1t';
    
cd svm
fprintf('\n Gera modelo DS1 ... \n')
mdl{k+1}=svmtrain(p1t,p1d,cmd);
cd ..
%}

if nargin < 2  % apenas gera modelos se nenhum foi passado por parâmetro
    save mdl
else
    mdl = modelo;
end

toc


fprintf('\n passo 1. \n')

%vetor para indices das caracteristicas que vao ficar
keep = zeros(1,size(p2d,2));
keep(1,1:9)=1; % inicializa com as nove features de yu

sweep = zeros(1,size(p2d,2));
best_in  = [];
best_acc = [];
best_acc_mean = 0;
%começo com 1 caraceristica

fprintf(arq,'------------------------------------------------\n');
fprintf(arq,'\nBusca para frente: \n');
fprintf('------------------------------------------------\n');
fprintf('\nBusca para frente: \n');

acc = zeros(size(registers,1),size(p2d,2));
    
%executa o processo de acordo com o número de caracteristicas
for i=1:size(p2d,2)
    fprintf(arq,'\n Passo %d ... \n',i)
    fprintf('\n Passo %d ... \n',i)
    tic
    
    %varre para todas as caracteristicas
    for j=1:size(p1d,2)
        if keep(1,j) ~= 1 || best_acc_mean == 0
            sweep = keep;
            sweep(1,j) = 1;
       % end
        
            %testa para todos os pacientes
            for k=1:size(registers,1)

                train_ds = [];
                count=1;
                for ll=1:size(registers,1)
                    if ll ~= k
                        train_ds(count) = str2double(registers(ll));
                        count = count +1;
                    end
                end

                test_ds = str2double(registers(k));
                [p1d p1t p2d p2t] = loadDataAAMI(0,feat_folder,train_ds,test_ds);
                %[p1d p1t p2d p2t] = loadDataAAMI(0,feat_folder,[],test_ds);

                %coloca valor médio nas colunas em que sweep estava zero 
                p2d(:,find(sweep==0)) = repmat(mean(p1d(:,find(sweep==0))),size(p2d,1),1 );

                %[p1d, scale_factor] = mapminmax(p1d');   
                p2d = mapminmax('apply',p2d',scale_fac{k});
                %p1d = p1d';
                p2d = p2d';

                %{
                for t=2:size(p1t,2)
                   p1t(find(p1t(:,t)==1),t)=t;     
                end
                p1t = sum(p1t');
                p1t = p1t';
                %}

                for t=2:size(p2t,2)
                   p2t(find(p2t(:,t)==1),t)=t;     
                end
                p2t = sum(p2t');
                p2t = p2t';

                acc(k,j) = evaluation(p2d,p2t,mdl{k});
                %fprintf('Acuracia para registro %d %x')
            end    
        
        else
            acc(k,j)=best_acc_mean;
        end
    end
    
    % acha a maior acurácia média para todos os pacientes
    acc_mean = mean(acc,1);
    index = find(acc_mean == max(acc_mean));
    
    if(acc_mean(index(1)) > best_acc_mean)
        best_acc_mean = acc_mean(index(1));
        % atualiza keep
        fprintf(arq,'\n **atualiza fs - Melhor configuração da rodada (acc medio=%f)\n fs =',acc_mean(index(1)));
        fprintf('\n **atualiza fs - Melhor configuração da rodada (acc medio=%f)\n fs =',acc_mean(index(1)));
        keep(1,index(1))=1; % mantem a melhor configuração
        for uu=1:size(keep,2)
            fprintf(arq,'%1d ',keep(1,uu));
        end
        for uu=1:size(keep,2)
            fprintf('%1d ',keep(1,uu));
        end
    
    else
        fprintf(arq,'\n Inclusão de novas caracteristicas não melhorou o resultado!\n Processo terminado. (acc medio=%f)\n fs =',acc_mean(index(1)));
        fprintf('\n Inclusão de novas caracteristicas não melhorou o resultado!\n Processo terminado. (acc medio=%f)\n fs =',acc_mean(index(1)));
        for uu=1:size(keep,2)
        fprintf(arq,'%1d ',keep(1,uu));
       end
       for uu=1:size(keep,2)
            fprintf('%1d ',keep(1,uu));
       end
       
       i=size(p2d,2); % força o término
       %fclose(arq);
        break;
        break;
    end
 
    toc
end

fprintf(arq,'\n\n Melhor configuração final \n');
fprintf(arq,'fs = ');
for uu=1:size(keep,2)
    fprintf(arq,'%1d ',keep(1,uu));
end

fprintf('\n\n Melhor configuração final \n');
fprintf('fs = ');
for uu=1:size(keep,2)
    fprintf('%1d ',keep(1,uu));
end
fprintf('\n');

fclose(arq);
end

function cv =evaluation(inst, label, mdl)

wa = @(e1,e2,e3,s1,s2,s3)(1-(e1*1/s1+e2*1/s2+e3*1/s3)/3)*100; %weighted accuracy

if max(unique(label)) <=1
    label(1)=2;
end

cp = classperf(label);

[class] = svmpredict(label,inst,mdl);
classperf(cp,class);

if(size(cp.ClassLabels,1) >= 3)
    cv = wa(cp.errorDistributionByClass(1), ...
    cp.errorDistributionByClass(2), ...
    cp.errorDistributionByClass(3), ...
    cp.SampleDistributionByClass(1),...
    cp.SampleDistributionByClass(2),...
    cp.SampleDistributionByClass(3));  
else
    if(size(cp.ClassLabels,1) >= 2)
        cv = wa(cp.errorDistributionByClass(1), ...
        cp.errorDistributionByClass(2), ...
        0, ...
        cp.SampleDistributionByClass(1),...
        cp.SampleDistributionByClass(2),...
        1); 
    else
        cv = wa(cp.errorDistributionByClass(1), ...
        0, ...
        0, ...
        cp.SampleDistributionByClass(1),...
        1,...
        1); 
    end
end
    
end