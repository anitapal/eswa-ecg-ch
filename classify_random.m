% Classifica��o de Arritmias
%
% type : 'AAMI' ou 'AAMI2'
%
% featureSet: conjunto de caracter�sticas
% Pode assimir os valores:
%
% '2005Chazal', '2005Guler', 2005Song, 2006Yu, 2007Yu, 2010YeCoimbra, VCGComplexNet
%
% classifier: classificador a ser utilizado
% Pode assumir os valores:
%
% 'SVM', 'MLP', 'PNN', 'LD'
%
% Autor: Eduardo Luz
%
% testar best c=128 g=0.0625 rate=55.3843%)
%

function classify_random(featureSet, classifier, type)

if nargin < 3
   type = 'AAMI';
end

s = char(featureSet);
fileNamed = ['results\',s,'_',classifier,'_random_results.tex'];
arq = fopen(fileNamed,'w');

% Tabela latex dos resultados
fprintf(arq,'\\documentclass{article}\n');
fprintf(arq,'\\usepackage{graphicx}\n');
fprintf(arq,'\\usepackage[latin1]{inputenc}\n');
fprintf(arq,'\\usepackage{tabularx}\n');
fprintf(arq,'\\usepackage{multirow}\n');
fprintf(arq,'\\newcommand{\\citep}{\\cite}\n');
fprintf(arq,'\\newcommand{\\citet}{\\cite}\n');
fprintf(arq,'\\newcommand{\\TFigure}{Fig.}\n');
fprintf(arq,'\\begin{document}\n');


fprintf(arq,'\n');
fprintf(arq,'\\begin{table*} \n');
fprintf(arq,'\\footnotesize \n');
s2 = char('\\caption{Tebela de resultados por paciente ');
s2 = [s2 s(10:end-1) '} \n'];
fprintf(arq,s2);
%fprintf(arq,' \\caption{Tebela dos registros do m�todo} \n');
fprintf(arq,' \\label{tab:regtable} \n');
fprintf(arq,' \\begin{center} \n');
fprintf(arq,'  \\begin{tabular}{|c|c|c|c|c|c|c|c|c|c|c|c|} \n');
fprintf(arq,'   \\hline \n');
fprintf(arq,'    Registro & Acc & N Se & SVEB Se & VEB Se & F Se & Q Se & N+P & SVEB+P & VEB+P & F+P & Q+P  \\\\ \n');
fprintf(arq,'   \\hline \n');

% iniciliza variaveis
n_tot=0;
s_tot=0;
v_tot=0;
f_tot=0;
q_tot=0;
n_line = 0;
s_line = 0;
v_line = 0;
f_line = 0;
q_line = 0;
n_col = 0;
s_col = 0;
v_col = 0;
f_col = 0;
q_col = 0;
finalCM = zeros(5,5);

% Inicializa os registros
registers = {'105';'100';'103';'111';'113';'117';'121';'123';'200';'202';'210';'212';'213';'214';'219';'221';'222';'228';'231';'232';'233';'234';'101';'106';'108';'109';'112';'114';'115';'116';'118';'119';'122';'124';'201';'203';'205';'207';'208';'209';'215';'220';'223';'230'};
%DS1 = {'101';'106';'108';'109';'112';'114';'115';'116';'118';'119';'122';'124';'201';'203';'205';'207';'208';'209';'215';'220';'223';'230'};
%DS2 = {'100';'103';'105';'111';'113';'117';'121';'123';'200';'202';'210';'212';'213';'214';'219';'221';'222';'228';'231';'232';'233';'234'};   

% imprime a tabela de registros em arquvo .tex
%printRegisterTable(['features\' featureSet '\']);

fprintf('\n Registro | Acc | N_Se SVEB_Se VEB_Se N+P SVEB+P VEB+P \n')

test_ds = [];
train_ds = [];
    
for j=1:size(registers,1)
    train_ds(j) = str2double(registers(j));
end

first_time=1;
    
for k=1:size(registers,1) % numero de registros
    test_ds = str2double(registers(k));
       
%% Carrega os dados
if(strcmp(featureSet,'2004Chazal'))
    feat_folder = ['features\' featureSet '\'];
    %cd(feat_folder)
    
    % j� esta sendo feita normalizacao dentro da fun��o : 
    % Xnorm = X - media_treino / std_treino

    [fs_1 fs_2 fs_3 fs_4 fs_5 fs_6 fs_7 fs_8 target] = loadData_random_chazal(feat_folder, train_ds,test_ds);
    
			 %primeira etapa com DS1 treino DS2 teste
			 % normaliza
			 [fs_1.train, scale_factor] = mapminmax(fs_1.train');   
			 fs_1.test = mapminmax('apply',fs_1.test',scale_factor);
								   
								   [fs_5.train, scale_factor] = mapminmax(fs_5.train');   
			 fs_5.test = mapminmax('apply',fs_5.test',scale_factor);
								   
								   fs_1.train = fs_1.train';
								   fs_1.test = fs_1.test';
								   
								   fs_5.train = fs_5.train';
								   fs_5.test = fs_5.test';
								   
								   fs1.train = [fs_1.train fs_5.train];
								   fs1.test = [fs_1.test fs_5.test];

    %cd ..
    %cd ..
else
    feat_folder = ['features\' featureSet '\'];
    %cd(feat_folder)
    
    % j� esta sendo feita normalizacao dentro da fun��o : 
    % Xnorm = X - media_treino / std_treino

    if(strcmp(type,'AAMI2'))
        [p1d p1t p2d p2t] = loadDataAAMI2(0,feat_folder,train_ds,test_ds);
    else
        [p1d p1t p2d p2t] = loadDataAAMI(0,feat_folder,train_ds,test_ds);
    end

								   %primeira etapa com DS1 treino DS2 teste
								   fs1.train = p1d;
								   fs1.test = p2d;
								   target.train = p1t;
								   target.test = p2t;
								   
								   [fs1.train, scale_factor] = mapminmax(fs1.train');   
			 fs1.test = mapminmax('apply',fs1.test',scale_factor);
								  
								  fs1.train = fs1.train';
								  fs1.test = fs1.test';
    
    %cd ..
    %cd .. 
end

%% aplica o classificador
if(strcmp(classifier,'LD'))
    cd('LD_classifier')

    if(strcmp(featureSet,'2004Chazal'))
        cm1 = ld_Classifier_chazal(fs_1,fs_5,target);
    else
        cm1 = ld_Classifier(fs1,target);
    end
    
    %fprintf('\n----------------------- Classificador LD ---------------------\n')
    cd ..
elseif(strcmp(classifier,'SVM'))
    cd('svm')
    
    %[best_c,best_g,best_cv,hC] = parameter_optimization(fs1.train, target.train);
    %best_c=128;
    best_c=1;
    best_g=0.0625;
    
    %[newData newTarget] = unedersampling_class1(5, fs1.train,target.train);
    %[best_c,best_g,best_cv,hC] = parameter_optimization(newData,newTarget);
    tic
    
    if(first_time==1)
        first_time=0;
        [cm1 model] = svm_Classifier(fs1.train,target.train,fs1.test,target.test);
    else
        testingLabels = [];
        targetsTest = [];
        % formata para uso em SVM - para cada classe
        for t=1:size(target.test,2)
            targetsTest(find(target.test(:,t)==1),t)=t;     
        end
        testingLabels = sum(targetsTest');
        
        [predict_label, accuracy, dec_values] = svmpredict(testingLabels', fs1.test, model); % test the training data
        %cm1 = confusionmat(testingLabels',predict_label);
        
			 clear predict_matrix;
        for i=1: size(predict_label,1)
            switch(predict_label(i))
                case 1,
                    predict_matrix(i,:) = [1 0 0 0 0];            
                case 2,
                    predict_matrix(i,:) = [0 1 0 0 0];            
                case 3,
                    predict_matrix(i,:) = [0 0 1 0 0];            
                case 4,
                    predict_matrix(i,:) = [0 0 0 1 0]; 
                case 5,
                    predict_matrix(i,:) = [0 0 0 0 1];
            end
            end

			 clear label_matrix;
            for i=1: size(testingLabels,2)
            switch(testingLabels(i))
                case 1,
                    label_matrix(i,:) = [1 0 0 0 0];            
                case 2,
                    label_matrix(i,:) = [0 1 0 0 0];            
                case 3,
                    label_matrix(i,:) = [0 0 1 0 0];            
                case 4,
                    label_matrix(i,:) = [0 0 0 1 0]; 
                case 5,
                    label_matrix(i,:) = [0 0 0 0 1];
            end
            end
            %[C,CM,IND,PER] = CONFUSION(TARGETS,OUTPUTS)
            [c cm1] = confusion(label_matrix', predict_matrix');
    end
    
    toc
    %fprintf('\n----------------------- Classificador SVM ---------------------\n')
    
    cd ..
%elseif(strcmp(classifier,'PNN'))
%    cd pnn
%    
%    [acc1 sensitivityN1 sensitivitySVEB1 sensitivityVEB1 sensitivityF1 sensitivityQ1 specificitySVEB1 specificityVEB1 specificityF1 specificityQ1] = pnn_Classifier(fs.train,target.train,fs.test,target.test);
%    [acc2 sensitivityN2 sensitivitySVEB2 sensitivityVEB2 sensitivityF2
%    sensitivityQ2 specificitySVEB2 specificityVEB2 specificityF2 specificityQ2] = svm_Classifier(fs.train,target.train,fs.test,target.test);%

%    cd ..
elseif(strcmp(classifier,'MLP'))
    cd('mlp')
    
    [cm1] = mlp_Classifier(fs1.train,target.train,fs1.test,target.test);
    
    %fprintf('\n----------------------- Classificador MLP comb ---------------------\n')
    
    cd ..
end

% n�o � t�o simpels assim calcular a MC
%for ii=1:size(cm1,1)
%    for jj=1:size(cm1,2)
%        finalCM(ii,jj) = finalCM(ii,jj)+cm1(ii,jj);
%    end
%end

% Calcula estat�Sticas
   cm1
   
   finalCM = finalCM + cm1;
   
   acc_num=0;
   acc_den=0;
   den1=0;
   den2=0;
   num=0;
   t=0;
   
    if(size(cm1,1)>=1)
       t = 1;
       
       num = cm1(t,t);
       den1 = sum(cm1(t,:));
       den2 = sum(cm1(:,t));
        
        if(den1~=0)
            sensitivityN = (num/den1) * 100;
        else
            sensitivityN = 0;
        end
        if(den2~=0)
            specificityN = (num/den2) * 100;
        else
            specificityN = 0;
        end
        acc_num = acc_num + num;
        acc_den = acc_den + den1;
        
    else
        sensitivityN = -1;
        specificityN = -1;
    end
   
    % caso especial para LD classifier
    %size(cm1,1)==5
    
    if(size(cm1,1)>=2)
       t = 2;
     
       num = cm1(t,t);
        den1 = sum(cm1(t,:));
        den2 = sum(cm1(:,t));
        
        if(den1~=0)
            sensitivitySVEB = (num/den1) * 100;
        else
            sensitivitySVEB = 0;
        end
        
        if(den2~=0)
            specificitySVEB = (num/den2) * 100;
        else
            specificitySVEB = 0;
        end
        acc_num = acc_num + num;
        acc_den = acc_den + den1;
        
    else
        sensitivitySVEB = -1;
        specificitySVEB = -1;
    end
   
    if(size(cm1,1)>=3)
       t = 3;
        num = cm1(t,t);
        den1 = sum(cm1(t,:));
        den2 = sum(cm1(:,t));
        if(den1~=0)
            sensitivityVEB = (num/den1) * 100;
        else
            sensitivityVEB = 0;
        end
        
        if(den2~=0)
            specificityVEB = (num/den2) * 100;
        else
            specificityVEB = 0;
        end
        acc_num = acc_num + num;
        acc_den = acc_den + den1;
        
    else
        sensitivityVEB = -1;
        specificityVEB = -1;
    end
   
    if(size(cm1,1)>=4)
       t = 4;
       
       num = cm1(t,t);
        den1 = sum(cm1(t,:));
        den2 = sum(cm1(:,t));
        if(den1~=0)
            sensitivityF = (num/den1) * 100;
        else
            sensitivityF = 0;
        end
        if(den2~=0)
        specificityF = (num/den2) * 100;
        else
            specificityF = 0;
        end
        acc_num = acc_num + num;
        acc_den = acc_den + den1;
        
    else
        sensitivityF = -1;
        specificityF = -1;
    end
   
    if(size(cm1,1)>=5)
       t = 5;
       
        num = cm1(t,t);
        den1 = sum(cm1(t,:));
        den2 = sum(cm1(:,t));
        if(den1~=0)
            sensitivityQ = (num/den1) * 100;
        else
            sensitivityQ = 0;
        end
        
        if(den2~=0)
            specificityQ = (num/den2) * 100;
        else
            specificityQ = 0;
        end
        acc_num = acc_num + num;
        acc_den = acc_den + den1;
        
    else
        sensitivityQ = -1;
        specificityQ = -1;
    end
   
    fprintf(arq,'%6d & %6.2f & %6.2f & %6.2f & %6.2f & %6.2f & %6.2f & %6.2f & %6.2f & %6.2f & %6.2f & %6.2f \\\\ \n',...
        str2double(registers(k)), 100*acc_num/acc_den, sensitivityN, sensitivitySVEB,sensitivityVEB, sensitivityF, sensitivityQ,...
        specificityN, specificitySVEB,specificityVEB, specificityF, specificityQ);         
    fprintf(arq,'\n');
    
   fprintf('\n');
    
   fprintf('Registro=%6.0f | Acc=%6.2f |', str2double(registers(k)), 100*acc_num/acc_den);
    
   fprintf(' N_Se=%6.2f SVEB_Se=%6.2f VEB_Se=%6.2f N+P=%6.2f SVEB+P=%6.2f VEB+P=%6.2f \n\n',...
       sensitivityN, sensitivitySVEB,sensitivityVEB,  specificityN, specificitySVEB,specificityVEB)
 
   
end % for t

fprintf(arq,'    \\hline \n');
%fprintf(arq,'    \\hline \n');

fprintf(arq, ' Gross & %6.2f & %6.2f & %6.2f & %6.2f & %6.2f & %6.2f & %6.2f & %6.2f & %6.2f & %6.2f & %6.2f \\\\ \n',...
       100*(finalCM(1,1)+finalCM(2,2)+finalCM(3,3)+finalCM(4,4)+finalCM(5,5))/(sum(finalCM(1,:))+sum(finalCM(2,:))+sum(finalCM(3,:))+sum(finalCM(4,:))+sum(finalCM(5,:))) ,...
       100*finalCM(1,1)/sum(finalCM(1,:)), 100*finalCM(2,2)/sum(finalCM(2,:)), 100*finalCM(3,3)/sum(finalCM(3,:)), 100*finalCM(4,4)/sum(finalCM(4,:)), 100*finalCM(5,5)/sum(finalCM(5,:)),...
       100*finalCM(1,1)/sum(finalCM(:,1)), 100*finalCM(2,2)/sum(finalCM(:,2)), 100*finalCM(3,3)/sum(finalCM(:,3)), 100*finalCM(4,4)/sum(finalCM(:,4)), 100*finalCM(5,5)/sum(finalCM(:,5)));


fprintf(arq,'   \\hline \n');
fprintf(arq,'  \\end{tabular} \n');
fprintf(arq,' \\end{center} \n');
fprintf(arq,'\\end{table*} \n');
fprintf(arq,'\n');
fprintf(arq,'\n');

fprintf(arq,'\n');
fprintf(arq,'\\begin{table*} \n');
fprintf(arq,' \\caption{Matriz e confus�o} \n');
fprintf(arq,' \\label{tab:regtable} \n');
fprintf(arq,' \\begin{center} \n');
fprintf(arq,'  \\begin{tabular}{|');
for tt=1:size(finalCM,1)
    fprintf(arq,'c|');
end
fprintf(arq,'} \n');
fprintf(arq,'\n');
fprintf(arq,'   \\hline \n');
for tt=1:size(finalCM,1)
    for uu=1:size(finalCM,2)
        if uu==5
            fprintf(arq,'%6.0f',finalCM(tt,uu));
        else
            fprintf(arq,'%6.0f & ',finalCM(tt,uu));
        end
    end
 fprintf(arq,'    \\\\ \n');
 fprintf(arq,'   \\hline \n');
 end
fprintf(arq,'   \\hline \n');
fprintf(arq,'  \\end{tabular} \n');
fprintf(arq,' \\end{center} \n');
fprintf(arq,'\\end{table*} \n');
fprintf(arq,'\n');
fprintf(arq,'\n');
fprintf(arq,'\\end{document}\n');
fclose(arq);

fprintf('\n Gross Statistics:\n');
fprintf(' Gross & %6.2f & %6.2f & %6.2f & %6.2f & %6.2f & %6.2f & %6.2f & %6.2f & %6.2f & %6.2f & %6.2f \\\\ \n',...
       100*(finalCM(1,1)+finalCM(2,2)+finalCM(3,3)+finalCM(4,4)+finalCM(5,5))/(sum(finalCM(1,:))+sum(finalCM(2,:))+sum(finalCM(3,:))+sum(finalCM(4,:))+sum(finalCM(5,:))) ,...
       100*finalCM(1,1)/sum(finalCM(1,:)), 100*finalCM(2,2)/sum(finalCM(2,:)), 100*finalCM(3,3)/sum(finalCM(3,:)), 100*finalCM(4,4)/sum(finalCM(4,:)), 100*finalCM(5,5)/sum(finalCM(5,:)),...
       100*finalCM(1,1)/sum(finalCM(:,1)), 100*finalCM(2,2)/sum(finalCM(:,2)), 100*finalCM(3,3)/sum(finalCM(:,3)), 100*finalCM(4,4)/sum(finalCM(:,4)), 100*finalCM(5,5)/sum(finalCM(:,5)));

   
end