% Classificação de Arritmias
%
% type : 'AAMI' ou 'AAMI2'
%
% featureSet: conjunto de características
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

function classify_hierarq_lopo(featureSet, classifier, type)

if nargin < 3
   type = 'AAMI';
end

s = char(featureSet);
fileNamed = ['results\',s,'\',classifier,'_lopo_DS1_HIERARQUICO.tex'];
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
s2 = [s2 s '} \n'];
fprintf(arq,s2);
%fprintf(arq,' \\caption{Tebela dos registros do método} \n');
fprintf(arq,' \\label{tab:regtable} \n');
fprintf(arq,' \\begin{center} \n');
fprintf(arq,'  \\begin{tabular}{|c|c|c|c||c|c|c|c|c|} \n');
fprintf(arq,'   \\hline \n');
fprintf(arq,'    Registro & Acc & N Se/+P/FPR & ARRH Se/+P/FPR & ACC etapa 2 & SVEB Se/+P/FPR & VEB Se/+P/FPR & F Se/+P/FPR & Q Se/+P/FPR \\\\ \n');
fprintf(arq,'   \\hline \n');

% iniciliza variaveis

finalCM = zeros(2,2);
finalCM_2= zeros(4,4);

% Inicializa os registros
%DS1 = {'101';'106';'108';'109';'112';'114';'115';'116';'118';'119';'122';'124';'201';'203';'205';'207';'208';'209';'215';'220';'223';'230'};
%DS2 = {'100';'103';'105';'111';'113';'117';'121';'123';'200';'202';'210';'212';'213';'214';'219';'221';'222';'228';'231';'232';'233';'234'};   
registers = {'101';'106';'108';'109';'112';'114';'115';'116';'118';'119';'122';'124';'201';'203';'205';'207';'208';'209';'215';'220';'223';'230'};

% imprime a tabela de registros em arquvo .tex
%printRegisterTable(['features\' featureSet '\']);
%printRegisterTable_15_classes(['features\' featureSet '\']);

test_ds = [];
train_ds = [];
    
for k=1:size(registers,1) % numero de registros
    
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
       
%% Carrega os dados

    feat_folder = ['features\' featureSet '\'];
    %cd(feat_folder)
    
    % já esta sendo feita normalizacao dentro da função : 
    % Xnorm = X - media_treino / std_treino

    if(strcmp(type,'AAMI2'))
        [p1d p1t p2d p2t] = loadDataAAMI2(0,feat_folder,train_ds,test_ds);
    else
        [p1d_step_1 p1t_step_1 p2d_step_1 p2t_step_1] = loadData_CH_step_1(0,feat_folder,train_ds,test_ds);
        [p1d_step_2 p1t_step_2 p2d_step_2 p2t_step_2] = loadData_CH_step_2(0,feat_folder,train_ds,test_ds);
    end

    if strcmp(featureSet,'2007Yu_VCG_n_2_A1_A2_A3_tq_06')
        %fsel = [1 2 3 4 5 6 7 8 9 11 13 17 25 26 29 37]; % 16 melhores
        fsel = [1 2 3 4 5 6 7 8 9 11 13 16 17 25 26 29 37]; % 17 melhores features
        p1d_step_1 = p1d_step_1(:,fsel) ; 
        p2d_step_1 = p2d_step_1(:,fsel) ; 
        p1d_step_2 = p1d_step_2(:,fsel) ; 
        p2d_step_2 = p2d_step_2(:,fsel) ;
    end
    
    %if strcmp(featureSet,'2007Yu')
    %    fsel =[1 2 3 4 5 6 7 11 12];
    %    p1d_step_1 = p1d_step_1(:,fsel) ; 
    %%    p2d_step_1 = p2d_step_1(:,fsel) ; 
    %    p1d_step_2 = p1d_step_2(:,fsel) ; 
    %    p2d_step_2 = p2d_step_2(:,fsel) ; 
    %end
    
    if strcmp(featureSet,'VCG_n_2_32_32_A2_A4_RR_PT_Penerg_QRSwid')
        %fsel =[1 2 3 4 5 6 7 11 12];
        fsel =[1 2 3 4 5 9 17 20 23];
        p1d_step_1 = p1d_step_1(:,fsel) ; 
        p2d_step_1 = p2d_step_1(:,fsel) ; 
        p1d_step_2 = p1d_step_2(:,fsel) ; 
        p2d_step_2 = p2d_step_2(:,fsel) ;
    end
    
    %primeira etapa com DS1 treino DS2 teste
    fs1.train = p1d_step_1;
    fs1.test = p2d_step_1;
    target.train = p1t_step_1;
    target.test = p2t_step_1;
    
    [fs1.train, scale_factor] = mapminmax(fs1.train');   
    fs1.test = mapminmax('apply',fs1.test',scale_factor);
    
    fs1.train = fs1.train';
    fs1.test = fs1.test';
    
    %primeira etapa com DS1 treino DS2 teste
    fs2.train = p1d_step_2;
    fs2.test = p2d_step_2;
    target2.train = p1t_step_2;
    target2.test = p2t_step_2;
    
    [fs2.train, scale_factor] = mapminmax(fs2.train');   
    fs2.test = mapminmax('apply',fs2.test',scale_factor);
    
    fs2.train = fs2.train';
    fs2.test = fs2.test';
    
    %cd ..
    %cd .. 


%% aplica o classificador
if(strcmp(classifier,'SVM'))
    cd('svm')
    
    %[best_c,best_g,best_cv,hC] = parameter_optimization(fs1.train, target.train);
    best_c=0.5;
    best_g=1/(8*size(fs1.train,2));
    best_c_2=0.5;
    best_g_2=best_g;
    
    tic
    
    first_time=0;
    % ***** CLASSIFICA PRIMEIRA ETAPA
    [cm1 model_step_1 predict_matrix label_matrix] = svm_Classifier_CH_1(fs1.train,target.train,fs1.test,target.test,best_c,best_g);
  
   % ***** CLASSIFICA SEGUNDA ETAPA (apenas os batimentos classificados como arritmicos vão para o segundo classificador)
    gg=find(label_matrix(:,2)==1);
    hh=find(predict_matrix(:,2)==1);

    anormal_id = find(predict_matrix(:,2)==1 & label_matrix(:,2));
    %anormal_id = [anormal_id;zeros(1,size(gg,1)-size(anormal_id,1))'];%completa com zero

    indice=[];
    for i=1:size(anormal_id,1)
        indice(i) = find(gg==anormal_id(i));
    end

    [cm2 model_step_2] = svm_Classifier_CH_2(fs2.train,target2.train,fs2.test(indice,:),target2.test(indice,:),best_c_2,best_g_2);
    
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
    
    % ***** CLASSIFICA PRIMEIRA ETAPA
    [cm1 model_step_1 predict_matrix label_matrix] = mlp_Classifier(fs1.train,target.train,fs1.test,target.test);

    % ***** CLASSIFICA SEGUNDA ETAPA (apenas os batimentos classificados como arritmicos vão para o segundo classificador)
    gg=find(label_matrix(:,2)==1);
    hh=find(predict_matrix(:,2)==1);

    anormal_id = find(predict_matrix(:,2)==1 & label_matrix(:,2));

    indice=[];
    for i=1:size(anormal_id,1)
        indice(i) = find(gg==anormal_id(i));
    end

    [cm2] = mlp_Classifier(fs2.train,target2.train,fs2.test(indice,:),target2.test(indice,:));

    %fprintf('\n----------------------- Classificador MLP comb ---------------------\n')
    
    cd ..

elseif(strcmp(classifier,'PNN')) 
	cd('pnn')

	if param1 == 0
		spread = 0.01;
	else
		spread = param1;		
	end
	
	% ***** CLASSIFICA PRIMEIRA ETAPA
  	[cm1 model_step_1 predict_matrix label_matrix] = classifyPNN(fs1.train,target.train,fs1.test,target.test, spread);

	% ***** CLASSIFICA SEGUNDA ETAPA (apenas os batimentos classificados como arritmicos vão para o segundo classificador)
	gg=find(label_matrix(:,2)==1);
	hh=find(predict_matrix(:,2)==1);

	anormal_id = find(predict_matrix(:,2)==1 & label_matrix(:,2));

	indice=[];
       	for i=1:size(anormal_id,1)
       		indice(i) = find(gg==anormal_id(i));
    	end

	[cm2] = classifyPNN(fs2.train,target2.train,fs2.test(indice,:),target2.test(indice,:));

	cd ..
end

 % ***** Calcula estatíSticas PRIMEIR ETAPA
   cm1
   
   finalCM = finalCM + cm1;
   
   acc_num=0;
   acc_den=0;
   den1=0;
   den2=0;
   num=0;
   t=0;
   TN=0;
   
    if(size(cm1,1)>=1)
       t = 1;
       
       num = cm1(t,t);
       den1 = sum(cm1(t,:));
       den2 = sum(cm1(:,t));
       
       TN = sum(sum(cm1(:,:))) - den1 - den2 + cm1(t,t);
       FP = den2 - cm1(t,t);
        
        if(den1~=0)
            sensitivityN = (num/den1) * 100;
        else
            sensitivityN = -1;
        end
        
        if(den2~=0)
            specificityN = (num/den2) * 100;
        else
            specificityN = -1;

        end
        
        if(TN + FP > 0)
            FPR_N = 100*FP/(TN+FP);
        else
            FPR_N=-1;
        end
        
        acc_num = acc_num + num;
        acc_den = acc_den + den1;
        
    else
        sensitivityN = -1;
        specificityN = -1;
        FPR_N=-1;
    end
   
    % caso especial para LD classifier
    %size(cm1,1)==5
    
    if(size(cm1,1)>=2)
       t = 2;
     
       num = cm1(t,t);
        den1 = sum(cm1(t,:));
        den2 = sum(cm1(:,t));
        
        TN = sum(sum(cm1(:,:))) - den1 - den2 + cm1(t,t);
        FP = den2 - cm1(t,t);
       
        if(den1~=0)
            sensitivityARRH = (num/den1) * 100;
        else
            sensitivityARRH = -1;
        end
        
        if(den2~=0)
            specificityARRH = (num/den2) * 100;
        else
            specificityARRH = -1;
        end
        
        if(TN + FP > 0)
            FPR_ARRH = 100*FP/(TN+FP);
        else
            FPR_ARRH=-1;
        end
        
        acc_num = acc_num + num;
        acc_den = acc_den + den1;
        
    else
        sensitivityARRH = -1;
        specificityARRH = -1;
        FPR_ARRH=-1;
    end
   
    fprintf(arq,'%6d & %6.1f & %6.1f/%6.1f/%6.1f & %6.1f/%6.1f/%6.1f &',...
        str2double(DS2(k)), 100*acc_num/acc_den, sensitivityN, specificityN,FPR_N, sensitivityARRH,specificityARRH,FPR_ARRH);         
    %fprintf(arq,'\n');
    
   fprintf('\n');
    
   fprintf('Registro=%6.0f | Acc=%6.1f |', str2double(DS2(k)), 100*acc_num/acc_den);
    
   fprintf(' N_Se=%6.1f N+P=%6.1f ARRH_Se=%6.1f ARRH+P=%6.1f ',...
       sensitivityN, specificityN, sensitivityARRH, specificityARRH)
 
   cm1=[];
   % **** Calcula estatíSticas PARA SEGUNDA ETAPA
   cm2
   
   if isempty(cm2)
       cm2 = zeros(4,4);
   end
   finalCM_2 = finalCM_2 + cm2;
   
   acc_num=0;
   acc_den=0;
   den1=0;
   den2=0;
   num=0;
   t=0;
   
    % caso especial para LD classifier
    %size(cm1,1)==5
    
    if(size(cm2,1)>=1)
       t = 1;
     
       num = cm2(t,t);
        den1 = sum(cm2(t,:));
        den2 = sum(cm2(:,t));
        
        TN = sum(sum(cm2(:,:))) - den1 - den2 + cm2(t,t);
        FP = den2 - cm2(t,t);
        
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
   
    if(size(cm2,1)>=2)
        t = 2;
        num = cm2(t,t);
        den1 = sum(cm2(t,:));
        den2 = sum(cm2(:,t));
        
        TN = sum(sum(cm2(:,:))) - den1 - den2 + cm2(t,t);
        FP = den2 - cm2(t,t);
        
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
   
    if(size(cm2,1)>=3)
       t = 3;
       
       num = cm2(t,t);
        den1 = sum(cm2(t,:));
        den2 = sum(cm2(:,t));
        
        TN = sum(sum(cm2(:,:))) - den1 - den2 + cm2(t,t);
        FP = den2 - cm2(t,t);
        
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
   
    if(size(cm2,1)>=4)
       t = 4;
       
        num = cm2(t,t);
        den1 = sum(cm2(t,:));
        den2 = sum(cm2(:,t));
        
        TN = sum(sum(cm2(:,:))) - den1 - den2 + cm2(t,t);
        FP = den2 - cm2(t,t);
        
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
   
    fprintf(arq,' %6.1f & %6.1f/%6.1f/%6.1f & %6.1f/%6.1f/%6.1f & %6.1f/%6.1f/%6.1f & %6.1f/%6.1f/%6.1f \\\\ \n',...
        100*acc_num/acc_den, sensitivitySVEB,specificitySVEB,FPR_SVEB,sensitivityVEB,specificityVEB,FPR_VEB,...
        sensitivityF,specificityF,FPR_F, sensitivityQ,specificityQ,FPR_Q);         
    fprintf(arq,'\n');
    
   %fprintf('\n');
    
   fprintf('Acc etapa 2 =%6.1f |', 100*acc_num/acc_den);
    
   fprintf(' SVEB_Se=%6.1f SVEB+P=%6.1f VEB_Se=%6.1f  VEB+P=%6.1f \n\n',...
        sensitivitySVEB,specificitySVEB,sensitivityVEB,specificityVEB)
    
end % for t

fprintf(arq,'    \\hline \n');
%fprintf(arq,'    \\hline \n');

TN = sum(sum(finalCM(:,:))) - sum(finalCM(1,:)) - sum(finalCM(:,1)) + finalCM(1,1);
FP = sum(finalCM(:,1)) - finalCM(1,1);
FPR_N = FP/(TN+FP);
        
TN = sum(sum(finalCM(:,:))) - sum(finalCM(2,:)) - sum(finalCM(:,2)) + finalCM(2,2);
FP = sum(finalCM(:,2)) - finalCM(2,2);
FPR_ARRH = FP/(TN+FP);

fprintf(arq, ' Gross & %6.1f & %6.1f/%6.1f/%6.1f & %6.1f/%6.1f/%6.1f ',...
       100*(finalCM(1,1)+finalCM(2,2))/(sum(finalCM(1,:))+sum(finalCM(2,:))) ,...
       100*finalCM(1,1)/sum(finalCM(1,:)), 100*finalCM(1,1)/sum(finalCM(:,1)),100*FPR_N,...
       100*finalCM(2,2)/sum(finalCM(2,:)), 100*finalCM(2,2)/sum(finalCM(:,2)),100*FPR_ARRH);

TN = sum(sum(finalCM_2(:,:))) - sum(finalCM_2(1,:)) - sum(finalCM_2(:,1)) + finalCM_2(1,1);
FP = sum(finalCM_2(:,1)) - finalCM_2(1,1);
FPR_SVEB = FP/(TN+FP);

TN = sum(sum(finalCM_2(:,:))) - sum(finalCM_2(2,:)) - sum(finalCM_2(:,2)) + finalCM_2(2,2);
FP = sum(finalCM_2(:,2)) - finalCM_2(2,2);
FPR_VEB = FP/(TN+FP);

TN = sum(sum(finalCM_2(:,:))) - sum(finalCM_2(3,:)) - sum(finalCM_2(:,3)) + finalCM_2(3,3);
FP = sum(finalCM_2(:,3)) - finalCM_2(3,3);
FPR_F = FP/(TN+FP);

TN = sum(sum(finalCM_2(:,:))) - sum(finalCM_2(4,:)) - sum(finalCM_2(:,4)) + finalCM_2(4,4);
FP = sum(finalCM_2(:,4)) - finalCM_2(4,4);
FPR_Q = FP/(TN+FP);

fprintf(arq, '& %6.1f & %6.1f/%6.1f/%6.1f & %6.1f/%6.1f/%6.1f & %6.1f/%6.1f/%6.1f & %6.1f/%6.1f/%6.1f  \\\\ \n',...
    100*(finalCM_2(1,1)+finalCM_2(2,2)+finalCM_2(3,3)+finalCM_2(4,4))/(sum(finalCM_2(1,:))+sum(finalCM_2(2,:))+sum(finalCM_2(3,:))+sum(finalCM_2(4,:))) ,...
       100*finalCM_2(1,1)/sum(finalCM_2(1,:)),100*finalCM_2(1,1)/sum(finalCM_2(1:end-1,1)),100*FPR_SVEB,...
       100*finalCM_2(2,2)/sum(finalCM_2(2,:)),100*finalCM_2(2,2)/sum(finalCM_2(1:end-2,2)),100*FPR_VEB,...
       100*finalCM_2(3,3)/sum(finalCM_2(3,:)),100*finalCM_2(3,3)/sum(finalCM_2(:,3)),100*FPR_F,...
       100*finalCM_2(4,4)/sum(finalCM_2(4,:)),100*finalCM_2(4,4)/sum(finalCM_2(:,4)),100*FPR_Q);
   
fprintf(arq,'   \\hline \n');
fprintf(arq,'  \\end{tabular} \n');
fprintf(arq,' \\end{center} \n');
fprintf(arq,'\\end{table*} \n');
fprintf(arq,'\n');
fprintf(arq,'\n');

fprintf(arq,'\n');
fprintf(arq,'\\begin{table*} \n');
fprintf(arq,' \\caption{Matriz e confusão etapa 1} \n');
fprintf(arq,' \\label{tab:regtable1} \n');
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

fprintf(arq,'\n');
fprintf(arq,'\\begin{table*} \n');
fprintf(arq,' \\caption{Matriz e confusão etapa 2} \n');
fprintf(arq,' \\label{tab:regtable2} \n');
fprintf(arq,' \\begin{center} \n');
fprintf(arq,'  \\begin{tabular}{|');
for tt=1:size(finalCM_2,1)
    fprintf(arq,'c|');
end
fprintf(arq,'} \n');
fprintf(arq,'\n');
fprintf(arq,'   \\hline \n');
for tt=1:size(finalCM_2,1)
    for uu=1:size(finalCM_2,2)
        if uu==5
            fprintf(arq,'%6.0f',finalCM_2(tt,uu));
        else
            fprintf(arq,'%6.0f & ',finalCM_2(tt,uu));
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
fprintf(' Gross & %6.1f & %6.1f/%6.1f & %6.1f/%6.1f ',...
       100*(finalCM(1,1)+finalCM(2,2))/(sum(finalCM(1,:))+sum(finalCM(2,:))) ,...
       100*finalCM(1,1)/sum(finalCM(1,:)), 100*finalCM(1,1)/sum(finalCM(:,1)),...
       100*finalCM(2,2)/sum(finalCM(2,:)), 100*finalCM(2,2)/sum(finalCM(:,2)));

fprintf(' %6.1f & %6.1f/%6.1f & %6.1f/%6.1f & %6.1f/%6.1f & %6.1f/%6.1f  \\\\ \n',...
       100*(finalCM_2(1,1)+finalCM_2(2,2)+finalCM_2(3,3)+finalCM_2(4,4))/(sum(finalCM_2(1,:))+sum(finalCM_2(2,:))+sum(finalCM_2(3,:))+sum(finalCM_2(4,:))) ,...
       100*finalCM_2(1,1)/sum(finalCM_2(1,:)), 100*finalCM_2(1,1)/sum(finalCM_2(:,1)),...
       100*finalCM_2(2,2)/sum(finalCM_2(2,:)),100*finalCM_2(2,2)/sum(finalCM_2(:,2)),...
       100*finalCM_2(3,3)/sum(finalCM_2(3,:)),100*finalCM_2(3,3)/sum(finalCM_2(:,3)),...
       100*finalCM_2(4,4)/sum(finalCM_2(4,:)),100*finalCM_2(4,4)/sum(finalCM_2(:,4)));
 
   
end
