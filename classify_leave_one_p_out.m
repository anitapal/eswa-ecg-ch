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
%

function classify_leave_one_p_out(featureSet, classifier, type, pca,outlier, indexOut)

withoutPreIndex=0;

if nargin < 3
   type = 'AAMI';
   pca=0;
   outlier=0;
   indexOut=[];
   withoutPreIndex = 1;
elseif nargin < 4
   pca=0;
   outlier=0;
   indexOut=[];
   withoutPreIndex = 1;
elseif nargin < 5
   outlier=0;
   indexOut=[];
   withoutPreIndex = 1;
elseif nargin < 6
   indexOut=[];
   withoutPreIndex = 1;
end

s = char(featureSet);
fileNamed = ['results\',s,'_',classifier,'_',type,'_DS1_results.tex'];
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
%fprintf(arq,' \\caption{Tebela dos registros do método} \n');
fprintf(arq,' \\label{tab:regtable} \n');
fprintf(arq,' \\begin{center} \n');
fprintf(arq,'  \\begin{tabular}{|c|c|c|c|c|c|c|} \n');
fprintf(arq,'   \\hline \n');
fprintf(arq,'    Registro & Acc & N Se/+P/FPR & SVEB Se/+P/FPR & VEB Se/+P/FPR & F Se/+P/FPR & Q Se/+P/FPR  \\\\ \n');
fprintf(arq,'   \\hline \n');

% iniciliza variaveis
finalCM = zeros(5,5);

% Inicializa os registros
%registers = {'232';'101';'106';'108';'109';'112';'114';'115';'116';'118';'119';'122';'124';'201';'203';'205';'207';'208';'209';'215';'220';'223';'230';...
%    '100';'103';'105';'111';'113';'117';'121';'123';'200';'202';'210';'212';'213';'214';'219';'221';'222';'228';'231';'233';'234'};   

%apenas registros de DS1
registers = {'101';'106';'108';'109';'112';'114';'115';'116';'118';'119';'122';'124';'201';'203';'205';'207';'208';'209';'215';'220';'223';'230'};

% imprime a tabela de registros em arquvo .tex
%printRegisterTable(['features\' featureSet '\']);

fprintf('\n Registro | Acc | N_Se SVEB_Se VEB_Se N+P SVEB+P VEB+P \n')

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
if(strcmp(featureSet,'2004Chazal')|| strcmp(featureSet,'2004Chazal_all_filt'))
    feat_folder = ['features\' featureSet '\'];
    %cd(feat_folder)
    
    % já esta sendo feita normalizacao dentro da função : 
    % Xnorm = X - media_treino / std_treino

    if(strcmp(type,'AAMI2'))
        [fs_1 fs_2 fs_3 fs_4 fs_5 fs_6 fs_7 fs_8 target] = loadDataAAMI2_chazal(feat_folder, train_ds,test_ds);
    else
        [fs_1 fs_2 fs_3 fs_4 fs_5 fs_6 fs_7 fs_8 target] = loadDataAAMI_chazal(feat_folder, train_ds,test_ds);
    end

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
    
    if(pca)
        [coeff index1]=applyPCA(fs1.train, 99);
        fs1.train = fs1.train * coeff(:,1:index1);
        fs1.test = fs1.test * coeff(:,1:index1);
    end
    
else
    feat_folder = ['features\' featureSet '\'];
    %cd(feat_folder)
    
    % já esta sendo feita normalizacao dentro da função : 
    % Xnorm = X - media_treino / std_treino

    if(strcmp(type,'AAMI2'))
        [p1d p1t p2d p2t] = loadDataAAMI2(0,feat_folder,train_ds,test_ds);
    else
        [p1d p1t p2d p2t] = loadDataAAMI(0,feat_folder,train_ds,test_ds);
    end

    if strcmp(featureSet,'2007Yu_VCG_n_2_A1_A2_A3_tq_06')
        %fsel =[1 2 3 4 5 6 7 8 9 10 12 15];
        %fsel =[11 12 13 14 15 16];
        %fsel = [ 1 2 3 4 5 6 7 8 9 17 26 29]; % 12 features
        fsel = [ 1 2 3 4 5 6 7 8 9 13 17 26 29]; % 13 features
        %fsel = [1 2 3 4 5 6 7 8 9 13 17 25 26 29 ]; % 15 features
        %fsel = [1 2 3 4 5 6 7 8 9 11 13 17 25 26 29 37]; % 16 melhores
        %fsel = [1 2 3 4 5 6 7 8 9 11 13 16 17 25 26 29 37]; % 17 melhores features
        
        p1d = p1d(:,fsel) ; 
        p2d = p2d(:,fsel) ; 
    end
    
    if strcmp(featureSet,'2007Yu_modified')
        fsel =[1 2 3 4 5 6 7 8 9 13 14];
        p1d = p1d(:,fsel) ; 
        p2d = p2d(:,fsel) ; 
    end
    
    if strcmp(featureSet,'2007Yu')
        fsel =[1 2 3 4 5 6 7 11 12]; % selecionado com busca para frente
        p1d = p1d(:,fsel) ; 
        p2d = p2d(:,fsel) ; 
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
    
    if(pca)
        [coeff index1]=applyPCA(fs1.train, 99);
        fs1.train = fs1.train * coeff(:,1:index1);
        fs1.test = fs1.test * coeff(:,1:index1);
    end
    
    if outlier
        if (outlier && ~withoutPreIndex)
            fs1.train(indexOut(k).N,:)=[];
            target.train(indexOut(k).N,:)=[];
            %remove outliers de S (maximo 5%)
            fs1.train(indexOut(k).S,:)=[];
            target.train(indexOut(k).S,:)=[];
            %remove outliers de V (maximo 5%)
            fs1.train(indexOut(k).V,:)=[];
            target.train(indexOut(k).V,:)=[];
            %remove outliers de F (maximo 5%)
            fs1.train(indexOut(k).F,:)=[];
            target.train(indexOut(k).F,:)=[];
        else
            indexOut(k)=[];
            %remove outliers de N (maximo 5%)
            indexOut(k).N = outlierRemoval( fs1.train, target.train, 1);
            fs1.train(indexOut(k).N,:)=[];
            target.train(indexOut(k).N,:)=[];
            %remove outliers de S (maximo 5%)
            indexOut(k).S = outlierRemoval( fs1.train, target.train, 2);
            fs1.train(indexOut(k).S,:)=[];
            target.train(indexOut(k).S,:)=[];
            %remove outliers de V (maximo 5%)
            indexOut(k).V = outlierRemoval( fs1.train, target.train, 3);
            fs1.train(indexOut(k).V,:)=[];
            target.train(indexOut(k).V,:)=[];
            %remove outliers de F (maximo 5%)
            indexOut(k).F = outlierRemoval( fs1.train, target.train, 4);
            fs1.train(indexOut(k).F,:)=[];
            target.train(indexOut(k).F,:)=[];

            %save([featureSet '_Outlier_index'], 'indexN', 'indexS', 'indexV', 'indexF');
            %save([featureSet '_Outlier_lopo_index'], 'indexOut');
        end
    end
            
    %cd ..
    %cd .. 
end

%% aplica o classificador
if(strcmp(classifier,'LD'))
    cd('LD_classifier')

    if(strcmp(featureSet,'2004Chazal') || strcmp(featureSet,'2004Chazal_all_filt'))
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
    %best_c=0.1;
    %best_g=1/(8*size(fs1.train,2));
    
    best_c=0.5;
    best_g=1/(8*size(fs1.train,2));
    
    %best_c=0.5;
    %best_g=0.00097656;
    
    %[newData newTarget] = unedersampling_class1(3, fs1.train,target.train);
    %[best_c,best_g,best_cv,hC] = parameter_optimization(newData,newTarget);
    tic
    
    clear cm1;
    [cm1] = svm_Classifier(fs1.train,target.train,fs1.test,target.test,best_c,best_g);
    
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


% Calcula estatíSticas
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
   
    if(size(cm1,1)>=3)
       t = 3;
        num = cm1(t,t);
        den1 = sum(cm1(t,:));
        den2 = sum(cm1(:,t));
        
        TN = sum(sum(cm1(:,:))) - den1 - den2 + cm1(t,t);
        FP = den2 - cm1(t,t);
        
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
   
    if(size(cm1,1)>=4)
       t = 4;
       
       num = cm1(t,t);
        den1 = sum(cm1(t,:));
        den2 = sum(cm1(:,t));
        
        TN = sum(sum(cm1(:,:))) - den1 - den2 + cm1(t,t);
        FP = den2 - cm1(t,t);
        
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
   
    if(size(cm1,1)>=5)
       t = 5;
       
        num = cm1(t,t);
        den1 = sum(cm1(t,:));
        den2 = sum(cm1(:,t));
        
        TN = sum(sum(cm1(:,:))) - den1 - den2 + cm1(t,t);
        FP = den2 - cm1(t,t);
        
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
    
    fprintf(arq,'%6d & %6.1f & %6.1f/%6.1f/%6.1f & %6.1f/%6.1f/%6.1f & %6.1f/%6.1f/%6.1f & %6.1f/%6.1f/%6.1f & %6.1f/%6.1f/%6.1f \\\\ \n',...
        str2double(registers(k)), 100*acc_num/acc_den, sensitivityN,specificityN,FPR_N,sensitivitySVEB,specificitySVEB,FPR_SVEB,sensitivityVEB,specificityVEB,FPR_VEB,...
        sensitivityF,specificityF,FPR_F, sensitivityQ,specificityQ,FPR_Q);         
    fprintf(arq,'\n');
    
   fprintf('\n');
    
   fprintf('Registro=%6.0f | Acc=%6.1f |', str2double(registers(k)), 100*acc_num/acc_den);
    
   fprintf(' N_Se=%6.1f SVEB_Se=%6.1f VEB_Se=%6.1f N+P=%6.1f SVEB+P=%6.1f VEB+P=%6.1f \n\n',...
       sensitivityN, sensitivitySVEB,sensitivityVEB,  specificityN, specificitySVEB,specificityVEB)
 
   
end % for t

fprintf(arq,'    \\hline \n');
%fprintf(arq,'    \\hline \n');

TN = sum(sum(finalCM(:,:))) - sum(finalCM(1,:)) - sum(finalCM(:,1)) + finalCM(1,1);
FP = sum(finalCM(:,1)) - finalCM(1,1);
FPR_N = FP/(TN+FP);
        
TN = sum(sum(finalCM(:,:))) - sum(finalCM(2,:)) - sum(finalCM(:,2)) + finalCM(2,2);
FP = sum(finalCM(:,2)) - finalCM(2,2);
FPR_SVEB = FP/(TN+FP);

TN = sum(sum(finalCM(:,:))) - sum(finalCM(3,:)) - sum(finalCM(:,3)) + finalCM(3,3);
FP = sum(finalCM(:,3)) - finalCM(3,3);
FPR_VEB = FP/(TN+FP);

if size(finalCM,1) > 3

    TN = sum(sum(finalCM(:,:))) - sum(finalCM(4,:)) - sum(finalCM(:,4)) + finalCM(4,4);
FP = sum(finalCM(:,4)) - finalCM(4,4);
FPR_F = FP/(TN+FP);

TN = sum(sum(finalCM(:,:))) - sum(finalCM(5,:)) - sum(finalCM(:,5)) + finalCM(5,5);
FP = sum(finalCM(:,5)) - finalCM(5,5);
FPR_Q = FP/(TN+FP);

fprintf(arq, ' Gross & %6.1f & %6.1f/%6.1f/%6.1f & %6.1f/%6.1f/%6.1f & %6.1f/%6.1f/%6.1f & %6.1f/%6.1f/%6.1f & %6.1f/%6.1f/%6.1f \\\\ \n',...
    100*(finalCM(1,1)+finalCM(2,2)+finalCM(3,3)+finalCM(4,4)+finalCM(5,5))/(sum(finalCM(1,:))+sum(finalCM(2,:))+sum(finalCM(3,:))+sum(finalCM(4,:))+sum(finalCM(5,:))) ,...
       100*finalCM(1,1)/sum(finalCM(1,:)), 100*finalCM(1,1)/sum(finalCM(:,1)),100*FPR_N,...
       100*finalCM(2,2)/sum(finalCM(2,:)),100*finalCM(2,2)/sum(finalCM(1:end-1,2)),100*FPR_SVEB,...
       100*finalCM(3,3)/sum(finalCM(3,:)),100*finalCM(3,3)/sum(finalCM(1:end-2,3)),100*FPR_VEB,...
       100*finalCM(4,4)/sum(finalCM(4,:)),100*finalCM(4,4)/sum(finalCM(:,4)),100*FPR_F,...
       100*finalCM(4,4)/sum(finalCM(5,:)),100*finalCM(5,5)/sum(finalCM(:,5)),100*FPR_Q);
   
else
    fprintf(arq, ' Gross & %6.1f & %6.1f/%6.1f/%6.1f & %6.1f/%6.1f/%6.1f & %6.1f/%6.1f/%6.1f & - & - \\\\ \n',...
    100*(finalCM(1,1)+finalCM(2,2)+finalCM(3,3))/(sum(finalCM(1,:))+sum(finalCM(2,:))+sum(finalCM(3,:))) ,...
       100*finalCM(1,1)/sum(finalCM(1,:)), 100*finalCM(1,1)/sum(finalCM(:,1)),100*FPR_N,...
       100*finalCM(2,2)/sum(finalCM(2,:)),100*finalCM(2,2)/sum(finalCM(:,2)),100*FPR_SVEB,...
       100*finalCM(3,3)/sum(finalCM(3,:)),100*finalCM(3,3)/sum(finalCM(:,3)),100*FPR_VEB);
end

fprintf(arq,'   \\hline \n');
fprintf(arq,'  \\end{tabular} \n');
fprintf(arq,' \\end{center} \n');
fprintf(arq,'\\end{table*} \n');
fprintf(arq,'\n');
fprintf(arq,'\n');

fprintf(arq,'\n');
fprintf(arq,'\\begin{table*} \n');
fprintf(arq,' \\caption{Matriz e confusão} \n');
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
fprintf(' Gross & %6.1f & %6.1f & %6.1f & %6.1f & %6.1f & %6.1f & %6.1f & %6.1f & %6.1f & %6.1f & %6.1f \\\\ \n',...
       100*(finalCM(1,1)+finalCM(2,2)+finalCM(3,3)+finalCM(4,4)+finalCM(5,5))/(sum(finalCM(1,:))+sum(finalCM(2,:))+sum(finalCM(3,:))+sum(finalCM(4,:))+sum(finalCM(5,:))) ,...
       100*finalCM(1,1)/sum(finalCM(1,:)), 100*finalCM(2,2)/sum(finalCM(2,:)), 100*finalCM(3,3)/sum(finalCM(3,:)), 100*finalCM(4,4)/sum(finalCM(4,:)), 100*finalCM(5,5)/sum(finalCM(5,:)),...
       100*finalCM(1,1)/sum(finalCM(:,1)), 100*finalCM(2,2)/sum(finalCM(:,2)), 100*finalCM(3,3)/sum(finalCM(:,3)), 100*finalCM(4,4)/sum(finalCM(:,4)), 100*finalCM(5,5)/sum(finalCM(:,5)));
  
   if outlier
       save([featureSet '_Outlier_lopo_index'], 'indexOut');
   end

end