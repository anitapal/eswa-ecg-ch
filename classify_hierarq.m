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

function classify_hierarq(featureSet, classifier, type, param1, param2, param3)

if nargin < 3
   type = 'AAMI';
    param1 = 0;
   param2 = 0;
   param3 = 0;  
elseif nargin < 4
   %C = 0;
   %gamma = 0; % default
   % default
   param1 = 0;
   param2 = 0;
   param3 = 0;  
elseif nargin < 5
   param2 = 0;
   param3 = 0;  
elseif nargin < 6
   param3 = 0;  
end
s = char(featureSet);
fileNamed = ['results\',s,'\',classifier,'_test_ds2_HIERARQUICO.tex'];
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
%fprintf(arq,' \\caption{Tebela dos registros do m�todo} \n');
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
DS1 = {'101';'106';'108';'109';'112';'114';'115';'116';'118';'119';'122';'124';'201';'203';'205';'207';'208';'209';'215';'220';'223';'230'};
DS2 = {'100';'103';'105';'111';'113';'117';'121';'123';'200';'202';'210';'212';'213';'214';'219';'221';'222';'228';'231';'232';'233';'234'};   
%DS1 = {'101';'106'};
%DS2 = {'100';'103';'105';'111';'113';'117'};   


% imprime a tabela de registros em arquvo .tex
%printRegisterTable(['features\' featureSet '\']);
%printRegisterTable_15_classes(['features\' featureSet '\']);

test_ds = [];
train_ds = [];
    
for j=1:size(DS1,1)
    train_ds(j) = str2double(DS1(j));
end

first_time=1;
    
for k=1:size(DS2,1) % numero de registros
    test_ds = str2double(DS2(k));
       
%% Carrega os dados

    feat_folder = ['features\' featureSet '\'];
    %cd(feat_folder)
    
    % j� esta sendo feita normalizacao dentro da fun��o : 
    % Xnorm = X - media_treino / std_treino

    if(strcmp(type,'AAMI2'))
        [p1d p1t p2d p2t] = loadDataAAMI2(0,feat_folder,train_ds,test_ds);
    else
        [p1d_step_1 p1t_step_1 p2d_step_1 p2t_step_1] = loadData_CH_step_1(0,feat_folder,train_ds,test_ds);
        [p1d_step_2 p1t_step_2 p2d_step_2 p2t_step_2] = loadData_CH_step_2(0,feat_folder,train_ds,test_ds);
    end

    if strcmp(featureSet,'VCG_n_2_64_64_A1_A2_A4_H_P_RR_PT_Penerg_QRSwid_autocorr')
        fsel =[ 1 2 6 8 16 17 19 20]; % 8 melhores features 
        
        p1d_step_1 = p1d_step_1(:,fsel) ; 
        p2d_step_1 = p2d_step_1(:,fsel) ; 
        p1d_step_2 = p1d_step_2(:,fsel) ; 
        p2d_step_2 = p2d_step_2(:,fsel) ;
    end
    
    if strcmp(featureSet,'2007Yu_VCG_n_2_A1_A2_A3_tq_06')
        %fsel = [1 2 3 4 5 6 7 8 9 11 13 17 25 26 29 37]; % 16 melhores
        fsel = [1 2 3 4 5 6 7 8 9 11 13 16 17 25 26 29 37]; % 17 melhores features
        p1d_step_1 = p1d_step_1(:,fsel) ; 
        p2d_step_1 = p2d_step_1(:,fsel) ; 
        p1d_step_2 = p1d_step_2(:,fsel) ; 
        p2d_step_2 = p2d_step_2(:,fsel) ;
    end
    
   % if strcmp(featureSet,'2007Yu')
   %     fsel =[1 2 3 4 5 6 7 11 12];
   %     p1d_step_1 = p1d_step_1(:,fsel) ; 
   %     p2d_step_1 = p2d_step_1(:,fsel) ; 
   %     p1d_step_2 = p1d_step_2(:,fsel) ; 
   %     p2d_step_2 = p2d_step_2(:,fsel) ; 
   % end
    
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
    
	
    tic
    
    if(first_time==1)
        first_time=0;

	if(param1==0)
	    %[best_c,best_g,best_cv,hC] = parameter_optimization(fs1.train, target.train);
	    best_c=0.5;
	    best_g=1/(8*size(fs1.train,2));
	    best_c_2=0.5;
	    best_g_2=best_g;
	elseif(param1>0)
	    best_c=param1;
	    best_g=param2;
	    best_c_2=param1;
	    best_g_2=param2;
	elseif(param1==-1)
		% ***** CLASSIFICA PRIMEIRA ETAPA
		[newData newTarget] = undersampling_all_class(20, fs1.train,target.train);
		[best_c,best_g,best_cv,hC] = parameter_optimization_CH_1(newData,newTarget);
		
		[newData2 newTarget2] = undersampling_all_class(20,fs2.train,target2.train);
        	[best_c_2,best_g_2,best_cv,hC] = parameter_optimization_CH_2(newData2,newTarget2);
	end
         
        %[newData newTarget] = SMOTE(fs1.train, target.train, 2, 4000, 5);
        [cm1 model_step_1 predict_matrix label_matrix] = svm_Classifier_CH_1(fs1.train,target.train,fs1.test,target.test,best_c,best_g);
        %[cm1 model_step_1 predict_matrix label_matrix] = svm_Classifier_CH_1(newData, newTarget,fs1.test,target.test,best_c,best_g);
        
        % ***** CLASSIFICA SEGUNDA ETAPA
        %[newData newTarget] = undersampling_all_class(10,
        %fs2.train,target2.train);
        %[best_c_2,best_g_2,best_cv,hC] = parameter_optimization_CH_2(fs2.train,target2.train);
         
        %[newData newTarget] = SMOTE(fs1.train, target.train, 2, 4000, 5);
        gg=find(label_matrix(:,2)==1);
        hh=find(predict_matrix(:,2)==1);

        anormal_id = find(predict_matrix(:,2)==1 & label_matrix(:,2));
        %anormal_id = [anormal_id;zeros(1,size(gg,1)-size(anormal_id,1))'];%completa com zero
        
        indice=[];
        for i=1:size(anormal_id,1)
            indice(i) = find(gg==anormal_id(i));
        end
        
        [cm2 model_step_2] = svm_Classifier_CH_2(fs2.train,target2.train,fs2.test(indice,:),target2.test(indice,:),best_c_2,best_g_2);
        
    else
        % ***** CLASSIFICA PRIMEIRA ETAPA
        testingLabels = [];
        targetsTest = [];
        % formata para uso em SVM - para cada classe
        for t=1:size(target.test,2)
            targetsTest(find(target.test(:,t)==1),t)=t;     
        end
        testingLabels = sum(targetsTest');
        
        [predict_label, accuracy, dec_values] = svmpredict(testingLabels', fs1.test, model_step_1); % test the training data
        %cm1 = confusionmat(testingLabels',predict_label);
        clear predict_matrix;
        
            for i=1: size(predict_label,1)
            switch(predict_label(i))
                case 1,
                    predict_matrix(i,:) = [1 0];            
                case 2,
                    predict_matrix(i,:) = [0 1];            
               
            end
            end

            clear label_matrix;
            for i=1: size(testingLabels,2)
            switch(testingLabels(i))
                case 1,
                    label_matrix(i,:) = [1 0];            
                case 2,
                    label_matrix(i,:) = [0 1];            
               
            end
            end
            %[C,CM,IND,PER] = CONFUSION(TARGETS,OUTPUTS)
            [c cm1] = confusion(label_matrix', predict_matrix');
            
            % **** CLASSIFICA SEGUNDA ETAPA
            
            % so classificaco quem foi considerado arritmico na etapa
            % anterior
            gg=find(label_matrix(:,2)==1);
            hh=find(predict_matrix(:,2)==1);
            
            anormal_id = find(predict_matrix(:,2)==1 & label_matrix(:,2));
            %anormal_id = [anormal_id;zeros(1,size(gg,1)-size(anormal_id,1))'];
            
            testingLabels = [];
            targetsTest = [];
            % formata para uso em SVM - para cada classe
            for t=1:size(target2.test,2)
                targetsTest(find(target2.test(:,t)==1),t)=t;     
            end
            testingLabels = sum(targetsTest');

            indice=[];
            for i=1:size(anormal_id,1)
                indice(i) = find(gg==anormal_id(i));
            end
        
            [predict_label_2, accuracy_2, dec_values_2] = svmpredict(testingLabels(indice)', fs2.test(indice,:), model_step_2); % test the training data
            %cm1 = confusionmat(testingLabels',predict_label);
            clear predict_matrix_2;
            predict_matrix_2=[];
                for i=1: size(predict_label_2,1)
                switch(predict_label_2(i))
                    case 1,
                        predict_matrix_2(i,:) = [1 0 0 0];            
                    case 2,
                        predict_matrix_2(i,:) = [0 1 0 0];            
                    case 3,
                        predict_matrix_2(i,:) = [0 0 1 0];            
                    case 4,
                        predict_matrix_2(i,:) = [0 0 0 1]; 

                end
                end

                tLabes = testingLabels(indice);
                clear label_matrix_2;
                label_matrix_2=[];
                for i=1: size(tLabes,2)
                switch(tLabes(i))
                    case 1,
                        label_matrix_2(i,:) = [1 0 0 0];            
                    case 2,
                        label_matrix_2(i,:) = [0 1 0 0];            
                    case 3,
                        label_matrix_2(i,:) = [0 0 1 0];            
                    case 4,
                        label_matrix_2(i,:) = [0 0 0 1]; 

                end
                end
                %[C,CM,IND,PER] = CONFUSION(TARGETS,OUTPUTS)
                [c2 cm2] = confusion(label_matrix_2', predict_matrix_2');
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
    
   if(first_time==1)
         first_time=0;
	
	% ***** CLASSIFICA PRIMEIRA ETAPA
   	 [cm1 net1_step1 net2_step1 net3_step1 net4_step1 combNet_step1 predict_matrix label_matrix] = mlp_Classifier(fs1.train,target.train,fs1.test,target.test);

	% ***** CLASSIFICA SEGUNDA ETAPA (apenas os batimentos classificados como arritmicos v�o para o segundo classificador)
	    gg=find(label_matrix(:,2)==1);
	    hh=find(predict_matrix(:,2)==1);

	    anormal_id = find(predict_matrix(:,2)==1 & label_matrix(:,2));

	    indice=[];
	    for i=1:size(anormal_id,1)
		indice(i) = find(gg==anormal_id(i));
	    end

    	[cm2 net1_step2 net2_step2 net3_step2 net4_step2 combNet_step2] = mlp_Classifier(fs2.train,target2.train,fs2.test(indice,:),target2.test(indice,:));
   else
	
           % ***** CLASSIFICA PRIMEIRA ETAPA
            testingLabels = [];
            targetsTest = [];
            % formata para uso em SVM - para cada classe
            for t=1:size(target.test,2)
                targetsTest(find(target.test(:,t)==1),t)=t;     
            end
            testingLabels = sum(targetsTest');

            %[predict_label, accuracy, dec_values] = svmpredict(testingLabels', fs1.test, model_step_1); % test the training data
            %cm1 = confusionmat(testingLabels',predict_label);
            outputs1 = sim(net1_step1,fs1.test');
            outputs2 = sim(net2_step1,fs1.test');
            outputs3 = sim(net3_step1,fs1.test');
            outputs4 = sim(net4_step1,fs1.test');

            newInputTest = [outputs1;outputs2;outputs3;outputs4];

            predict_label = sim(combNet_step1,newInputTest);
            predict_label = predict_label';
            
            clear predict_matrix;
            predict_matrix = round(predict_label);
            
            %predict_matrix=[];
            %for i=1: size(predict_label,1)
            %switch(predict_label(i))
            %    case 1,
            %        predict_matrix(i,:) = [1 0];            
            %    case 2,
            %        predict_matrix(i,:) = [0 1];            
               
            %end
            %end

            clear label_matrix;
            label_matrix=[];
            for i=1: size(testingLabels,2)
            switch(testingLabels(i))
                case 1,
                    label_matrix(i,:) = [1 0];            
                case 2,
                    label_matrix(i,:) = [0 1];            
               
            end
            end
            %[C,CM,IND,PER] = CONFUSION(TARGETS,OUTPUTS)
            [c cm1] = confusion(label_matrix', predict_matrix');
            
            % **** CLASSIFICA SEGUNDA ETAPA
            
            % so classificaco quem foi considerado arritmico na etapa
            % anterior
            gg=find(label_matrix(:,2)==1);
            hh=find(predict_matrix(:,2)==1);
            
            anormal_id = find(predict_matrix(:,2)==1 & label_matrix(:,2));
            %anormal_id = [anormal_id;zeros(1,size(gg,1)-size(anormal_id,1))'];
            
            testingLabels = [];
            targetsTest = [];
            % formata para uso em SVM - para cada classe
            for t=1:size(target2.test,2)
                targetsTest(find(target2.test(:,t)==1),t)=t;     
            end
            testingLabels = sum(targetsTest');

            indice=[];
            for i=1:size(anormal_id,1)
                indice(i) = find(gg==anormal_id(i));
            end
        
            %[predict_label_2, accuracy_2, dec_values_2] = svmpredict(testingLabels(indice)', fs2.test(indice,:), model_step_2); % test the training data
            %cm1 = confusionmat(testingLabels',predict_label);
            outputs1 = sim(net1_step2,fs2.test(indice,:)');
            outputs2 = sim(net2_step2,fs2.test(indice,:)');
            outputs3 = sim(net3_step2,fs2.test(indice,:)');
            outputs4 = sim(net4_step2,fs2.test(indice,:)');

            newInputTest = [outputs1;outputs2;outputs3;outputs4];

            predict_label_2 = sim(combNet_step2,newInputTest);
            predict_label_2 = predict_label_2';
            
            predict_label_2 = round(predict_label_2);
            
            clear predict_matrix_2;
            predict_matrix_2 = round(predict_label_2);
            
            %predict_matrix_2=[];
            %    for i=1: size(predict_label_2,1)
            %    switch(predict_label_2(i))
            %        case 1,
            %            predict_matrix_2(i,:) = [1 0 0 0];            
            %        case 2,
            %            predict_matrix_2(i,:) = [0 1 0 0];            
            %        case 3,
            %            predict_matrix_2(i,:) = [0 0 1 0];            
            %        case 4,
            %            predict_matrix_2(i,:) = [0 0 0 1]; %

                %end
                %end

                tLabes = testingLabels(indice);
                clear label_matrix_2;
                label_matrix_2=[];
                for i=1: size(tLabes,2)
                switch(tLabes(i))
                    case 1,
                        label_matrix_2(i,:) = [1 0 0 0];            
                    case 2,
                        label_matrix_2(i,:) = [0 1 0 0];            
                    case 3,
                        label_matrix_2(i,:) = [0 0 1 0];            
                    case 4,
                        label_matrix_2(i,:) = [0 0 0 1]; 

                end
                end
                %[C,CM,IND,PER] = CONFUSION(TARGETS,OUTPUTS)
                if size(label_matrix_2,2)==0 || size(predict_matrix_2,2)==0
                    cm2 = zeros(4,4);
                    c2=0;
                else
                    [c2 cm2] = confusion(label_matrix_2', predict_matrix_2');
                end
            
   end
    
    %fprintf('\n----------------------- Classificador MLP comb ---------------------\n')
    
    cd ..
elseif(strcmp(classifier,'PNN')) 
	cd('pnn')

	if(first_time==1)
        first_time=0;
		if param1 == 0
			spread = 0.1;
		else
			spread = param1;		
		end
	  	[cm1 net1 predict_matrix label_matrix] = classifyPNN_ch1(fs1.train,target.train,fs1.test,target.test, spread);
        
        % ***** CLASSIFICA SEGUNDA ETAPA (apenas os batimentos classificados como arritmicos v�o para o segundo classificador)
	    gg=find(label_matrix(:,2)==1);
	    hh=find(predict_matrix(:,2)==1);

        predict_matrix=predict_matrix';
	    anormal_id = find(predict_matrix(:,2)==1 & label_matrix(:,2));

	    indice=[];
	    for i=1:size(anormal_id,1)
		indice(i) = find(gg==anormal_id(i));
	    end

    	[cm2 net2] = classifyPNN_ch2(fs2.train,target2.train,fs2.test(indice,:),target2.test(indice,:), spread);
        
    else

         % ***** CLASSIFICA PRIMEIRA ETAPA
            testingLabels = [];
            targetsTest = [];
            % formata para uso em SVM - para cada classe
            for t=1:size(target.test,2)
                targetsTest(find(target.test(:,t)==1),t)=t;     
            end
            testingLabels = sum(targetsTest');

            for(i=1:length(fs1.test(:,1)))
                for(j=1:length(fs1.test(1,:)))
                    %normaliza tb a matriz de teste
                    fs1.test(i,j) = tansig((fs1.test(i,j)-mean(fs1.test(i,:)))/std(fs1.test(i,:)));
                end
            end

            %predict_label = sim(combNet_step1,newInputTest);
            predict_label = sim(net1,fs1.test');

            predict_label = predict_label';
            
            clear predict_matrix;
            predict_matrix=[];
            for t=1:2
                predict_matrix(find(predict_label(:,t)==1),t)=1;     
            end
            %predict_label_vec = sum(predict_label_vec);
            
            %clear predict_matrix;
            %predict_matrix=[];

            
            %for i=1: size(predict_label,1)
            %switch(predict_label(i))
            %    case 1,
            %        predict_matrix(i,:) = [1 0];            
            %    case 2,
            %        predict_matrix(i,:) = [0 1];            
               
            %end
            %end

            clear label_matrix;
            label_matrix=[];
            for i=1: size(testingLabels,2)
            switch(testingLabels(i))
                case 1,
                    label_matrix(i,:) = [1 0];            
                case 2,
                    label_matrix(i,:) = [0 1];            
               
            end
            end
            %[C,CM,IND,PER] = CONFUSION(TARGETS,OUTPUTS)
            [c cm1] = confusion(label_matrix', predict_matrix');
            
            % **** CLASSIFICA SEGUNDA ETAPA
            
            % so classificaco quem foi considerado arritmico na etapa
            % anterior
            gg=find(label_matrix(:,2)==1);
            hh=find(predict_matrix(:,2)==1);
            
            %anormal_id = find(predict_matrix(:,2)==1 & label_matrix(:,2));
            anormal_id = find(predict_matrix(:,2)==1 & label_matrix(:,2));
            
            testingLabels = [];
            targetsTest = [];
            % formata para uso em SVM - para cada classe
            for t=1:size(target2.test,2)
                targetsTest(find(target2.test(:,t)==1),t)=t;     
            end
            testingLabels = sum(targetsTest');

            indice=[];
            for i=1:size(anormal_id,1)
                indice(i) = find(gg==anormal_id(i));
            end
        
            for(i=1:length(fs2.test(:,1)))
                for(j=1:length(fs2.test(1,:)))
                    %normaliza tb a matriz de teste
                    fs2.test(i,j) = tansig((fs2.test(i,j)-mean(fs2.test(i,:)))/std(fs2.test(i,:)));
                end
            end

            %predict_label = sim(combNet_step1,newInputTest);
            predict_label_2 = sim(net2,fs2.test(indice,:)');

            predict_label_2 = predict_label_2';
            clear predict_matrix_2;
            predict_matrix_2=[];
            
            for t=1:4
                predict_matrix_2(find(predict_label_2(:,t)==1),t)=1;     
            end
           % predict_label_2_vec = sum(predict_label_2_vec);
            
            %clear predict_matrix_2;
            %predict_matrix_2=[];
            %predict_label_2 = predict_label_2';
            
             %   for i=1: size(predict_label_2,1)
              %  switch(predict_label_2(i))
               %     case 1,
               %         predict_matrix_2(i,:) = [1 0 0 0];            
               %     case 2,
               %         predict_matrix_2(i,:) = [0 1 0 0];            
               %     case 3,
               %         predict_matrix_2(i,:) = [0 0 1 0];            
               %     case 4,
               %         predict_matrix_2(i,:) = [0 0 0 1]; 

%                end
 %               end

                tLabes = testingLabels(indice);
                clear label_matrix_2;
                label_matrix_2=[];
                for i=1: size(tLabes,2)
                switch(tLabes(i))
                    case 1,
                        label_matrix_2(i,:) = [1 0 0 0];            
                    case 2,
                        label_matrix_2(i,:) = [0 1 0 0];            
                    case 3,
                        label_matrix_2(i,:) = [0 0 1 0];            
                    case 4,
                        label_matrix_2(i,:) = [0 0 0 1]; 

                end
                end
                %[C,CM,IND,PER] = CONFUSION(TARGETS,OUTPUTS)
                %[c2 cm2] = confusion(label_matrix_2', predict_matrix_2');
                %[C,CM,IND,PER] = CONFUSION(TARGETS,OUTPUTS)
                if size(label_matrix_2,2)==0 || size(predict_matrix_2,2)==0
                    cm2 = zeros(4,4);
                    c2=0;
                else
                    [c2 cm2] = confusion(label_matrix_2', predict_matrix_2');
                end
                
    end

	cd ..
end

 % ***** Calcula estat�Sticas PRIMEIR ETAPA
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
   % **** Calcula estat�Sticas PARA SEGUNDA ETAPA
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
fprintf(arq,' \\caption{Matriz e confus�o etapa 1} \n');
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
fprintf(arq,' \\caption{Matriz e confus�o etapa 2} \n');
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

if(strcmp(classifier,'SVM'))
    fprintf(arq,'SVM parameters etapa1 - C = %6.2f, gamma= %6.2f',best_c, best_g);
    fprintf(arq,'SVM parameters etapa2- C = %6.2f, gamma= %6.2f',best_c_2, best_g_2);
end

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
