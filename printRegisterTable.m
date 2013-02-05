% Autores : Eduardo Luz 
% Função : Imprime tabele de registros em arquivo
function printRegisterTable(file_path)

s = char(file_path);
fileNamed = ['results\',s(10:end-1),'\regTable_5_classes.tex'];
arq = fopen(fileNamed,'w');

dataset1 = {'101';'106';'108';'109';'112';'114';'115';'116';'118';'119';'122';'124';'201';'203';'205';'207';'208';'209';'215';'220';'223';'230'};
dataset2 = {'100';'103';'105';'111';'113';'117';'121';'123';'200';'202';'210';'212';'213';'214';'219';'221';'222';'228';'231';'232';'233';'234'};

%----------------- inicializacao das variaveis
dados1  = [];
rotulo1 = [];
dados2  = [];
rotulo2 = [];
%-------------------------------

% Tabela latex dos registros
fprintf(arq,'\\documentclass{article}\n');
fprintf(arq,'\\usepackage{graphicx}\n');
fprintf(arq,'\\usepackage[latin1]{inputenc}\n');
fprintf(arq,'\\usepackage{tabularx}\n');
fprintf(arq,'\\usepackage{multirow}\n');
fprintf(arq,'\\usepackage[dvips]{color}');
fprintf(arq,'\\usepackage{colortab}');

fprintf(arq,'\\definecolor{lightgray}{gray}{0.8}');
fprintf(arq,'\\definecolor{darkgray}{gray}{0.5}');

fprintf(arq,'\\newcommand{\\lgray}{\\color{lightgray}}');
fprintf(arq,'\\newcommand{\\dgray}{\\color{darkgray}}');

fprintf(arq,'\\newcommand{\\citep}{\\cite}\n');
fprintf(arq,'\\newcommand{\\citet}{\\cite}\n');
fprintf(arq,'\\newcommand{\\TFigure}{Fig.}\n');
fprintf(arq,'\\begin{document}\n');

fprintf(arq,'\n');
fprintf(arq,'\\begin{table*} \n');
s2 = char('\\caption{Tebela dos registros do método ');
s2 = [s2 s(10:end-1) '} \n'];
fprintf(arq,s2);
%fprintf(arq,' \\caption{Tebela dos registros do método} \n');
fprintf(arq,' \\label{tab:regtable} \n');
fprintf(arq,' \\begin{center} \n');
fprintf(arq,'  \\begin{tabular}{|c|c|c|c|c|c|c|} \n');
fprintf(arq,'   \\hline \n');
fprintf(arq,'    Registro & N & SVEB & VEB & F & Q & Total \\\\ \n');
fprintf(arq,'   \\hline \n');

%carrega dos arquivos e separa apenas os batimentos desejados
for i=1:size(dataset1,1)
%    if nargin < 4
 %       dataset1(i) = str2double(dataset1(i));
  %  end
    filename = strcat(dataset1(i),'.txt');
    filename = strcat(file_path,filename);
    [data target] = loadarq(char(filename));
    
    sN = size(find(target(:,1)==1),1);
    sSVEB = size(find(target(:,2)==1),1);
    sVEB = size(find(target(:,3)==1),1);
    sF = size(find(target(:,4)==1),1);
    sQ = size(find(target(:,5)==1),1);
    total = sN + sSVEB + sVEB + sF + sQ;
    
    fprintf(arq,'\\textbf{ %6d } & %6.0f & %6.0f & %6.0f & %6.0f & %6.0f & %6.0f & %6.0f \\\\ \n', str2double(dataset1(i)), sN, sSVEB, sVEB, sF, sQ, total );
    fprintf(arq,'\n');
    
    dados1 = [dados1;data];
    rotulo1 = [rotulo1;target];    
end

for i=1:size(dataset2,1)
 %   if nargin < 4
  %      dataset2(i) = str2double(dataset2(i));
   % end
    filename = strcat(dataset2(i),'.txt');
    filename = strcat(file_path,filename);
    [data target] = loadarq(char(filename));
    
    sN = size(find(target(:,1)==1),1);
    sSVEB = size(find(target(:,2)==1),1);
    sVEB = size(find(target(:,3)==1),1);
    sF = size(find(target(:,4)==1),1);
    sQ = size(find(target(:,5)==1),1);
    total = sN + sSVEB + sVEB + sF + sQ;
    
    fprintf(arq,'%6d & %6.0f & %6.0f & %6.0f & %6.0f & %6.0f & %6.0f  \\\\ \n', str2double(dataset2(i)), sN, sSVEB, sVEB, sF, sQ, total );
    fprintf(arq,'\n');
    
    dados2 = [dados2;data];
    rotulo2 = [rotulo2;target];    
end

fprintf(arq,'    \\hline \n');
fprintf(arq,'    \\hline \n');
%fprintf(arq,'  \\begin{tabular}{|c|c|c|c|c|c|} \n')

sN1 = size(find(rotulo1(:,1)==1),1);
sSVEB1 = size(find(rotulo1(:,2)==1),1);
sVEB1 = size(find(rotulo1(:,3)==1),1);
sF1 = size(find(rotulo1(:,4)==1),1);
sQ1 = size(find(rotulo1(:,5)==1),1);
total1 = sN1 + sSVEB1 + sVEB1 + sF1 + sQ1;

sN2 = size(find(rotulo2(:,1)==1),1);
sSVEB2 = size(find(rotulo2(:,2)==1),1);
sVEB2 = size(find(rotulo2(:,3)==1),1);
sF2 = size(find(rotulo2(:,4)==1),1);
sQ2 = size(find(rotulo2(:,5)==1),1);
total2 = sN2 + sSVEB2 + sVEB2 + sF2 + sQ2;

fprintf(arq,'\\textbf{DS1}  & %6.0f & %6.0f & %6.0f & %6.0f & %6.0f & %6.0f\\\\ \n',  sN1, sSVEB1, sVEB1, sF1, sQ1, total1 );
fprintf(arq,'DS2   & %6.0f & %6.0f & %6.0f & %6.0f & %6.0f & %6.0f \\\\ \n', sN2, sSVEB2, sVEB2, sF2, sQ2, total2 );
fprintf(arq,'TOTAL & %6.0f & %6.0f & %6.0f & %6.0f & %6.0f & %6.0f \\\\ \n' , sN1+sN2, sSVEB1+sSVEB2, sVEB1+sVEB2, sF1+sF2, sQ1+sQ2, total1+total2 );
    
fprintf(arq,'   \\hline \n');
fprintf(arq,'  \\end{tabular} \n');
fprintf(arq,' \\end{center} \n');
fprintf(arq,'\\end{table*} \n');
fprintf(arq,'\n');
fprintf(arq,'\n');
fprintf(arq,'\\end{document}\n');
fclose(arq);

end

%-------------------- Função de separação dos tipos
% os rótulos estão em ascii code
% Seguindo o padrão AAMI heartbeat classes, Normal (N), supraventricular
% ectopic beat (SVEB), Ventricular ectopic beat(VEB), fusion beat(F) e
% unknown beat (Q)
% 
function [data target]=loadarq(arq)
amostra = load(arq);
tam = size(amostra,2);

pos = find(amostra(:,tam)==78); % N -> N
N = amostra(pos,1:tam-1);
sz = size(N,1);
tN = repmat([1 0 0 0 0],sz,1); 

pos = find(amostra(:,tam)==76); % L -> N
L = amostra(pos,1:tam-1);
sz = size(L,1);
tL = repmat([1 0 0 0 0],sz,1); 

pos = find(amostra(:,tam)==82);  % R -> N
R  = amostra(pos,1:tam-1);
sz = size(R,1);
tR = repmat([1 0 0 0 0],sz,1); 

pos = find(amostra(:,tam)==65); % A -> SVEB
A  = amostra(pos,1:tam-1);
sz = size(A,1);
tA = repmat([0 1 0 0 0],sz,1); 

pos = find(amostra(:,tam)==86);  % V -> VEB
V  = amostra(pos,1:tam-1);
sz = size(V,1);
tV = repmat([0 0 1 0 0],sz,1); 

pos = find(amostra(:,tam)==47);  % paced (/) -> Q
P  = amostra(pos,1:tam-1);
sz = size(P,1);
tP = repmat([0 0 0 0 1],sz,1); 

pos = find(amostra(:,tam)==97);  % a -> SVEB
a  = amostra(pos,1:tam-1);
sz = size(a,1);
ta = repmat([0 1 0 0 0],sz,1); 

pos = find(amostra(:,tam)==70);  % F -> F
F = amostra(pos,1:tam-1);
sz = size(F,1);
tF = repmat([0 0 0 1 0 ],sz,1); 

pos = find(amostra(:,tam)==106);  % j -> N
j  = amostra(pos,1:tam-1);
sz = size(j,1);
tj = repmat([1 0 0 0 0],sz,1); 

pos = find(amostra(:,tam)==102);  % f -> Q
f = amostra(pos,1:tam-1);
sz = size(f,1);
tf = repmat([0 0 0 0 1],sz,1); 

pos = find(amostra(:,tam)==69);  % E -> VEB
E  = amostra(pos,1:tam-1);
sz = size(E,1);
tE = repmat([0 0 1 0 0],sz,1); 

pos = find(amostra(:,tam)==74);  % J -> SVEB
J  = amostra(pos,1:tam-1);
sz = size(J,1);
tJ = repmat([0 1 0 0 0],sz,1); 

pos = find(amostra(:,tam)==101);  % e -> N
e  = amostra(pos,1:tam-1);
sz = size(e,1);
te = repmat([1 0 0 0 0],sz,1); 

pos = find(amostra(:,tam)==83);  % S -> SVEB
S  = amostra(pos,1:tam-1);
sz = size(S,1);
tS = repmat([0 1 0 0 0],sz,1); 

pos = find(amostra(:,tam)==81);  % Q -> Q
Q  = amostra(pos,1:tam-1);
sz = size(Q,1);
tQ = repmat([0 0 0 0 1],sz,1);

data = [N;L;R;A;V;P;a;F;S;j;f;E;J;e;Q];
target = [tN;tL;tR;tA;tV;tP;ta;tF;tS;tj;tf;tE;tJ;te;tQ];
end

