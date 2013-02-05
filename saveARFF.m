function saveARFF(filename,wekaOBJ)
% Save a weka java Instances object, holding data, to a weka formatted ARFF
% file. You can first convert matlab data to a weka java Instances object
% using the matlab2weka function. 
% 
% filename - the path and filename for the target .arff file, including
%            extension. i.e. 'c:\data\mydata.arff'
%
% wekaOBJ  - a weka java Instances object storing the data. 
%
% Written by Matthew Dunham

WEKA_HOME = 'C:\Arquivos de programas\Weka-3-6';
javaaddpath([WEKA_HOME '\weka.jar']);

%    if(~wekaPathCheck),return,end
    import weka.core.converters.ArffSaver;
    import java.io.File;
    
    saver = ArffSaver();
    saver.setInstances(wekaOBJ);
    saver.setFile(File(filename));
    saver.writeBatch();

end