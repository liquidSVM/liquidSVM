
 %% LiquidSVM_DataSourceHandle.addToFileList
 % Method to add a object of class LiquidSVM_DataFile to the fileList
 % property
 %% Syntax
 %   addToFileList(dataSourceHandleObj,dataFileObj);
 %   dataSourceHandleObj.addToFileList(dataFileObj);
 %% Description
 % Method to add a object of class LiquidSVM_DataFile to the fileList
 % property
 %% Input Arguments
 %
 %% required
 % |dataFileObj|: object of class LiquidSVM_DataFile
 %% Example:
 %%
 %
 dataSources1 = LiquidSVM_DataSourceHandle('http://www.isa.uni-stuttgart.de/liquidData','url');
 testFile1 = LiquidSVM_DataFile('testFile','.csv','type','test');
 size(dataSources1.fileList)
 dataSources1.addToFileList(testFile1);
 size(dataSources1.fileList)
 %% See also
 % LiquidSVM_DataFile, LiquidSVM_Data
