
 %% LiquidSVM_DataFile.fileName
 % short description
 %% Syntax
 %   obj = BHCS_ClassDefTemplate(key,value);
 %% Description
 % method to return the name of the dataset. If the type
 % property is nonempty, the return value is
 % [dataFileObj.name,'.',dataFileObj.type] and [dataFileObj.name]
 % otherwise.
 %% Output:
 % |fileName| character string.
 %% Example:
 %%
 file1 = LiquidSVM_DataFile('testFile1','.csv');
 file1.fileName
 %%
 file2 = LiquidSVM_DataFile('testFile2','.csv','type','test')
 file2.fileName
 %% See also
 % LiquidSVM_DataFile
