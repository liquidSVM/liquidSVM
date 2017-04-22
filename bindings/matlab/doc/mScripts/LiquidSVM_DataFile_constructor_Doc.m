
 %% LiquidSVM_DataFile (Constructor)
 % Constructor of the class DataFile
 %% Syntax
 %   dataFileObj = LiquidSVM_DataFile(name,ext,key,value);
 %% Description
 % long description
 %% Input Arguments
 %
 %% required
 % |name|: character string describing the name of the data file
 %
 %%
 % |ext|: Extention of the data file (allowed values are '.csv'
 % and '.csv.gz')
 %% optional
 % *key:* |'type'|
 % *value:* A character string describing the type of the data
 % file (supported values are 'train' and 'test')
 % Default: |''|
 %%
 % *key:* |'displayMode'|
 % *value:* Numeric scalar vaule defining the amount of
 % information which is displayed to the console.
 % Default: |0|
 %% Output:
 % |dataFileObj| a new object of class LiquidSVM_DataFile.
 %% Example:
 %%
 % default call
 LiquidSVM_DataFile('testFile','.csv')
 %%
 % specify type
 LiquidSVM_DataFile('testFile','.csv','type','test')
 %% See also
 %
