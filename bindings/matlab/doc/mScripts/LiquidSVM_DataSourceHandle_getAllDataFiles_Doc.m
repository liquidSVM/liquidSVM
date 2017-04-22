
 %% LiquidSVM_DataSourceHandle.getAllDataFiles
 % Constructor of the class DataFile
 %% Syntax
 %   getAllDataFiles(dataSourceHandleObj,key,value);
 %   dataSourceHandleObj.getAllDataFiles(key,value);
 %% Description
 % Method to scan the source property for available data files
 % matching the a LiquidSVM_DataFile format, which are currently all *.csv
 % and *.csv.gz files. This method is primary called by the
 % constructor LiquidSVM_DataSourceHandle. Matching data files are stored
 % in the 'fileList' property
 %% Input Arguments
 %
 %% optional
 % *key:* |'refresh'|
 % *value:* logical scalar defining whether the 'fileList'
 % property should be refreshed in case its already non empty
 % which is the case when getAllDataFiles was called already
 % Default: |false|
 %%
 % *key:* |'displayMode'|
 % *value:* Numeric scalar vaule defining the amount of
 % information which is displayed to the console.
 % Default: |0|
 %% Example:
 %%
 % url source
 dataSources1 = LiquidSVM_DataSourceHandle('http://www.isa.uni-stuttgart.de/liquidData','url');
 dataSources1.getAllDataFiles('display',2)
 %%
 % local source
 dataSources2 = LiquidSVM_DataSourceHandle(pwd,'local');
 dataSources1.getAllDataFiles('refresh',true)
 %% See also
 % LiquidSVM_DataSourceHandle
