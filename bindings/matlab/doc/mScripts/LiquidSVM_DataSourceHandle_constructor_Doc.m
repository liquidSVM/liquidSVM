
 %% LiquidSVM_DataSourceHandle (Constructor)
 % Constructor of the class DataSourceHandle
 %% Syntax
 %   dataSourceHandleObj = LiquidSVM_DataSourceHandle(source,type,key,value);
 %% Description
 %
 %% Input Arguments
 %
 %% required
 % |source|: Depending of the type this is eighter a path to a
 % local folder or the url address.
 %
 %%
 % |type|: The type of the data source. Valid values are 'local'
 % and 'url'
 %% optional
 % *key:* |'displayMode'|
 % *value:* Numeric scalar vaule defining the amount of
 % information which is displayed to the console.
 % Default: |0|
 %% Output:
 % |dataSourceHandleObj| a new object of class
 % LiquidSVM_DataSourceHandle. In case a invalid source is given,
 % an empty object is returned
 %% Example:
 %%
 % url source
 dataSources1 = LiquidSVM_DataSourceHandle('http://www.isa.uni-stuttgart.de/liquidData','url');
 %%
 % local source
 dataSources2 = LiquidSVM_DataSourceHandle(pwd,'local');
 %% See also
 % LiquidSVM_DataFile, LiquidSVM_Data
