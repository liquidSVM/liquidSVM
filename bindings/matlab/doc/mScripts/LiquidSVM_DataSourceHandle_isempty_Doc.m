
 %% LiquidSVM_DataSourceHandle.isempty
 %% Syntax
 %   logicalVal = isempty(dataSourceHandleObj);
 %   logicalVal = dataSourceHandleObj.isempty();
 %% Description
 % Method to determine if a object is empty
 %% Output:
 % |ret| logical scalar
 %% Example:
 %%
 % url source
 dataSources1 = LiquidSVM_DataSourceHandle('http://www.isa.uni-stuttgart.de/liquidData','url');
 isempty(dataSources1)
 %%
 % local source
 dataSources2 = LiquidSVM_DataSourceHandle(pwd,'local');
 isempty(dataSources2)
 %% See also
 % LiquidSVM_DataFile, LiquidSVM_Data
