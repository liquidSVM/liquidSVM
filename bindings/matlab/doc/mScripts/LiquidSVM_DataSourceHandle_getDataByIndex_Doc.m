
 %% LiquidSVM_DataSourceHandle.getDataByIndex
 %% Syntax
 %   dataMat = getDataByIndex(dataSourceHandleObj,index,key,value);
 %   dataMat = dataSourceHandleObj.getDataByIndex(index,key,value);
 %% Description
 % Method to actually read out the data from a given dataFile.
 % This differs depending on the source type and dataFile ext.
 % This method is primary called by the getLiquidData method
 %% Input Arguments
 %
 %% required
 % |index|: numerical scalar defining which element in the
 % fileList property should be pcked.
 %
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
 % pick index 1
 dataMat1 = dataSources1.getDataByIndex(1);
 size(dataMat1)
 %%
 % set display
 dataMat2 = dataSources1.getDataByIndex(2,'display',2);
 size(dataMat2)
 %% See also
 % LiquidSVM_DataFile, LiquidSVM_Data
