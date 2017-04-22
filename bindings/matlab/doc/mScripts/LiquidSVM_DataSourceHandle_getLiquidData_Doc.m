
 %% LiquidSVM_DataSourceHandle.getLiquidData
 %% Syntax
 %   liquidDataObj = getLiquidData(dataSourceHandleObj,name,key,value);
 %   liquidDataObj = dataSourceHandleObj.getLiquidData(name,key,value);
 %% Description
 % Main method of the class, scanning the source for data matching
 % the given name and if a match is found the method looks for train
 % as well as for test data.
 %% Input Arguments
 %
 %% required
 % |name|: character string describing the name of the dataset one
 % looks for.
 %
 %% optional
 % *key:* |'displayMode'|
 % *value:* Numeric scalar vaule defining the amount of
 % information which is displayed to the console.
 % Default: |0|
 %% Output:
 % |liquidDataObj| an object of class LiquidSVM_Data
 %% Example:
 %%
 % url source
 dataSources1 = LiquidSVM_DataSourceHandle('http://www.isa.uni-stuttgart.de/liquidData','url');
 % pick index 1
 liquidDat1  = dataSources1.getLiquidData('crime');
 % set display
 liquidDat2 = dataSources1.getLiquidData('banana-bc','display',2);
 %% See also
 % LiquidSVM_DataFile, LiquidSVM_Data
