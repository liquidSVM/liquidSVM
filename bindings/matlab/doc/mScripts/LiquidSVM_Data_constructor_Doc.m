
 %% LiquidSVM_Data (Constructor)
 % Constructor of the class LiquidSVM_Data
 %% Syntax
 %   liquidDataObj = LiquidSVM_Data(key,value);
 %% Description
 %
 %% Input Arguments
 %% optional
 % *key:* |'name'|
 % *value:* character string of the name of the dataset one is
 % looking for.
 % Default: |''|
 %%
 % *key:* |'source'|
 % *value:* Eighter a character string or an object of class
 % LiquidSVM_DataSourceHandle. It's defining the search paths in which
 % the dataset is loaded from.
 %%
 % Default:
 %%
 %   |defaultDataSources{1} = LiquidSVM_DataSourceHandle(fullfile('..','..','data'),'local');|
 %   |defaultDataSources{2} = LiquidSVM_DataSourceHandle(fullfile('..','data'),'local');|
 %   |defaultDataSources{3} = LiquidSVM_DataSourceHandle(fullfile('data'),'local');|
 %   |defaultDataSources{4} = LiquidSVM_DataSourceHandle(pwd,'local');|
 %   |defaultDataSources{5} = LiquidSVM_DataSourceHandle('http://www.isa.uni-stuttgart.de/liquidData','url');|
 %%
 % *key:* |'data'|
 % *value:* Double Matrix which is eighter eighter be taken as
 % trainData or splitted ijnto train and testdata according to
 % input parameter 'split' and 'ratio'
 % Default: |[]|
 %%
 % *key:* |'split'|
 % *value:* Character string defining how a given 'data' is
 % handeled. Supported values is 'random' and 'non'.
 % Default: |'non'|
 %%
 % *key:* |'ratio'|
 % *value:* Numeric value Defining which ration the data is
 % splitted into train and test sets if 'split' is not 'non'
 % If e.g. split='random' and 'ratio'= 9 than 90 percent of the
 % data is taken as trainData and 10 as testData. Default is 2,
 % meaning half of the data is for training, the other half for
 % testing.
 % Default: |2|
 %%
 % *key:* |'responseCol'|
 % *value:* Integer value defining which cloum of the data
 % is taken to be the respose variable
 % Default: |1|
 %%
 % *key:* |'trainData'|
 % *value:* Double Matrix which is taken as
 % trainData
 % Default: |[]|
 %%
 % *key:* |'testData'|
 % *value:* Double Matrix which is taken as
 % testData
 % Default: |[]|
 %%
 %
 %% Output:
 % |liquidDataObj| a new object of class LiquidSVM_Data
 %
 %% Example:
 %%
 % Most straigt forward way (just provide train data)
 myTrainData = [[rand(100,1)>.8;rand(100,1)>.2],[rand(100,3)+.5;rand(100,3)-.5]];
 myRandomLiquidSVM_Data = LiquidSVM_Data('trainData',myTrainData)
 % which is the same as
 myRandomData = LiquidSVM_Data('data',myTrainData,'split','non')
 %%
 % Provide train and test data
 myTestData = [[rand(100,1)>.8;rand(100,1)>.2],[rand(100,3)+.5;rand(100,3)-.5]];
 myRandomLiquidSVM_Data = LiquidSVM_Data('trainData',myTrainData,'testData',myTestData)
 %%
 % Provide train and test data by splitting one dataset in half
 myData = [[rand(100,1)>.8;rand(100,1)>.2],[rand(100,3)+.5;rand(100,3)-.5]];
 myRandomLiquidSVM_Data = LiquidSVM_Data('data',myData,'split','random')
 %%
 % Provide train and test data by splitting one dataset in specified
 % ratio
 myData = [[rand(100,1)>.8;rand(100,1)>.2],[rand(100,3)+.5;rand(100,3)-.5]];
 myRandomLiquidSVM_Data = LiquidSVM_Data('data',myData,'split','random','ratio',9)
 %%
 % Provide train and test data by specifing name of the data set and
 % loading the data from default source
 myRandomLiquidSVM_Data = LiquidSVM_Data('name','crime')
 %%
 % Provide train and test data by loading data from specified source
 % |myRandomLiquidSVM_Data = LiquidSVM_Data('name','crime','source','http://www.isa.uni-stuttgart.de/liquidData') |
 % this is the same as:
 myDataSource = LiquidSVM_DataSourceHandle('http://www.isa.uni-stuttgart.de/liquidData','url')
 myRandomLiquidSVM_Data = myDataSource.getLiquidData('crime')
 %% See also
 % LiquidSVM_DataFile, LiquidSVM_Data
