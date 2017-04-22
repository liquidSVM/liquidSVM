classdef Liquid_Data < matlab.mixin.Copyable
% %%
% % -----------------------------------------------------------------------
% % ----------------- classDef Liquid_Data -----------------------------
% % -----------------------------------------------------------------------
% % HEADER:
% %
% % 	autor:       nico schmid           
% % 	autor email: n_schmid@gmx.de      
% % 	create date:       
% % 	version:     0.1
% % 	update:				
% % 	remark:            
% % 
% % -----------------------------------------------------------------------
% % DESCRIPTION:
% % Basic class to ease the handling of data and to provide a natural input 
% % data Type for the functionalty provided by the SimonsSVM toolbox.
% % 
% % Liquid_Data Properties:
% % name        - character string descibing the name of the data file
% % trainData   - double matrix containing the training data 
% % testData    - double matrix containing the test data Note that in 
% % contrast to the trainData this property might be empty 
% % in cases where no testing data is given.
% % responseCol - numeric scalar identifing the column in the data which 
% % is will be taken as the response variable. 
% % header      - A logical scalar, describing whether or not a header is 
% % given. (this property is not used so far)
% %
% % -----------------------------------------------------------------------
% % -----------------------------------------------------------------------
% %
% %    
% % Detail information see:   
% % 
% %                         !!!<a href="matlab:open(fullfile('html','Liquid_Data_Doc.html'))">Liquid_Data</a>!!!
% %
% % 
% % See also: Liquid_DataSourceHandle
% %
% % Copyright 2015-2016 Nico Schmid
% % see the COPYING file included with this software
    

    properties
        % character string descibing the name of the data file
        name
        % A logical scalar, describing whether or not a header is given.
        %   (this property is not used so far) 
        header
        % double matrix containing the training data
        trainData
        % double matrix containing the test data Note that in contrast
        %   to the trainData this property might be empty in cases where no testing
        %   data is given.
        testData
        % numeric scalar identifing the column in the data which is
        % will be taken as the response variable. 
        responseCol
    end
    
    methods
        function liquidDataObj = Liquid_Data(varargin)
        % %% Liquid_Data (Constructor)
        % % Constructor of the class Liquid_Data
        % 
        % %% Syntax
        % %   liquidDataObj = Liquid_Data(key,value);      
        % 
        % %% Description
        % % 
        % 
        % %% Input Arguments      
        % 
        % %% optional
        %
        % % *key:* |'name'| 
        % % *value:* character string of the name of the dataset one is
        % % looking for. 
        % % Default: |''|
        % %%
        %
        % % *key:* |'source'| 
        % % *value:* Eighter a character string or an object of class
        % % Liquid_DataSourceHandle. It's defining the search paths in which
        % % the dataset is loaded from.  
        % %%        
        % % Default: 
        % %%
        % %   |defaultDataSources{1} = Liquid_DataSourceHandle(fullfile('..','..','data'),'local');| 
        % %   |defaultDataSources{2} = Liquid_DataSourceHandle(fullfile('..','data'),'local');| 
        % %   |defaultDataSources{3} = Liquid_DataSourceHandle(fullfile('data'),'local');| 
        % %   |defaultDataSources{4} = Liquid_DataSourceHandle(pwd,'local');| 
        % %   |defaultDataSources{5} = Liquid_DataSourceHandle('http://www.isa.uni-stuttgart.de/liquidData','url');|         
        % %%
        %
        % % *key:* |'data'| 
        % % *value:* Double Matrix which is eighter eighter be taken as
        % % trainData or splitted ijnto train and testdata according to
        % % input parameter 'split' and 'ratio'
        % % Default: |[]|        
        % %%
        %
        % % *key:* |'split'| 
        % % *value:* Character string defining how a given 'data' is
        % % handeled. Supported values is 'random' and 'non'. 
        % % Default: |'non'|         
        % %%
        %
        % % *key:* |'ratio'| 
        % % *value:* Numeric value Defining which ration the data is
        % % splitted into train and test sets if 'split' is not 'non' 
        % % If e.g. split='random' and 'ratio'= 9 than 90 percent of the
        % % data is taken as trainData and 10 as testData. Default is 2,
        % % meaning half of the data is for training, the other half for
        % % testing.
        % % Default: |2|   
        % %%
        %
        % % *key:* |'responseCol'| 
        % % *value:* Integer value defining which cloum of the data
        % % is taken to be the respose variable
        % % Default: |1|  
        % %%
        %
        % % *key:* |'trainData'| 
        % % *value:* Double Matrix which is taken as
        % % trainData 
        % % Default: |[]|  
        % %%
        %
        % % *key:* |'testData'| 
        % % *value:* Double Matrix which is taken as
        % % testData 
        % % Default: |[]|  
        % %%
        % %
        % %% Output:
        % % |liquidDataObj| a new object of class Liquid_Data      
        % %
        % 
        % %% Example:
        % %%
        % % Most straigt forward way (just provide train data)
        % %    myTrainData = [[rand(100,1)>.8;rand(100,1)>.2],[rand(100,3)+.5;rand(100,3)-.5]];
        % %    myRandomLiquid_Data = Liquid_Data('trainData',myTrainData)
        % %    % which is the same as 
        % %    myRandomData = Liquid_Data('data',myTrainData,'split','non')
        % %% 
        % % Provide train and test data
        % %    myTestData = [[rand(100,1)>.8;rand(100,1)>.2],[rand(100,3)+.5;rand(100,3)-.5]];
        % %    myRandomLiquid_Data = Liquid_Data('trainData',myTrainData,'testData',myTestData)
        % %%
        % % Provide train and test data by splitting one dataset in half
        % %    myData = [[rand(100,1)>.8;rand(100,1)>.2],[rand(100,3)+.5;rand(100,3)-.5]];
        % %    myRandomLiquid_Data = Liquid_Data('data',myData,'split','random')
        % %%
        % % Provide train and test data by splitting one dataset in specified
        % % ratio
        % %    myData = [[rand(100,1)>.8;rand(100,1)>.2],[rand(100,3)+.5;rand(100,3)-.5]];
        % %    myRandomLiquid_Data = Liquid_Data('data',myData,'split','random','ratio',9)
        % %%  
        % % Provide train and test data by specifing name of the data set and
        % % loading the data from default source
        % %    myRandomLiquid_Data = Liquid_Data('name','crime') 
        % %%   
        % % Provide train and test data by loading data from specified source
        % % |myRandomLiquid_Data = Liquid_Data('name','crime','source','http://www.isa.uni-stuttgart.de/liquidData') |
        % % this is the same as:
        % %    myDataSource = Liquid_DataSourceHandle('http://www.isa.uni-stuttgart.de/liquidData','url')
        % %    myRandomLiquid_Data = myDataSource.getLiquidData('crime')
        %
        % %% See also
        % % Liquid_DataFile, Liquid_Data                
            
            
            % check input parameter
            p = inputParser();
            addParamValue(p,'name','',@ischar)
            addParamValue(p,'header',false,@(x) islogical(x) && isscalar(x))
            addParamValue(p,'responseCol',1,@(x) isnumeric(x) && isscalar(x))
            addParamValue(p,'data',[],@isnumeric)
            addParamValue(p,'split','non',@ischar)  
            addParamValue(p,'ratio',2,@(x) isnumeric(x) && isscalar(x))
            addParamValue(p,'source','',@(x)Liquid_Data.checkValidSource(x))
            addParamValue(p,'trainData',[],@isnumeric)
            addParamValue(p,'testData',[],@isnumeric)
            parse(p,varargin{:})  
            
            % first look if a name is given
            if ~isempty(p.Results.name)
                % If a name is determine the source in which the name
                % should be looked up
                
                if isempty(p.Results.source)
                    % Define default Sources
                    defaultDataSources(1) = Liquid_DataSourceHandle(fullfile('..','..','data'),'local');
                    defaultDataSources(2) = Liquid_DataSourceHandle(fullfile('..','data'),'local');
                    defaultDataSources(3) = Liquid_DataSourceHandle(fullfile('data'),'local');
                    defaultDataSources(4) = Liquid_DataSourceHandle(pwd,'local');
                    defaultDataSources(5) = Liquid_DataSourceHandle('http://www.isa.uni-stuttgart.de/liquidData','url');                    
                    source = defaultDataSources(~arrayfun(@isempty,defaultDataSources));
                elseif ischar(p.Results.source)
                    localSource = Liquid_DataSourceHandle(source,'local');
                    urlSource = Liquid_DataSourceHandle(source,'url');
                    source = [localSource,urlSource];
                    source = source(~arrayfun(@isempty,source));
                elseif isa(p.Results.source,'Liquid_DataSourceHandle') && ~isempty(p.Results.source)
                    source = p.Results.source;
                else
                    warning('Data source is empty')
                end            

                % look in the source if a dataset with the given name is 
                % available 
                liquidDataObj = Liquid_Data.empty; 
                k = 1;
                while  (isempty(liquidDataObj) && k <= length(source)) %~exist('liquidDataObj','var') ||
                    liquidDataObj = source(k).getLiquidData(p.Results.name);
                    k = k+1;
                end
                if isempty(liquidDataObj)
                    warning('data with name %s was not found in given data source(s)',p.Results.name)
                end
            end            
            
            % if data is provided 
            if ~isempty(p.Results.data)           
                switch p.Results.split
                    case 'random'
                        dataSize   = size(p.Results.data,1);
                        trainIndex = true(dataSize,1);
                        mixIndex =  randperm(dataSize);
                        trainIndex(mixIndex(1:floor(dataSize/p.Results.ratio))) = false;
                        %trainIndex(randsample(dataSize,floor(dataSize/p.Results.ratio))) = false;
                        liquidDataObj = Liquid_Data('trainData',p.Results.data(trainIndex,:),'testData',p.Results.data(~trainIndex,:));
                    case 'non'
                        liquidDataObj = Liquid_Data('trainData',p.Results.data);
                    otherwise
                        error('No valid split method')
                end
                if ~isempty(p.Results.name)
                    liquidDataObj.name =  p.Results.name;
                end
            end            

            % if trainData is given
            if ~isempty(p.Results.trainData)
                liquidDataObj.trainData  = p.Results.trainData;
                if ~isempty(p.Results.testData)
                    if size(liquidDataObj.trainData,2) == size(p.Results.testData,2)
                        liquidDataObj.testData  = p.Results.testData;
                    end
                end                                
            end
            % Check responseCol and store to property via method
            % checkValidResponseCol
            liquidDataObj.checkValidResponseCol(p.Results.responseCol);
            % 
            liquidDataObj.header     =  p.Results.header;
        end
        
        function trainFeaturMat = trainFeatures(liquidDataObj)
        % %% trainFeatures 
        % % Method to get the training features
        % 
        % %% Syntax
        % %   trainFeaturMat = trainFeatures(liquidDataObj);
        % %   trainFeaturMat = liquidDataObj.trainFeatures();
        % 
        % %% Description
        % % 
        % 
        % %% Output:
        % % |trainFeaturMat| double matrix containing the training features     
        % %
        % 
        % %% Example:
        % %%
        % %    banana = liquidData('banana-mc');
        % %    trainFeat = banana.trainFeatures;
        %
        % %% See also
        % % Liquid_DataFile, Liquid_Data             
            ncol = size(liquidDataObj.trainData,2);
            trainFeaturMat = liquidDataObj.trainData(:,setdiff(1:ncol,liquidDataObj.responseCol));
        end
    
        function trainLabelVec = trainLabel(liquidDataObj)
        % %% trainLabel 
        % % Method to get the training labels
        % 
        % %% Syntax
        % %   trainFeaturMat = trainLabel(liquidDataObj);
        % %   trainFeaturMat = liquidDataObj.trainLabel();
        % 
        % %% Description
        % % 
        % 
        % %% Output:
        % % |trainLabelVec| vector containing the training labels     
        % %
        % 
        % %% Example:
        % %%
        % %    banana = liquidData('banana-mc');
        % %    trainFeat = banana.trainLabel;
        %
        % %% See also
        % % Liquid_DataFile, Liquid_Data  

            trainLabelVec = liquidDataObj.trainData(:,liquidDataObj.responseCol);
        end
        
        function testFeaturMat = testFeatures(liquidDataObj)
        % %% trainFeatures 
        % % Method to get the testing features
        % 
        % %% Syntax
        % %   testFeaturMat = testFeatures(liquidDataObj);
        % %   testFeaturMat = liquidDataObj.testFeatures();
        % 
        % %% Description
        % % 
        % 
        % %% Output:
        % % |testFeaturMat| double matrix containing the testing features     
        % %
        % 
        % %% Example:
        % %%
        % %    banana = liquidData('banana-mc');
        % %    testFeat = banana.testFeatures;
        %
        % %% See also
        % % Liquid_DataFile, Liquid_Data    
            
            if isempty(liquidDataObj.testData)
                error('liquidDataObj does not contain any test data');
            end
        
            ncol = size(liquidDataObj.testData,2);
            testFeaturMat = liquidDataObj.testData(:,setdiff(1:ncol,liquidDataObj.responseCol));
        end
        
        function testLabelVec = testLabel(liquidDataObj)
        % %% testLabel 
        % % Method to get the testing labels
        % 
        % %% Syntax
        % %   testFeaturMat = testLabel(liquidDataObj);
        % %   testFeaturMat = liquidDataObj.testLabel();
        % 
        % %% Description
        % % 
        % 
        % %% Output:
        % % |testLabelVec| vector containing the testing labels     
        % %
        % 
        % %% Example:
        % %%
        % %    banana = liquidData('banana-mc');
        % %    trainFeat = banana.testLabel;
        %
        % %% See also
        % % Liquid_DataFile, Liquid_Data  

            if isempty(liquidDataObj.testData)
                error('liquidDataObj does not contain any test data');
            end
        
            testLabelVec = liquidDataObj.testData(:,liquidDataObj.responseCol);
        end
        
        function trainData = train(liquidDataObj)
            trainData = copy(liquidDataObj);
            trainData.testData = [];
        end
        function trainData = test(liquidDataObj)
            trainData = copy(liquidDataObj);
            trainData.trainData = [];
        end
    end        
    methods(Hidden)
        function checkValidResponseCol(liquidDataObj,colNr)
            if (colNr < 1) || (colNr > size(liquidDataObj.trainData,2))
                warning('Provided responseCol does not fit to the size of the trainData and is set to 1')
                liquidDataObj.responseCol = 1;
            end
            liquidDataObj.responseCol = colNr;
        end
    end
        
    methods(Static, Hidden)
        function check = checkValidSource(source) 
            %disp('checking');    
            check = isa(source,'Liquid_DataSourceHandle') ||...
                    ~isempty(Liquid_DataSourceHandle(source,'local')) ||...
                    ~isempty(Liquid_DataSourceHandle(source,'url')) ||...
                    isempty(source);
            if ~check
                error('No valid data source. source must eighter a Liquid_DataSourceHandle object or specify a path to a vailid data source')
            end
        end
    end
end



