classdef Liquid_DataSourceHandle < handle
% %%
% % -----------------------------------------------------------------------
% % ----------------- classDef Liquid_DataSourceHandle -----------------
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
% % Class to handle 'local' as well as 'url' sources of data files. 
% %
% % Liquid_DataSourceHandle Properties:
% % type - The type of the data source. Valid values are 'local' and 'url'
% % source - depending of the type this is eighter a path to a local
% % folder or the url address.
% % fileList - Contains all files included in the source matching the
% % Liquid_DataFile format, which are currently all *.csv and *.csv.gz files.
% % 
% %
% % -----------------------------------------------------------------------
% % -----------------------------------------------------------------------
% %
% %    
% % Detail information see:   
% % 
% %                         !!!<a href="matlab:open(fullfile('html','Liquid_DataSourceHandle_Doc.html'))">Liquid_DataSourceHandle</a>!!!
% %
% % 
% % See also: Liquid_DataSourceHandle
% %
% % Copyright 2015-2016 Nico Schmid
% % see the COPYING file included with this software
% %
    properties
        % The type of the data source. Valid values are 'local' and 'url'
        type
        % Depending of the type this is eighter a path to a local
        % folder or the url address.
        source
        % Contains all files included in the source matching the
        % Liquid_DataFile format, which are currently all *.csv and *.csv.gz files.
        fileList
    end
    
    methods
        function dataSourceHandleObj = Liquid_DataSourceHandle(source,type,varargin)
        % %% Liquid_DataSourceHandle (Constructor)
        % % Constructor of the class DataSourceHandle
        % 
        % %% Syntax
        % %   dataSourceHandleObj = Liquid_DataSourceHandle(source,type,key,value);
        % 
        % %% Description
        % % 
        % 
        % %% Input Arguments
        % %
        % %% required
        % % |source|: Depending of the type this is eighter a path to a 
        % % local folder or the url address.
        % % 
        % %%
        % % |type|: The type of the data source. Valid values are 'local' 
        % % and 'url'       
        % 
        % %% optional
        %
        % % *key:* |'displayMode'| 
        % % *value:* Numeric scalar vaule defining the amount of
        % % information which is displayed to the console.
        % % Default: |0|
        %
        % %% Output:
        % % |dataSourceHandleObj| a new object of class
        % % Liquid_DataSourceHandle. In case a invalid source is given, 
        % % an empty object is returned       
        %
        % %% Example:
        % %%
        % %    % url source
        % %    dataSources1 = Liquid_DataSourceHandle('http://www.isa.uni-stuttgart.de/liquidData','url');
        % %% 
        % %    % local source
        % %    dataSources2 = Liquid_DataSourceHandle(pwd,'local');       
        % 
        % %% See also
        % % Liquid_DataFile, Liquid_Data             
        
            % check and confirm input parameter
            p = inputParser();
            p.FunctionName = 'Liquid_DataSourceHandle';
            addRequired(p,'type',@(varargin)Liquid_DataSourceHandle.checkValidType(varargin{:}))
            addParamValue(p,'displayMode',0,@(x) isscalar(x) && isnumeric(x))
            parse(p,type,varargin{:})
            displayMode = p.Results.displayMode;
            % store input parameter type to property
            dataSourceHandleObj.type = p.Results.type;
            % depending on the type of the source check the source 
            dataSourceHandleObj.source = '';
            switch dataSourceHandleObj.type
                case 'local'                    
                    if exist(source,'dir')
                        dataSourceHandleObj.source = source;
                    else
                       %dataSourceHandleObj = Liquid_DataSourceHandle('','empty'); %Liquid_DataSourceHandle.empty;
                       if displayMode > 0
                           warning('No valid source, empty DataSourceHandle returned')
                       end
                       dataSourceHandleObj.type = 'empty';
                       dataSourceHandleObj.source = '';                       
                       return
                    end
                case 'url'
                    try
                        urlread(source);
                        dataSourceHandleObj.source = source;
                    catch
                        if displayMode > 0
                            warning('No valid source, empty DataSourceHandle returned')
                        end
                       dataSourceHandleObj.type = 'empty';
                       dataSourceHandleObj.source = '';                        
                        return
                    end
                case 'empty'
                    return
                otherwise
                    warning('Should not happen due to input parameter check')                    
            end
            % init property fileList with an empty Liquid_DataFile array             
            dataSourceHandleObj.fileList = {}; %Liquid_DataFile.empty;
            % call method getAllDataFiles to get all available *.csv and
            % *.csv.gz files in the source 
            dataSourceHandleObj.getAllDataFiles('displayMode',displayMode)
        end
        
        function getAllDataFiles(dataSourceHandleObj,varargin)
        % %% Liquid_DataSourceHandle.getAllDataFiles 
        % % Constructor of the class DataFile
        % 
        % %% Syntax
        % %   getAllDataFiles(dataSourceHandleObj,key,value);
        % %   dataSourceHandleObj.getAllDataFiles(key,value);        
        % 
        % %% Description
        % % Method to scan the source property for available data files 
        % % matching the a Liquid_DataFile format, which are currently all *.csv 
        % % and *.csv.gz files. This method is primary called by the
        % % constructor Liquid_DataSourceHandle. Matching data files are stored 
        % % in the 'fileList' property 
        % 
        % %% Input Arguments       
        % %
        % %% optional
        % % *key:* |'refresh'| 
        % % *value:* logical scalar defining whether the 'fileList' 
        % % property should be refreshed in case its already non empty
        % % which is the case when getAllDataFiles was called already
        % % Default: |false|
        % 
        % %% 
        % % *key:* |'displayMode'| 
        % % *value:* Numeric scalar vaule defining the amount of
        % % information which is displayed to the console.
        % % Default: |0|
        %
        % %% Example:
        % %%
        % %    % url source
        % %    dataSources1 = Liquid_DataSourceHandle('http://www.isa.uni-stuttgart.de/liquidData','url');
        % %    dataSources1.getAllDataFiles('display',2)
        % %% 
        % %    % local source
        % %    dataSources2 = Liquid_DataSourceHandle(pwd,'local');
        % %    dataSources1.getAllDataFiles('refresh',true)
        % 
        % %% See also
        % % Liquid_DataSourceHandle                           
            
            % check input parameter 
            p = inputParser();
            p.FunctionName = 'Liquid_DataSourceHandle.getAllDataFiles';
            addParamValue(p,'displayMode',0,@(x) isscalar(x) && isnumeric(x))
            addParamValue(p,'refresh',false,@(x) isscalar(x) && logical(x))
            parse(p,varargin{:})            
            displayMode = p.Results.displayMode;
            
            % depending on the source type scan the source for
            % regexpressions 
            if isempty(dataSourceHandleObj.fileList) || p.Results.refresh
            
            
                switch dataSourceHandleObj.type
                    case 'local'
                        folderContent = dir(dataSourceHandleObj.source);
                        fileNames = arrayfun(@(x)x.name,folderContent,'Uniform',false);
                        csvFileNames = regexp(fileNames,'[a-zA-Z_0-9\.-]*\.csv','match');                                        
                        csvGzFileNames = regexp(fileNames,'[a-zA-Z_0-9\.-]*\.csv.gz','match');                    
                    case 'url'
                        siteContent = urlread(dataSourceHandleObj.source);                        
                        csvFileNames = regexp(siteContent,'<a href="([\w\-\.]*.csv)"','tokens');
                        csvGzFileNames = regexp(siteContent,'<a href="([\w\-\.]*.csv.gz)"','tokens');
                    case 'empty'
                        csvFileNames = {};
                        csvGzFileNames = {};
                end   
                % combin csv and csv.gz files
                allFiles = [csvFileNames{:},csvGzFileNames{:}];
                if displayMode > 0
                    fprintf('\nFiles found in %s',dataSourceHandleObj.source)
                end
                % convert all detected files to Liquid_DataFile objects
                if(~isempty(allFiles))
                    for i = 1:length(allFiles)
                        [~,runName,ext1] = fileparts(allFiles{i});
                        ext2 = '';                    
                        if strcmp(ext1,'.gz')
                            [~,runName,ext2] = fileparts(runName);                        
                        end
                        ext = [ext2,ext1];
                        if strcmp(runName(end-5:end),'.train')
                            tmpDataFile = Liquid_DataFile(runName(1:end-6),ext,'type',runName(end-4:end),'displayMode',displayMode);
                        elseif strcmp(runName(end-4:end),'.test');
                            tmpDataFile = Liquid_DataFile(runName(1:end-5),ext,'type',runName(end-3:end),'displayMode',displayMode);
                        else
                            tmpDataFile = Liquid_DataFile(runName,ext,'displayMode',displayMode);
                        end
                        dataSourceHandleObj.addToFileList(tmpDataFile)
                    end  
                end
            end
            
        end
        
        function addToFileList(dataSourceHandleObj,dataFileObj)
        % %% Liquid_DataSourceHandle.addToFileList
        % % Method to add a object of class Liquid_DataFile to the fileList
        % % property
        % 
        % %% Syntax
        % %   addToFileList(dataSourceHandleObj,dataFileObj);
        % %   dataSourceHandleObj.addToFileList(dataFileObj);        
        % 
        % %% Description
        % % Method to add a object of class Liquid_DataFile to the fileList
        % % property
        % 
        % %% Input Arguments
        % %
        % %% required
        % % |dataFileObj|: object of class Liquid_DataFile
        %
        % %% Example:
        % %%
        % %    % 
        % %    dataSources1 = Liquid_DataSourceHandle('http://www.isa.uni-stuttgart.de/liquidData','url');
        % %    testFile1 = Liquid_DataFile('testFile','.csv','type','test');
        % %    size(dataSources1.fileList)
        % %    dataSources1.addToFileList(testFile1);      
        % %    size(dataSources1.fileList)
        % 
        % %% See also
        % % Liquid_DataFile, Liquid_Data     
                       
            
            % check input parameter
            p = inputParser();
            p.FunctionName = 'Liquid_DataSourceHandle.addToFileList';
            addRequired(p,'dataFileObj',@(x)isa(x,'Liquid_DataFile'))
            parse(p,dataFileObj)    
            if ~isempty(dataSourceHandleObj.fileList)
                dataSourceHandleObj.fileList(end+1) = p.Results.dataFileObj; 
            else
                dataSourceHandleObj.fileList = p.Results.dataFileObj;
            end
        end
        
        function dataMat = getDataByIndex(dataSourceHandleObj,index,varargin)
        % %% Liquid_DataSourceHandle.getDataByIndex
        % 
        % %% Syntax
        % %   dataMat = getDataByIndex(dataSourceHandleObj,index,key,value);
        % %   dataMat = dataSourceHandleObj.getDataByIndex(index,key,value);
        % 
        % %% Description
        % % Method to actually read out the data from a given dataFile.
        % % This differs depending on the source type and dataFile ext.       
        % % This method is primary called by the getLiquidData method
        % 
        % %% Input Arguments
        % %
        % %% required
        % % |index|: numerical scalar defining which element in the
        % % fileList property should be pcked.
        % %        
        % 
        % %% optional
        %
        % % *key:* |'displayMode'| 
        % % *value:* Numeric scalar vaule defining the amount of
        % % information which is displayed to the console.
        % % Default: |0|
        %
        % %% Output:
        % % |dataSourceHandleObj| a new object of class
        % % Liquid_DataSourceHandle. In case a invalid source is given, 
        % % an empty object is returned       
        %
        % %% Example:
        % %%
        % %    % url source
        % %    dataSources1 = Liquid_DataSourceHandle('http://www.isa.uni-stuttgart.de/liquidData','url');
        % %    % pick index 1
        % %    dataMat1 = dataSources1.getDataByIndex(1);
        % %    size(dataMat1)
        % %%
        % %    % set display
        % %    dataMat2 = dataSources1.getDataByIndex(2,'display',2);
        % %    size(dataMat2)
        % 
        % %% See also
        % % Liquid_DataFile, Liquid_Data               
            
        
            % check if the fileList is empty and return warning if so
            if isempty(dataSourceHandleObj.fileList)
                warning('empty fileList')
                dataMat = double.empty;
                return
            end
            % determine size of the fileList
            listSize = length(dataSourceHandleObj.fileList);
            % check input parameter
            p = inputParser();
            p.FunctionName = 'Liquid_DataSourceHandle.getDataByIndex';
            p.addRequired('index',@(x) isscalar(x) && isnumeric(x) && 0<x && x<=listSize)
            p.addParamValue('displayMode',0,@(x) isscalar(x) && isnumeric(x))
            parse(p,index,varargin{:})            
            displayMode = p.Results.displayMode;            
            % extract the selected dataFile object in the list
            tmpDataFile = dataSourceHandleObj.fileList(index);
            % read out the content of the file according to the 
            % dataSourceHandleObj.type and tmpDataFile.ext
            if displayMode > 0
                fprintf('\nRead "*.%s"-data from %s source',tmpDataFile.ext,dataSourceHandleObj.type)
            end
            switch dataSourceHandleObj.type                
                case 'local'
                    fileLoc = fullfile(dataSourceHandleObj.source,[tmpDataFile.fileName,tmpDataFile.ext]);
                    switch tmpDataFile.ext
                        case '.csv'
                            dataMat = csvread(fileLoc);
                        case '.csv.gz'
                            tmpFile = tempname();
                            gunzip(fileLoc,tmpFile)
                            dataMat = csvread(fullfile(tmpFile,[tmpDataFile.fileName,tmpDataFile.ext]));
                            rmdir(tmpFile,'s');
                    end
                case 'url'
                    fileLoc = [dataSourceHandleObj.source,'/',[tmpDataFile.fileName,tmpDataFile.ext]];
                    switch tmpDataFile.ext
                        case '.csv'
                            fileContent = urlread(fileLoc);
                            dataMat = str2num(fileContent);
                        case '.csv.gz'
                            tmpFile = tempname();
                            gunzip(fileLoc,tmpFile);
                            dataMat = csvread(fullfile(tmpFile,[tmpDataFile.fileName,'.csv']));
                            rmdir(tmpFile,'s');
                    end
            end
            if displayMode > 0
                fprintf(' - done')
            end
        end 
        
        function liquidDataObj = getLiquidData(dataSourceHandleObj,name,varargin)
        % %% Liquid_DataSourceHandle.getLiquidData
        % 
        % %% Syntax
        % %   liquidDataObj = getLiquidData(dataSourceHandleObj,name,key,value);
        % %   liquidDataObj = dataSourceHandleObj.getLiquidData(name,key,value);
        % 
        % %% Description
        % % Main method of the class, scanning the source for data matching 
        % % the given name and if a match is found the method looks for train
        % % as well as for test data.
        % 
        % %% Input Arguments
        % %
        % %% required
        % % |name|: character string describing the name of the dataset one
        % % looks for.
        % %        
        % 
        % %% optional
        %
        % % *key:* |'displayMode'| 
        % % *value:* Numeric scalar vaule defining the amount of
        % % information which is displayed to the console.
        % % Default: |0|
        %
        % %% Output:
        % % |liquidDataObj| an object of class Liquid_Data  
        %
        % %% Example:
        % %%
        % %    % url source
        % %    dataSources1 = Liquid_DataSourceHandle('http://www.isa.uni-stuttgart.de/liquidData','url');
        % %    % pick index 1
        % %    liquidDat1  = dataSources1.getLiquidData('crime');
        % %    % set display
        % %    liquidDat2 = dataSources1.getLiquidData('banana-bc','display',2);
        % 
        % %% See also
        % % Liquid_DataFile, Liquid_Data             
            
          
            p = inputParser();
            p.FunctionName = 'Liquid_DataSourceHandle.getLiquidData';
            addRequired(p,'name',@ischar)
            addParamValue(p,'displayMode',0,@(x) isscalar(x) && isnumeric(x))
            parse(p,name,varargin{:})
            
            if dataSourceHandleObj.isempty()
                warning('FileList of DataSourceHandle object is empty - empty Liquid_Data object returned') 
                liquidDataObj = Liquid_Data.empty;
                return
            end

            matchIndex = arrayfun(@(x)strcmp(x.name,name),dataSourceHandleObj.fileList);
            if sum(matchIndex) > 2
                warning('name is not unique in the datasource\nEmpty Liquid_Data returned')
            elseif sum(matchIndex) == 1
                tmpDataFile = dataSourceHandleObj.fileList(matchIndex);
                if ~strcmp(tmpDataFile.type,'train')
                    warning('Only one file found matching %s with type unequal to "train"',name)                                            
                end
                trainData = dataSourceHandleObj.getDataByIndex(find(matchIndex,1),'displayMode',p.Results.displayMode);
                liquidDataObj = Liquid_Data('dataSource',dataSourceHandleObj.source,...
                                     'trainData',trainData);
                liquidDataObj.name =  name;                 
            elseif sum(matchIndex) == 2
                index = find(matchIndex);
                dataFile1 = dataSourceHandleObj.fileList(index(1));
                dataFile2 = dataSourceHandleObj.fileList(index(2));
                trainIndex = find(strcmp({dataFile1.type,dataFile2.type},'train'));   
                if isempty(trainIndex) || length(trainIndex) > 1
                    error('no, (or no unique) training type data found with name %s',name)
                end
                if trainIndex == 1
                   trainData = dataSourceHandleObj.getDataByIndex(index(1)); 
                   testData  = dataSourceHandleObj.getDataByIndex(index(2)); 
                else
                   trainData = dataSourceHandleObj.getDataByIndex(index(2)); 
                   testData  = dataSourceHandleObj.getDataByIndex(index(1)); 
                end
                liquidDataObj = Liquid_Data('trainData',trainData,...
                                     'testData',testData);
                liquidDataObj.name =  name;
            else                     
               warning('No dataset matching %s found in the datasource %s\nEmpty Liquid_Data returned',name,dataSourceHandleObj.source)
               liquidDataObj = Liquid_Data.empty;
            end 

        end
        
        function ret = isempty(dataSourceHandleObj)
        % %% Liquid_DataSourceHandle.isempty
        % 
        % %% Syntax
        % %   logicalVal = isempty(dataSourceHandleObj);
        % %   logicalVal = dataSourceHandleObj.isempty();
        % 
        % %% Description
        % % Method to determine if a object is empty
        % 
        % %% Output:
        % % |ret| logical scalar 
        %
        % %% Example:
        % %%
        % %    % url source
        % %    dataSources1 = Liquid_DataSourceHandle('http://www.isa.uni-stuttgart.de/liquidData','url');
        % %    isempty(dataSources1)
        %
        % %% 
        % %    % local source
        % %    dataSources2 = Liquid_DataSourceHandle(pwd,'local');
        % %    isempty(dataSources2)
        % %% See also
        % % Liquid_DataFile, Liquid_Data
        
            if any(size(dataSourceHandleObj)==0)
                ret = true;
                return
            end
            if any(size(dataSourceHandleObj)>1)
                ret = arrayfun(@isempty,dataSourceHandleObj);
                return
            end
            ret = isempty(dataSourceHandleObj.fileList) || ...
                  strcmp(dataSourceHandleObj.type,'empty');
        end
        
    end
    
    methods(Static,Hidden)
        function check = checkValidType(type)
            if ~ischar(type)
                check = false;
                return
            end
            if ~any(strcmp(type,{'local','url','empty'}))
                check = false;
                return        
            end
            check = true;
        end
    end
end




