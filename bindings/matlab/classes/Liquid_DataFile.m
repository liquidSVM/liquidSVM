classdef Liquid_DataFile
% %%
% % -----------------------------------------------------------------------
% % ----------------- classDef Liquid_DataFile -------------------------
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
% % Basic class to handle data files provided in the Liquid toolbox
% % 
% % Liquid_DataFile Properties:
% % name - character string descibing the name of the data file
% % ext - character string giving the format of the data file (allowed
% % values are '.csv' and '.csv.gz')
% % type - character string describing the type of the data file (supported
% % values are 'train' and 'test')
% % 
% % Liquid_DataFile Methods:
% % fileName - method to return the name of the dataset. If the type
% % property is nonempty, the return value is
% % [dataFileObj.name,'.',dataFileObj.type] and [dataFileObj.name]
% % otherwise.
% % -----------------------------------------------------------------------
% % -----------------------------------------------------------------------
% %
% %    
% % Detail information see:   
% % 
% %                         !!!<a href="matlab:open(fullfile('doc','html','Liquid_DataFile_Doc.html'))">Liquid_DataFile</a>!!!
% %
% %     
% %
% % See also: Liquid_DataSourceHandle
% %
% % Copyright 2015-2016 Nico Schmid
% % see the COPYING file included with this software
% %
    properties
        % character string describing the name of the data file
        name
        % Extention of the data file (allowed values are '.csv'
        % and '.csv.gz')
        ext
        % A character string describing the type of the data file
        % (supported values are 'train' and 'test')
        type
    end

    methods
        
        function dataFileObj = Liquid_DataFile(name,ext,varargin)
        % %% Liquid_DataFile (Constructor)
        % % Constructor of the class DataFile
        % 
        % %% Syntax
        % %   dataFileObj = Liquid_DataFile(name,ext,key,value);
        % 
        % %% Description
        % % long description
        % 
        % %% Input Arguments
        % %
        % %% required
        % % |name|: character string describing the name of the data file
        % % 
        % %%
        % % |ext|: Extention of the data file (allowed values are '.csv'
        % % and '.csv.gz')        
        % 
        % %% optional
        % % *key:* |'type'| 
        % % *value:* A character string describing the type of the data 
        % % file (supported values are 'train' and 'test')
        % % Default: |''|
        % 
        % %% 
        % % *key:* |'displayMode'| 
        % % *value:* Numeric scalar vaule defining the amount of
        % % information which is displayed to the console.
        % % Default: |0|
        %
        % %% Output:
        % % |dataFileObj| a new object of class Liquid_DataFile.      
        % 
        % %% Example:
        % %%
        % %    % default call
        % %    Liquid_DataFile('testFile','.csv')
        % %% 
        % %    % specify type
        % %    Liquid_DataFile('testFile','.csv','type','test')       
        % 
        % %% See also
        % %              

            % check and confirm input Parameter
            p = inputParser();
            p.FunctionName = 'Liquid_DataFile';
            p.addRequired('name',@ischar);
            p.addRequired('ext',@(x) strcmp(x,'.csv') || strcmp(x,'.csv.gz'))
            p.addParamValue('type','',@ischar)
            p.addParamValue('displayMode',0,@(x) isscalar(x) && isnumeric(x))
            p.parse(name,ext,varargin{:})
            % store confirmed input parameter to class properties
            dataFileObj.name = p.Results.name;
            dataFileObj.ext  = p.Results.ext;
            dataFileObj.type = p.Results.type;
            % optional display
            if p.Results.displayMode > 0
                fprintf('\n%s file %s ',dataFileObj.fileName())
            end
        end

        function fileName = fileName(dataFileObj)
        % %% Liquid_DataFile.fileName 
        % % short description 
        % 
        % %% Syntax
        % %   obj = BHCS_ClassDefTemplate(key,value);
        % 
        % %% Description
        % % method to return the name of the dataset. If the type
        % % property is nonempty, the return value is
        % % [dataFileObj.name,'.',dataFileObj.type] and [dataFileObj.name]
        % % otherwise.
        %
        % %% Output:
        % % |fileName| character string.      
        % 
        % %% Example:
        % %% 
        % %    file1 = Liquid_DataFile('testFile1','.csv');
        % %    file1.fileName
        % %%
        % %    file2 = Liquid_DataFile('testFile2','.csv','type','test')         
        % %    file2.fileName        
        % 
        % %% See also
        % % Liquid_DataFile             

            % return filename dependent on the value of type
            if ~isempty(dataFileObj.type)
                fileName = [dataFileObj.name,'.',dataFileObj.type];
            else
                fileName = [dataFileObj.name];
            end
        end

    end

end

