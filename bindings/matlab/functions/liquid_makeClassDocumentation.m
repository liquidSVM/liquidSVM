function liquid_makeClassDocumentation(filePath,varargin)

    % check input parameter for consistency
    p = inputParser();
    p.FunctionName = 'makeClassDocumentation';
    p.addRequired('filePath',@(x)exist(x,'file'))
    p.addParameter('docPath','',@(x)exist(x,'dir'))
    p.parse(filePath)
    % determine filename and path
    [~,fileName,~] = fileparts(filePath);
    % check if the folders for documentation exist
    if isempty(p.Results.docPath)
        [functionPath,~,~] = fileparts(mfilename('fullpath'));
        rootPath = fullfile(functionPath,'..');
        docPath = fullfile(rootPath,'doc');
    end
    if ~exist(fullfile(docPath,'mScripts'),'dir')
        mkdir(fullfile(docPath,'mScripts'));        
    end
    if ~exist(fullfile(docPath,'html'),'dir')
        mkdir(fullfile(docPath,'html'));
    end    
    docScriptPath = fullfile(docPath,'mScripts');
    docHtmlPath = fullfile(docPath,'html');
    
    % create a doc.m file for the class
    classDocFile = fullfile(docScriptPath,[fileName,'_Doc.m']);
    docFileCon   = fopen(classDocFile,'w+');
    
    % open connection to the file in which the class isdefined in order to 
    % read out the comments for documentation
    classFileCon = fopen(filePath);
    
    % first read out the first line and check wether it defines a new class
    classDefLine = fgetl(classFileCon);
    if isempty(regexp(classDefLine,'classdef', 'once'))
        error('first line must provide keyword "classdef"')
    end
    % get Info from classdefinition
    className = regexp(classDefLine,'classdef (\w+)','tokens','once');
    if isempty(regexp(classDefLine,'classdef', 'once'))
        error('first line must provide keyword "classdef"')
    end
    if ~isempty(regexp(classDefLine,'<', 'once'))
        superClassName = regexp(classDefLine,'\w+ < (\w+)','tokens','once');
    else
        superClassName = {''};
    end    
    % print this info to the documentatin
    fprintf(docFileCon,'\n%%%% Name');
    fprintf(docFileCon,'\n%% *|%s|*',className{1});  
    fprintf(docFileCon,' from Superclass: *|%s|*',superClassName{1});
    fprintf(docFileCon,'\n%%%% Info');
   
    
    % set parameter to track region of the file
    lineCount = 1;
    section = 'classHeader';
    lineType = 'undefined';
    
    while(~feof(classFileCon))
        try
        nextLine = fgetl(classFileCon);
        lineCount = lineCount +1;
        nextLine = strtrim(nextLine);

        if isempty(nextLine)
            continue
        end
        
        if regexp(nextLine,'^properties','once');
            section = 'properties';  
            fprintf(docFileCon,'\n%%%% Properties');
            lineType = 'definition';
            propDoc = '';
        end         
        
        if regexp(nextLine,'^methods','once');
            section = 'methods'; 
            fprintf(docFileCon,'\n%%%% Methodes');
            lineType = 'definition';
        end

        if regexp(nextLine,'^function','once');
            lineType = 'definition';            
            methodName = regexp(nextLine,'(\w+)\(','tokens','once');
            if strcmp(methodName{1},fileName)
                methodDocFile = fullfile(docScriptPath,[fileName '_constructor_Doc.m']);
            else
                methodDocFile = fullfile(docScriptPath,[fileName '_' methodName{1} '_Doc.m']);
            end
            methodDocFileId = fopen(methodDocFile,'w+');
            htmlLink = strrep(methodDocFile,'.m','.html');
            htmlLink = strrep(htmlLink,'mScripts','html');
            fprintf(docFileCon,'\n%% <%s %s>\n%%',htmlLink,methodName{1});
        end
        
 

        if (strcmp(nextLine(1),'%') )%&& startReadComments)
            if length(nextLine) >=3 && strcmp(nextLine(1:3),'% %')
                lineType = 'docComment';
                if length(nextLine) >=7 && strcmp(nextLine(1:7),'% %    ');
                    lineType = 'docCommentExample';            
                end
                switch section
                    case 'classHeader'
                        if regexp(nextLine,'autor:')
                            author = regexp(nextLine,'autor: {1,20}(\w*) (\w*)','tokens','once');
                            fprintf(docFileCon,'\n%% \n%% *Author:* %s %s',author{1},author{2});
                        end
                        if regexp(nextLine,'version:')
                            version = regexp(nextLine,'version: {1,20}(\d).(\d)','tokens','once');
                            fprintf(docFileCon,'\n%% \n%% *Version:* %s.%s',version{1},version{2});
                        end                    
                    case 'properties'
                        fprintf(docFileCon,'\n%s',nextLine(2:end));
                    case 'methods'
                        switch lineType
                            case 'docCommentExample'
                                fprintf(methodDocFileId,'\n%s',nextLine(7:end));
                            otherwise
                                fprintf(methodDocFileId,'\n%s',nextLine(2:end));
                        end

                end
            else
                lineType = 'codeComment';
                if strcmp(section,'properties') && strcmp(nextLine(1),'%')
                    propDoc = [propDoc,' ',nextLine(2:end)];
                end
%                if strcmp(section,'classHeader')
%                       fprintf(methodDocFileId,'\n%s',nextLine(1:end));
%                     if regexp(nextLine,'autor:')
%                         author = regexp(nextLine,'autor: {1,20}(\w*) (\w*)','tokens','once');
%                         fprintf(docFileCon,'\n%%%% Author');
%                         fprintf(docFileCon,'\n%% %s',author{1});
%                     end
%                     
%                     if regexp(nextLine,'autor email:')
%                         author = regexp(nextLine,'autor: {1,20}(\w*) (\w*)','tokens','once');
%                         fprintf(docFileCon,'\n%%%% Author');
%                         fprintf(docFileCon,'\n%% %s',author{1});
%                     end
%                 end
            end
        else
            lineType = 'code';
            if strcmp(section,'properties') && ~strcmp(nextLine(1:end),'end') && ~strcmp(nextLine(1:end),'properties')
                fprintf(docFileCon,'\n%% *|%s|:*',nextLine(1:end));
                fprintf(docFileCon,'\n%% %s\n%%',propDoc);
                propDoc = '';
            end                       
        end
        catch ME
            disp('???')
        end
        
    end
    fclose(classFileCon);
    fclose(docFileCon);
    fclose(methodDocFileId);
    
    docScriptPathContent = dir(docScriptPath);
    addpath(docScriptPath)
    for i = 1:length(docScriptPathContent)
        if isempty(regexp(docScriptPathContent(i).name,['^',fileName],'once'))
            continue
        end
        publish(fullfile(docScriptPath,docScriptPathContent(i).name),'outputDir',docHtmlPath)
    end
    
    rmpath(docScriptPath)
    
end
