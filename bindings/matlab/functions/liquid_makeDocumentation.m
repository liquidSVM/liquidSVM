function liquid_makeDocumentation()

    [functionPath,~,~] = fileparts(mfilename('fullpath'));
    rootPath = fullfile(functionPath,'..');
    classDir = fullfile(rootPath,'classes');
    docDir = fullfile(rootPath,'doc');
    docHtmlDir = fullfile(docDir,'html');
    helptocFile = fullfile(docDir,'helptoc.xml');
    helptocFileCon = fopen(helptocFile,'w+');
    %add header to helptoc file
    fprintf(helptocFileCon,'\n<?xml version=''1.0'' encoding="utf-8"?>');
    fprintf(helptocFileCon,'\n<toc version="2.0">');
    
    classDirContent = dir(classDir);
    if length(classDirContent) > 2
        fprintf(helptocFileCon,'\n<tocitem">Classes:');
        for i = 3:length(classDirContent)            
            [~,runClassName,ext] = fileparts(classDirContent(i).name);
            if ~strcmp(ext,'.m')
                continue
            end
            liquid_makeClassDocumentation(fullfile(classDir,classDirContent(i).name))
            htmlContent = dir(docHtmlDir);
            for j = 3:length(htmlContent)
                [~,htmlFileName,~] = fileparts(htmlContent(j).name);
                parts = strsplit(htmlFileName,'_');
                if length(parts) == 3
                    % class main
                    if strcmp([parts{1} '_' parts{2}],runClassName)
                        fprintf(helptocFileCon,'\n<tocitem target="html\\%s">%s',htmlContent(j).name,runClassName);
                    end
                end
            end
            methodTokenSet = false;
            for j = 3:length(htmlContent)
                [~,htmlFileName,~] = fileparts(htmlContent(j).name);
                parts = strsplit(htmlFileName,'_');
                if length(parts) == 4
                    % class method
                    if ~methodTokenSet
                        fprintf(helptocFileCon,'\n<tocitem">Methodes:');
                        methodTokenSet = true;
                    end
                    if strcmp([parts{1} '_' parts{2}],runClassName) 
                        fprintf(helptocFileCon,'\n<tocitem target="html\\%s">%s</tocitem>',htmlContent(j).name,parts{3});
                    end
                end
            end
            if methodTokenSet
                fprintf(helptocFileCon,'\n</tocitem>');% end methodes
            end
            fprintf(helptocFileCon,'\n</tocitem>');% end class
        end
        fprintf(helptocFileCon,'\n</tocitem>');% close classList token
    end
    fprintf(helptocFileCon,'\n</toc>');
    fclose(helptocFileCon);
end
