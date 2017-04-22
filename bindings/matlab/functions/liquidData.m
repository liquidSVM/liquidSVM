function data = liquidData(varargin)
    
    if nargin == 1
        switch class(varargin{1})
            case 'char'
                data = Liquid_Data('name',varargin{1});
            case 'double'                
                data = Liquid_Data('data',varargin{1});
            otherwise
                error('first argumentmust provide data as name or as double matrix')
        end
    elseif nargin > 1
        switch class(varargin{1})
            case 'char'
                data = Liquid_Data('name',varargin{1},varargin(2:end));
            case 'double'                
                data = Liquid_Data('data',varargin{1},varargin(2:end));
            otherwise
                error('first argumentmust provide data as name or as double matrix')
        end
    else
        error('no data provided')
    end
end

