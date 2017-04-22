function model = mcSVM(varargin)
    model = Liquid_MultiClassSVM(varargin{:});
    model.autoTrainSelect()
end

% Copyright 2015-2017 Philipp Thomann
% 
% This file is part of liquidSVM.
% 
% liquidSVM is free software: you can redistribute it and/or modify
% it under the terms of the GNU Affero General Public License as
% published by the Free Software Foundation, either version 3 of the
% License, or (at your option) any later version.
% 
% liquidSVM is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
% GNU Affero General Public License for more details.
% 
% You should have received a copy of the GNU Affero General Public License
% along with liquidSVM. If not, see <http://www.gnu.org/licenses/>.

% classdef mcSVM < liquidSvmModel
%    methods
%       function model = mcSVM(x,y, varargin)
%          args = [ {x,y,'SCENARIO','MC'}, varargin ];
%          model@liquidSvmModel(args{:});
%          model.autoTrainSelect()
%       end
%    end
% end
