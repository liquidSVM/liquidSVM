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

classdef Liquid_RecOpCharSVM < Liquid_Model
   properties (Hidden)
      weight_steps = [];
   end
   
   methods
      function model = Liquid_RecOpCharSVM(x,y, weight_steps, varargin)
         if ~isnumeric(weight_steps)
             varargin = [ {weight_steps}, varargin ];
             weight_steps = [];
         end
         if isempty(weight_steps);
             weight_steps = 9;
         end
         args = [ {x,y,'SCENARIO','ROC','WEIGHT_STEPS',weight_steps}, varargin ];
         model@Liquid_Model(args{:});
         model.weight_steps = weight_steps;
         model.autoTrainSelect()
      end
      
      function errSelect = select(model, varargin)
          for i = 1:model.weight_steps
              model.set('WEIGHT_NUMBER',i);
              errSelect = select@Liquid_Model(model, varargin{:});
          end
      end
   end
end
