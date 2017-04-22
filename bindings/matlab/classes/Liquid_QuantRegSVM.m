classdef Liquid_QuantRegSVM < Liquid_Model
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


   properties (Hidden)
      weights = [];
   end
   
   methods
      function model = Liquid_QuantRegSVM(x,y, weights, varargin)
         if ~isnumeric(weights)
             varargin = [ {weights}, varargin ];
             weights = [];
         end
         if isempty(weights);
             weights = [0.05,0.1,0.5,0.9,0.95];
         end
         args = [ {x,y,'SCENARIO','QT','WEIGHTS',weights}, varargin ];
         model@Liquid_Model(args{:});
         model.weights = weights;
         model.autoTrainSelect()
      end
      
      function errSelect = select(model, varargin)
          for i = 1:length(model.weights)
              model.set('WEIGHT_NUMBER',i);
              errSelect = select@Liquid_Model(model, varargin{:});
          end
      end
   end
end
