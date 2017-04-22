classdef Liquid_NeymanPearsonSVM < Liquid_Model  
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
      constraint = 0.05;
      nplClass = 1;
      constraintFactors = [ 0.5,2/3,1,1.5,2 ];
   end
   
   methods
      function model = Liquid_NeymanPearsonSVM(x,y, nplClass, constraint, varargin)
         if ~isnumeric(nplClass)
             disp(nplClass);
             error('nplClass has to be an integer but got instead\n');
         end
         if ~isnumeric(constraint)
             varargin = [ {constraint}, varargin ];
             constraint = 0.05;
         end
         args = [ {x,y,'SCENARIO',['NPL ',int2str(nplClass)]}, varargin ];
         model@Liquid_Model(args{:});
         model.nplClass = nplClass;
         model.constraint = constraint;
         model.autoTrainSelect()
      end
      
      function errSelect = select(model, varargin)
          for c = model.constraintFactors * model.constraint
              model.set('NPL_CLASS',model.nplClass);
              model.set('NPL_CONSTRAINT',c);
              errSelect = select@Liquid_Model(model, varargin{:});
          end
      end
   end
end
