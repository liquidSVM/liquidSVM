classdef Liquid_LeastSquareSVM < Liquid_Model
% %%
% % -----------------------------------------------------------------------
% % ----------------- classDef LeastSquareSVM --------------------------------------
% % -----------------------------------------------------------------------
% % HEADER:
% %
% % 	autor:       Philipp Thomann & Nico Schmid          
% % 	autor email: n_schmid@gmx.de      
% % 	create date:       
% % 	version:     0.9
% % 	update:				
% % 	remark:            
% % 
% % -----------------------------------------------------------------------
% % DESCRIPTION:
% % Copyright 2015-2017 Philipp Thomann
% % 
% % This file is part of Liquid.
% % 
% % Liquid is free software: you can redistribute it and/or modify
% % it under the terms of the GNU Affero General Public License as
% % published by the Free Software Foundation, either version 3 of the
% % License, or (at your option) any later version.
% % 
% % Liquid is distributed in the hope that it will be useful,
% % but WITHOUT ANY WARRANTY; without even the implied warranty of
% % MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
% % GNU Affero General Public License for more details.
% % 
% % You should have received a copy of the GNU Affero General Public License
% % along with Liquid. If not, see <http://www.gnu.org/licenses/>.
% % LeastSquareSVM Performs Least-Squares
% % 
% % An interesting property then still is CLIPPING.
% % 
% % Liquid_Model Methods:
% % train  - train the model on a hyperparameter grid
% % select - select the best hyperparameteers
% % test   - evaluate predictions for new test data and calculate test error.
% % See also: LeastSquareSVM, mcSVM
% % 
% % Copyright 2015-2017 Philipp Thomann
% % see the COPYING file included with this software
% %
% % -----------------------------------------------------------------------
% % -----------------------------------------------------------------------
% %
% %    
% % Detail information see:   
% % 
% %                         !!!<a href="matlab:open(fullfile('html','LeastSquareSVM_Doc.html'))">LeastSquareSVM</a>!!!
% %
% % 
% % See also: 

    
   methods
        function model = Liquid_LeastSquareSVM(varargin)
        % %% LeastSquareSVM (Constructor)
        % % Constructor of the class LeastSquareSVM
        % 
        % %% Syntax
        % %   model = LeastSquareSVM(x,y,key,value);
        % %   model = LeastSquareSVM(liquidData,key,value);
        % 
        % %% Description
        % % 
        % 
        % %% Input Arguments
        % %
        % %% required
        % %%
        % % |liquidData|: An object of class Liquid_Data. Such an object provides at least
        % % the train feature data as well as the train labels. In case test data is
        % % provided, testing is done automaticaly.
        % % 
        % %%
        % % |x|: double matrix providing the input (train) data for the model.
        % % 
        % %%
        % % |y|: double vector providing the input (train) labls for the model. The
        % % length of y must fit the length of x, meaning |size(y,1) == size(x,1)|.   
        % 
        % %% optional
        % % 
        % % For the optional parameters see:
        % % <Liquid_Model_constructor_Doc.html Overview of Configuration Parameters>
        %
        % %% Output:
        % % |model| an object of class LeastSquareSVM          
        %
        % %% Example:
        % % %% 
        % % % example of how to use the class
        % %    myData = [[rand(100,1)>.8;rand(100,1)>.2],[rand(100,3)+.5;rand(100,3)-.5]];
        % %    myRandomLiquid_Data = Liquid_Data('data',myData,'split','random')
        % %    model = Liquid_LeastSquareSVM(myRandomLiquid_Data);
        % %    [result,error] = model.test(myRandomLiquid_Data);
        % % %% 
        % % % example of how to use the class
        % %    reg = Liquid_Data('name','reg-1d');
        % %    model = Liquid_LeastSquareSVM(reg);
        % %    [result,error] = model.test(reg);
        % % 
        % %% See also
        % % % <Liquid_SVM.html Liquid_SVM>, <Liquid_Data.html Liquid_Data> 

        args = [varargin {'SCENARIO','LS'}];
        model@Liquid_Model(args{:});
        model.autoTrainSelect()
      end
   end
end
