classdef Liquid_LeastSquareSVM < Liquid_Model
   % LeastSquareSVM Performs Least-Squares
   % 
   % An interesting property still is CLIPPING.
   % 
   % Methods
   % -------
   % train
   %     train the model on a hyperparameter grid
   %
   % select
   %     select the best hyperparameteers
   %
   % test
   %     evaluate predictions for new test data and calculate test error.
   %
   % See also
   % --------
   % `LeastSquareSVM <LeastSquareSVM.html>`_, mcSVM
   
   % Copyright 2015-2017 Philipp Thomann
   % see the COPYING file included with this software
   %    
   % Detail information see:   
   % 
   %                         !!!<a href="matlab:open(fullfile('html','LeastSquareSVM_Doc.html'))">LeastSquareSVM</a>!!!
   %
   % % 
   
    
   methods
        function model = Liquid_LeastSquareSVM(varargin)
        % LeastSquareSVM (Constructor)
        % Constructor of the class LeastSquareSVM
        %
        % Syntax::
        %
        %   model = LeastSquareSVM(x,y,key,value);
        %   model = LeastSquareSVM(liquidData,key,value);
        %
        % Parameters
        % ----------
        % x: double
        %    matrix providing the input (train) data for the model.
        % y: double (optional)
        %    vector providing the input (train) labls for the model. The
        %    length of y must fit the length of x, meaning ``size(y,1) == size(x,1)``.   
        % liquidData: An object of class Liquid_Data.
        %    Such an object provides at least the train feature data as well as
        %    the train labels. In case test data is provided, testing is done automaticaly.
        % 
        % 
        % For the optional parameters see:
        % See `Overview of Configuration Parameters <Liquid_Model.constructor_Doc.html>`_
        %
        % Returns
        % -------
        % ``model`` an object of class LeastSquareSVM          
        %
        % Examples
        % --------
        % 
        % example of how to use the class::
        %
        %    myData = [[rand(100,1)>.8;rand(100,1)>.2],[rand(100,3)+.5;rand(100,3)-.5]];
        %    myRandomLiquid_Data = Liquid_Data('data',myData,'split','random')
        %    model = Liquid_LeastSquareSVM(myRandomLiquid_Data);
        %    [result,error] = model.test(myRandomLiquid_Data);
        % 
        % example of how to use the class::
        %
        %    reg = Liquid_Data('name','reg-1d');
        %    model = Liquid_LeastSquareSVM(reg);
        %    [result,error] = model.test(reg);
        % 
        % See also Liquid_Model, Liquid_Data

        args = [varargin {'SCENARIO','LS'}];
        model@Liquid_Model(args{:});
        model.autoTrainSelect()
      end
   end
end
