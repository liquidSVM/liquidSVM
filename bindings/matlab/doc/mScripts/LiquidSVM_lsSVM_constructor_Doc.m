
 %% lsSVM (Constructor)
 % Constructor of the class lsSVM
 %% Syntax
 %   model = lsSVM(key,value);
 %   model = lsSVM(liquidData,key,value);
 %% Description
 %
 %% Input Arguments
 %
 %% required
 %%
 % |liquidData|: An object of class LiquidSVM_Data. Such an object provides at least
 % the train feature data as well as the train labels. In case test data is
 % provided, testing is done automaticaly.
 %
 %%
 % |x|: double matrix providing the input (train) data for the model.
 %
 %%
 % |y|: double vector providing the input (train) labls for the model. The
 % length of y must fit the length of x, meaning |size(y,1) == size(x,1)|.
 %% optional
 %
 % For the optional parameters see:
 % <LiquidSVM_Model_constructor_Doc.html Overview of Configuration Parameters>
 %% Output:
 % |model| an object of class lsSVM
 %% Example:
 % %%
 % % example of how to use the class
 myData = [[rand(100,1)>.8;rand(100,1)>.2],[rand(100,3)+.5;rand(100,3)-.5]];
 myRandomLiquidSVM_Data = LiquidSVM_Data('data',myData,'split','random')
 model = LiquidSVM_lsSVM(myRandomLiquidSVM_Data);
 [result,error] = model.test(myRandomLiquidSVM_Data);
 % %%
 % % example of how to use the class
 reg = LiquidSVM_Data('name','reg-1d');
 model = LiquidSVM_lsSVM(reg);
 [result,error] = model.test(reg);
 %
 %% See also
 % % <SimonsSvmModel.html SimonsSvmModel>, <SmlDataClassDoc.html SmlData>
