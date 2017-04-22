
%% Name
% *|LiquidSVM_Model|* from Superclass: *|handle|*
%% Info
% 
% *Author:* Philipp Thomann
% 
% *Version:* 0.9
%% Properties
% *|trainErrors = [];|:*
%   reflects the train error
%
% *|selectErrors = [];|:*
%   reflects the select error
%
% *|trainFeatures|:*
%   training features
%
% *|trainLabels|:*
%   training labels
%
%% Properties
% *|properties (Hidden)|:*
% 
%
% *|cookie = -1;|:*
%   the cookie is used to link the matlab object to the corresponding  svm_manager C++ object
%
% *|labels = {};|:*
%   cell to remember the labels
%
% *|trained = 0;|:*
%   flag indicating whether the model is already trained
%
% *|selected = 0;|:*
%   flag indicating whether the model is already selected
%
% *|scenario = '';|:*
%   szenario
%
%% Properties
% *|properties (Constant)|:*
% 
%
% *|errors_colnames = {'task','cell','fold','gamma','pos_weight',...|:*
% 
%
% *|'lambda','train_error','val_error',...|:*
% 
%
% *|'init_iterations','train_iterations',...|:*
% 
%
% *|'val_iterations','init_iterations',...|:*
% 
%
% *|'gradient_updates','SVs'};|:*
% 
%
%% Methodes
% <C:\Users\scn3wa2\Documents\SVN\branches\liquidSVM\code\functions\..\doc\html\LiquidSVM_Model_constructor_Doc.html LiquidSVM_Model>
%
% <C:\Users\scn3wa2\Documents\SVN\branches\liquidSVM\code\functions\..\doc\html\LiquidSVM_Model_train_Doc.html train>
%
% <C:\Users\scn3wa2\Documents\SVN\branches\liquidSVM\code\functions\..\doc\html\LiquidSVM_Model_select_Doc.html select>
%
% <C:\Users\scn3wa2\Documents\SVN\branches\liquidSVM\code\functions\..\doc\html\LiquidSVM_Model_test_Doc.html test>
%
% <C:\Users\scn3wa2\Documents\SVN\branches\liquidSVM\code\functions\..\doc\html\LiquidSVM_Model_predict_Doc.html predict>
%
% <C:\Users\scn3wa2\Documents\SVN\branches\liquidSVM\code\functions\..\doc\html\LiquidSVM_Model_autoTrainSelect_Doc.html autoTrainSelect>
%
% <C:\Users\scn3wa2\Documents\SVN\branches\liquidSVM\code\functions\..\doc\html\LiquidSVM_Model_getNum_Doc.html getNum>
%
% <C:\Users\scn3wa2\Documents\SVN\branches\liquidSVM\code\functions\..\doc\html\LiquidSVM_Model_get_Doc.html get>
%
% <C:\Users\scn3wa2\Documents\SVN\branches\liquidSVM\code\functions\..\doc\html\LiquidSVM_Model_set_Doc.html set>
%
% <C:\Users\scn3wa2\Documents\SVN\branches\liquidSVM\code\functions\..\doc\html\LiquidSVM_Model_config_line_Doc.html config_line>
%
% <C:\Users\scn3wa2\Documents\SVN\branches\liquidSVM\code\functions\..\doc\html\LiquidSVM_Model_cover_Doc.html cover>
%
% <C:\Users\scn3wa2\Documents\SVN\branches\liquidSVM\code\functions\..\doc\html\LiquidSVM_Model_solution_Doc.html solution>
%
% <C:\Users\scn3wa2\Documents\SVN\branches\liquidSVM\code\functions\..\doc\html\LiquidSVM_Model_read_solution_Doc.html read_solution>
%
% <C:\Users\scn3wa2\Documents\SVN\branches\liquidSVM\code\functions\..\doc\html\LiquidSVM_Model_write_solution_Doc.html write_solution>
%
% <C:\Users\scn3wa2\Documents\SVN\branches\liquidSVM\code\functions\..\doc\html\LiquidSVM_Model_delete_Doc.html delete>
%
% <C:\Users\scn3wa2\Documents\SVN\branches\liquidSVM\code\functions\..\doc\html\LiquidSVM_Model_saveobj_Doc.html saveobj>
%
%% Methodes
% <C:\Users\scn3wa2\Documents\SVN\branches\liquidSVM\code\functions\..\doc\html\LiquidSVM_Model_loadobj_Doc.html loadobj>
%
%% Methodes
% <C:\Users\scn3wa2\Documents\SVN\branches\liquidSVM\code\functions\..\doc\html\LiquidSVM_Model_parseVarargin_Doc.html parseVarargin>
%
% <C:\Users\scn3wa2\Documents\SVN\branches\liquidSVM\code\functions\..\doc\html\LiquidSVM_Model_checkTrainFeatures_Doc.html checkTrainFeatures>
%
% <C:\Users\scn3wa2\Documents\SVN\branches\liquidSVM\code\functions\..\doc\html\LiquidSVM_Model_checkTrainLabels_Doc.html checkTrainLabels>
%
