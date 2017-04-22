% load some data sets with train/test split from http://www.isa.uni-stuttgart.de/liquidData/
banana = liquidData('banana-bc');  % binary labels
banana_mc = liquidData('banana-mc');  % labels with four unique values
reg = liquidData('reg-1d');  % real labels

%% Least Squares Regression
model = svm_ls(reg.train,'DISPLAY','1');
[result, err] = model.test(reg.test);
result = model.predict(reg.testFeatures);

%% Mutli-Class classification
model = svm_mc(banana_mc.train,'DISPLAY','1','folds','3');
[result, err] = model.test(banana_mc.test);

%% Quantile Regression here for the 20%, 50%, and 80% quantiles
model = svm_qt(reg.trainFeatures, reg.trainLabel,[0.2,0.5,0.8],'DISPLAY','1');
[quantiles, err] = model.test(reg.testFeatures,reg.testLabel);
plot(reg.testFeatures, reg.testLabel, '.', reg.testFeatures, quantiles(:,1),'.',...
    reg.testFeatures, quantiles(:,2),'.',reg.testFeatures, quantiles(:,3),'.')

% now quantiles has three columns corresponding to the three requested quantiles

%% Expectile Regression here for the 20% and 50% expectiles
model = svm_ex(reg.trainFeatures, reg.trainLabel,[.05,.5],'DISPLAY','1');
[expectiles, err] = model.test(reg.testFeatures,reg.testLabel);
plot(reg.testFeatures, reg.testLabel, '.', reg.testFeatures, expectiles(:,1),'.',...
    reg.testFeatures, expectiles(:,2),'.')

%% Receiver Operating Characteristic curve
model = svm_roc(banana.trainFeatures, banana.trainLabel,6,'DISPLAY','1');
[result, err] = model.test(banana.test);

%% Neyman-Pearson lemma
model = svm_npl(banana.trainFeatures, banana.trainLabel, 1,'DISPLAY','1');
[result, err] = model.test(banana.test);

%% Write a solution (after train and select have been performed)
model = svm_ls(reg.train,'DISPLAY','1');
save myModelFile model
clear model

%% read a solution from file
load myModelFile model
[result, err] = model.test(reg.test);
