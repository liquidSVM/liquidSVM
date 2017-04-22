

LIQUIDSVM FOR MATLAB


Welcome to the MATLAB bindings for liquidSVM.

  This is a preview version of the new MATLAB bindings to liquidSVM,
  stay tuned for updates. On Windows there is a heavy Bug at the moment
  that renders it unusable.

Both liquidSVM and these bindings are provided under the AGPL 3.0
license.


Installation

-   Download the Toolbox from
    http://www.isa.uni-stuttgart.de/software/matlab/liquidSVM.mltbx and
    install it in MATLAB by double clicking it.
-   You can compile the native library in MATLAB (for MacOS and Windows
    we currently ship binaries in the toolbox)

        mex -setup c++
        makeliquidSVM native

    For this you need to have a compiler installed, and you might to
    issue mex -setup c++ before.


Usage

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

The meaning of the configurations in the constructor is described in the
next chapter.

  NOTE: MATLAB does not respect flushing of print methods, hence setting
  display to 1 does not help in monitoring progress during execution
  because the output only shows at the end of the computation.

  NOTE: On macOS if you use MATLAB 2016a and Xcode 8 you have to make
  the new version available to MATLAB by changing
  /Applications/MATLAB_R2015b.app/bin/maci64/mexopts/clang_maci64.xml to
  also include MacOSX10.12.sdk on two occasions - similar details (for
  other versions) can be found int
  https://de.mathworks.com/matlabcentral/answers/243868-mex-can-t-find-compiler-after-xcode-7-update-r2015b.
  Remark that this change needs admin privileges.

Octave

Since Octave 4.0.x the classdef type of object-orientation is
(experimentally) implemented so liquidSVM can be used there as well.
Unzip the file
http://www.isa.uni-stuttgart.de/software/matlab/liquidSVM-octave.zip
change into a directory, start octave and issue:

    makeliquidSVM native

If this works you can use demo_svm etc. as above.
