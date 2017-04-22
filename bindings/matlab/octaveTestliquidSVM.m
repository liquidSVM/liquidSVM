%! %% Main function to generate tests
%!     tests = functiontests(localfunctions);
%! 
%! 
%! %% Optional file fixtures  
%!shared d %  % do not change function name
%!     load data
%!     I = randperm(4000,400);
%!     d.TestData.train = {banana_mc_train_x(I,:), banana_mc_train_y(I)};
%!     d.TestData.test = {banana_mc_test_x(I,:), banana_mc_test_y(I)};
%! 
%! 
%! %% Test Functions
%!test %mcSVM_AvA_hinge(d)
%! model = mcSVM(d.TestData.train{:},'threads',1);
%! hand_err = 1-mean(model.predict(d.TestData.test{1})==d.TestData.test{2});
%! [res, test_err] = model.test(d.TestData.test{:});
%! assert(all(size(test_err) == [1+6,3]))
%! test_err = test_err(1);
%! assert(hand_err < 0.3)
%! assert(test_err < 0.3)
%! assert(abs(test_err - hand_err) < 1e-8)
%! 
%! 
%!xtest %mcSVM_OvA_ls(d)
%! model = mcSVM(d.TestData.train{:},'threads',1,'SCENARIO','MC OvA_ls');
%! hand_err = 1-mean(model.predict(d.TestData.test{1})==d.TestData.test{2});
%! [res, test_err] = model.test(d.TestData.test{:});
%! assert(all(size(test_err) == [1+4,3]))
%! test_err = test_err(1);
%! assert(hand_err < 0.3)
%! assert(test_err < 0.3)
%! assert(abs(test_err - hand_err) < 1e-8)
%! 
%!xtest %mcSVM_OvA_hinge(d)
%! model = mcSVM(d.TestData.train{:},'threads',1,'SCENARIO','MC OvA_hi');
%! hand_err = 1-mean(model.predict(d.TestData.test{1})==d.TestData.test{2});
%! [res, test_err] = model.test(d.TestData.test{:});
%! assert(all(size(test_err) == [1+4,3]))
%! test_err = test_err(1);
%! assert(hand_err < 0.3)
%! assert(test_err < 0.3)
%! assert(abs(test_err - hand_err) < 1e-8)
%! 
%!test %mcSVM_AvA_ls(d)
%! model = mcSVM(d.TestData.train{:},'threads',1,'SCENARIO','MC AvA_hi');
%! hand_err = 1-mean(model.predict(d.TestData.test{1})==d.TestData.test{2});
%! [res, test_err] = model.test(d.TestData.test{:});
%! assert(all(size(test_err) == [1+6,3]))
%! test_err = test_err(1);
%! assert(hand_err < 0.3)
%! assert(test_err < 0.3)
%! assert(abs(test_err - hand_err) < 1e-8)
%! 
%!test %lsSVM(d)
%! model = svm_ls(d.TestData.train{:},'threads',1);
%! hand_err = mean((model.predict(d.TestData.test{1})-d.TestData.test{2}).^2);
%! [res, test_err] = model.test(d.TestData.test{:});
%! assert(all(size(test_err) == [1,3]))
%! test_err = test_err(1);
%! assert(hand_err < 0.75)
%! assert(test_err < 0.75)
%! assert(abs(test_err - hand_err) < 1e-8)
%! 
%!test %qtSVM(d)
%! quantiles_list = [0.05, 0.1, 0.5, 0.9, 0.95];
%! model = svm_qt(d.TestData.train{:},'threads',1,'weights',quantiles_list);
%! [res, test_err] = model.test(d.TestData.test{:});
%! %hand_err = mean((res(:,3)-d.TestData.test{2}).^2)
%! assert(all(size(test_err) == [length(quantiles_list),3]))
%! test_err = test_err(1);
%! %assert(hand_err < 1.5)
%! assert(max(test_err) < 0.1)
%! %assert(abs(test_err - hand_err) < 1e-8)
%! 
%!test %exSVM(d)
%! expectiles_list = [0.05, 0.1, 0.5, 0.9, 0.95];
%! model = svm_ex(d.TestData.train{:},'threads',1,'weights',expectiles_list);
%! [res, test_err] = model.test(d.TestData.test{:});
%! %hand_err = mean((res(:,3)-d.TestData.test{2}).^2)
%! assert(all(size(test_err) == [length(expectiles_list),3]))
%! test_err = test_err(1);
%! %assert(hand_err < 1.5)
%! assert(max(test_err) < 0.15)
%! %assert(abs(test_err - hand_err) < 1e-8)
%! 
%! 
%! %{
%! %% nplSVM (alarm = +1)
%! %npl_constraints = [3,4,6,9,12]./120
%! model = nplSVM(d.TestData.train{:}, 1, 0.04, 'threads',1);
%! [res, test_err] = model.test(d.TestData.test{:});
%! %hand_err = mean((res(:,3)-d.TestData.test{2}).^2)
%! assert(all(size(test_err) == [length(expectiles_list),3]))
%! test_err = test_err(1);
%! %assert(hand_err < 1.5)
%! assert(max(test_err) < 0.72)
%! %assert(abs(test_err - hand_err) < 1e-8)
%! 
%! % {
%!   model = nplSVM(Y ~ ., tt$train,threads=1, class=-1, constraint.factor=npl_constraints)
%!   result = test(model, tt$test)
%!   false_alarm_rate = apply(result[tt$test$Y==-1,]==1,2,mean)
%!   detection_rate = apply(result[tt$test$Y==1,]==1,2,mean)
%!   test_err = errors(result, showall=T)
%!   expect_equal(nrow(test_err),length(npl_constraints))
%!   expect_equal(test_err[,3], false_alarm_rate, tolerance=.0001, check.attributes=F )
%!   expect_equal(1-test_err[,1], detection_rate, tolerance=.0001, check.attributes=F )
%!   expect_equal(npl_constraints, false_alarm_rate, tolerance=.06, check.attributes=F )
%! 
%! %% nplSVM (alarm = -1)
%! npl_constraints = c(3,4,6,9,12)/120
%! model = nplSVM(Y ~ ., tt$train,threads=1, class=1, constraint.factor=npl_constraints)
%! result = test(model, tt$test)
%! false_alarm_rate = apply(result[tt$test$Y==1,]==-1,2,mean)
%! detection_rate = apply(result[tt$test$Y==-1,]==-1,2,mean)
%! test_err = errors(result, showall=T)
%! expect_equal(nrow(test_err),length(npl_constraints))
%! expect_equal(test_err[,2], false_alarm_rate, tolerance=.0001, check.attributes=F )
%! expect_equal(1-test_err[,1], detection_rate, tolerance=.0001, check.attributes=F )
%! expect_equal(npl_constraints, false_alarm_rate, tolerance=.06, check.attributes=F )
%! 
%! %% rocSVM
%! weight_steps = 4
%! model = rocSVM(Y ~ ., tt$train,threads=1, weight_steps=weight_steps)
%! result = test(model, tt$test)
%! test_err = errors(result, showall=T)
%! 
%! expect_equal(dim(test_err),c(weight_steps,3))
%! %}
%! 
%!xtest %ReadWrite(d)
%! model = mcSVM(d.TestData.train{:},'threads',1);
%! 
%! save svmModel1 model
%! clear model
%! assert(~exist('model'))
%! 
%! load svmModel1 model
%! 
%! hand_err = 1-mean(model.predict(d.TestData.test{1})==d.TestData.test{2});
%! [res, test_err] = model.test(d.TestData.test{:});
%! assert(all(size(test_err) == [1+6,3]))
%! test_err = test_err(1);
%! assert(hand_err < 0.3)
%! assert(test_err < 0.3)
%! assert(abs(test_err - hand_err) < 1e-8)
%! 
%! 
