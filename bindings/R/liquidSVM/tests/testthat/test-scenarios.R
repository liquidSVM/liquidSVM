# Copyright 2015-2017 Philipp Thomann
# 
# This file is part of liquidSVM.
# 
#  liquidSVM is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
# 
# liquidSVM is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
# 
# You should have received a copy of the GNU Affero General Public License
# along with liquidSVM. If not, see <http://www.gnu.org/licenses/>.
# 

require(liquidSVM)

context("liquidSVM-scenarios")

orig <- options(liquidSVM.warn.suboptimal=FALSE)[[1]]

hand_err_name <- 'result'

test_that("mcSVM_AvA_hinge",{
  set.seed(123)
  
  #  tt <- liquidData('banana-mc')
  tt <- ttsplit(iris,testSize=30)
  model <- mcSVM(Species ~ ., tt$train,threads=1)
  hand_err <- 1-mean(predict(model, tt$test)==tt$test$Species)
  names(hand_err) <- hand_err_name
  test_err <- errors(test(model, tt$test))
  expect_equal(length(test_err),1+3)
  test_err <- test_err[1]
  expect_lt(hand_err,0.3)
  expect_lt(test_err,0.3)
  expect_equal(test_err,hand_err)
})

test_that("mcSVM_OvA_ls",{
  set.seed(123)
  
  #  tt <- liquidData('banana-mc')
  tt <- ttsplit(iris,testSize=30)
  model <- mcSVM(Species ~ ., tt$train,threads=1,mc_type="OvA_ls")
  hand_err <- 1-mean(predict(model, tt$test)==tt$test$Species)
  names(hand_err) <- hand_err_name
  test_err <- errors(test(model, tt$test))
  expect_equal(length(test_err),1+3)
  test_err <- test_err[1]
  expect_lt(hand_err,0.3)
  expect_lt(test_err,0.3)
  expect_equal(test_err,hand_err)
})

test_that("mcSVM_OvA_hinge",{
  set.seed(123)
  
  #  tt <- liquidData('banana-mc')
  tt <- ttsplit(iris,testSize=30)
  model <- mcSVM(Species ~ ., tt$train,threads=1,mc_type="OvA_hinge")
  hand_err <- 1-mean(predict(model, tt$test)==tt$test$Species)
  names(hand_err) <- hand_err_name
  test_err <- errors(test(model, tt$test))
  expect_equal(length(test_err),1+3)
  test_err <- test_err[1]
  expect_lt(hand_err,0.3)
  expect_lt(test_err,0.3)
  expect_equal(test_err,hand_err)
})

test_that("mcSVM_AvA_ls",{
  set.seed(123)
  
  #  tt <- liquidData('banana-mc')
  tt <- ttsplit(iris,testSize=30)
  model <- mcSVM(Species ~ ., tt$train,threads=1,mc_type="AvA_ls")
  hand_err <- 1-mean(predict(model, tt$test)==tt$test$Species)
  names(hand_err) <- hand_err_name
  test_err <- errors(test(model, tt$test))
  expect_equal(length(test_err),1+3)
  test_err <- test_err[1]
  expect_lt(hand_err,0.3)
  expect_lt(test_err,0.3)
  expect_equal(test_err,hand_err)
})

test_that("mcSVM_predict.prob.3levels",{
  set.seed(123)
  
  #  tt <- liquidData('banana-mc')
  tt <- ttsplit(iris,testSize=30)
  model <- mcSVM(Species ~ ., tt$train,threads=1,predict.prob=TRUE)
  probs <- predict(model, tt$test)
  
  expect_equal(ncol(probs),length(levels(iris$Species)))
  expect_lte(max(probs),1.001)
  expect_gte(min(probs),-0.001)
  expect_true(any(probs<0.5))
  
  result <- test(model, tt$test)
  expect_equivalent(apply(probs,1,which.max), as.integer(result[,1]))
})

test_that("mcSVM_predict.prob.4levels",{
  set.seed(123)
  
  tt <- liquidData('banana-mc',trainSize=200)
  model <- mcSVM(Y ~ ., tt$train,threads=1,predict.prob=TRUE,folds=3)
  probs <- predict(model, tt$test)
  
  expect_equal(ncol(probs),length(levels(tt$train$Y)))
  expect_lte(max(probs),1.001)
  expect_gte(min(probs),-0.001)
  expect_true(any(probs<0.5))
  
  result <- test(model, tt$test)
  expect_equivalent(apply(probs,1,which.max), as.integer(result[,1]))
})

test_that("mcSVM_predict.prob.2levels",{
  set.seed(123)
  
  tt <- liquidData('banana-bc',trainSize=300)
  model <- mcSVM(Y ~ ., tt$train,threads=1,predict.prob=TRUE,folds=3)
  probs <- predict(model, tt$test)
  
  expect_equal(ncol(probs),2)
  expect_lte(max(probs),1.001)
  expect_gte(min(probs),-0.001)
  expect_true(any(probs<0.5))
  
  result <- as.integer(colnames(probs))[apply(probs,1,which.max)]
  expect_lt(mean(result != tt$test$Y),0.3)
})

test_that("svm_predict.prob.3levels",{
  set.seed(123)
  
  #  tt <- liquidData('banana-mc')
  tt <- ttsplit(iris,testSize=30)
  model <- svm(Species ~ ., tt$train,threads=1,predict.prob=TRUE,folds=3)
  probs <- predict(model, tt$test)
  
  expect_equal(ncol(probs),length(levels(iris$Species)))
  expect_lte(max(probs),1.001)
  expect_gte(min(probs),-0.001)
  expect_true(any(probs<0.5))
  
  result <- test(model, tt$test)
  expect_equivalent(apply(probs,1,which.max), as.integer(result[,1]))
})

test_that("svm_predict.prob.4levels",{
  set.seed(123)
  
  tt <- liquidData('banana-mc',trainSize=200)
  model <- svm(Y ~ ., tt$train,threads=1,predict.prob=TRUE,folds=3)
  probs <- predict(model, tt$test)
  
  expect_equal(ncol(probs),length(levels(tt$train$Y)))
  expect_lte(max(probs),1.001)
  expect_gte(min(probs),-0.001)
  expect_true(any(probs<0.5))
  
  result <- test(model, tt$test)
  expect_equivalent(apply(probs,1,which.max), as.integer(result[,1]))
})


#####################################
#######    Regression    ############
#####################################


test_that("lsSVM",{
  set.seed(123)
  
  tt <- ttsplit(quakes,testSize=600)
  model <- lsSVM(mag ~ ., tt$train,threads=1)
  hand_err <- mean((predict(model, tt$test)-tt$test$mag)^2,folds=3)
  names(hand_err) <- hand_err_name
  test_err <- errors(test(model, tt$test))
  expect_equal(length(test_err),1)
  expect_lt(hand_err,0.2)
  expect_lt(test_err,0.2)
  expect_lt(abs(test_err-hand_err),1e5)
})

test_that("qtSVM",{
  set.seed(123)
  quantiles_list <- c(0.05, 0.1, 0.5, 0.9, 0.95)
  
  tt <- ttsplit(quakes,testSize=600)
  model <- qtSVM(mag ~ ., tt$train,threads=1, weights=quantiles_list,folds=3)
  result <- test(model, tt$test)
  hand_err <- mean((result[,3]-tt$test$mag)^2)
  names(hand_err) <- hand_err_name
  test_err <- errors(result)
  expect_equal(length(test_err),length(quantiles_list))
  expect_lt(hand_err,0.25)
  expect_lt(max(test_err),0.2)
  # expect_lt(abs(test_err-hand_err),1e5)
})

test_that("exSVM",{
  set.seed(123)
  expectiles_list <- c(0.05, 0.1, 0.5, 0.9, 0.95)
  
  tt <- ttsplit(quakes,testSize=600)
  model <- exSVM(mag ~ ., tt$train,threads=1, weights=expectiles_list,folds=3)
  result <- test(model, tt$test)
  hand_err <- mean((result[,3]-tt$test$mag)^2)
  names(hand_err) <- hand_err_name
  test_err <- errors(result)
  expect_equal(length(test_err),length(expectiles_list))
  expect_lt(hand_err,0.25)
  expect_lt(max(test_err),0.2)
  # expect_lt(abs(test_err-hand_err),1e5)
})

test_that("nplSVM (alarm = +1)",{
  set.seed(123)
  npl_factors <- c(3,6,12)/6
  
  tt <- liquidData('banana-bc',trainSize=30)
  
  model <- nplSVM(Y ~ ., tt$train,threads=1, class=-1, constraint.factors=npl_factors,folds=3)
  result <- test(model, tt$test)
  false_alarm_rate <- apply(result[tt$test$Y==-1,]==1,2,mean)
  detection_rate <- apply(result[tt$test$Y==1,]==1,2,mean)
  test_err <- errors(result, showall=T)
  expect_equal(nrow(test_err),length(npl_factors))
  expect_equal(test_err[,3], false_alarm_rate, tolerance=.0001, check.attributes=F )
  expect_equal(1-test_err[,1], detection_rate, tolerance=.0001, check.attributes=F )
  expect_equal(0.05 * npl_factors, false_alarm_rate, tolerance=.06, check.attributes=F )
})

test_that("nplSVM (alarm = -1)",{
  set.seed(123)
  npl_factors <- c(3,6,12)/6
  
  tt <- liquidData('banana-bc',trainSize=300)
  
  model <- nplSVM(Y ~ ., tt$train,threads=1, class=1, constraint.factors=npl_factors,folds=3)
  result <- test(model, tt$test)
  false_alarm_rate <- apply(result[tt$test$Y==1,]==-1,2,mean)
  detection_rate <- apply(result[tt$test$Y==-1,]==-1,2,mean)
  test_err <- errors(result, showall=T)
  expect_equal(nrow(test_err),length(npl_factors))
  expect_equal(test_err[,2], false_alarm_rate, tolerance=.0001, check.attributes=F )
  expect_equal(1-test_err[,1], detection_rate, tolerance=.0001, check.attributes=F )
  expect_equal(0.05 * npl_factors, false_alarm_rate, tolerance=.06, check.attributes=F,use.names=F )
})

test_that("rocSVM",{
  set.seed(123)
  weight_steps <- 4
  
  tt <- liquidData('banana-bc',trainSize=400)
  
  model <- rocSVM(Y ~ ., tt$train,threads=1, weight_steps=weight_steps,folds=3)
  result <- test(model, tt$test)
  test_err <- errors(result, showall=T)
  
  expect_equal(dim(test_err),c(weight_steps,3))
  #....????
})

test_that("plotROC p",{
  set.seed(123)
  weight_steps <- 4
  
  tt <- liquidData('banana-bc',trainSize=300)
  
  model <- rocSVM(Y~.,tt$train,threads=1, weight_steps=weight_steps,folds=3)

  plotROC(model ,tt$test)
  # or:
  result <- test(model, tt$test)
  plotROC(result, tt$test$Y)

  model.ls <- lsSVM(Y~., tt$train,threads=1,folds=3)
  result <- plotROC(model.ls, tt$test)
  
  if(file.exists('Rplots.pdf')){
    unlink('Rplots.pdf')
  }
})


test_that("bsSVM ls",{
  set.seed(123)
  solver <- 1
  
  tt <- ttsplit(quakes,testSize=600)

  model <- bsSVM(mag ~ ., tt$train, ws.size=50,threads=1, solver=solver)
  result <- test(model, tt$test)
  test_err <- errors(result)
  
  expect_equal(length(test_err),6)
  #....????
})

options(liquidSVM.warn.suboptimal=orig)



