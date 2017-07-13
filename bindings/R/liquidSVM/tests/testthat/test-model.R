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

context("liquidSVM-model")

orig <- options(liquidSVM.warn.suboptimal=FALSE, threads=1)[[1]]

hand_err_name <- 'result'

test_that("print model",{
  set.seed(123)
  
  #  tt <- liquidData('banana-mc')
  tt <- ttsplit(iris,testSize=30)
  
  # only init
  model <- init.liquidSVM(Species ~ ., tt$train)
  expect_output(print(model))
  
  # after delete:
  clean(model)
  expect_output(print(model), 'deleted')
  
  
  # without selecting
  model <- mcSVM(Species ~ ., tt$train, do.select=FALSE)
  expect_output(print(model))
  
  # now for th full train/select/test
  model <- mcSVM(Species ~ ., tt)
  expect_output(print(model))
})

test_that("save/load model",{
  skip_on_cran()
  
  set.seed(123)
  
  #  tt <- liquidData('banana-mc')
  tt <- ttsplit(iris,testSize=30)
  filename <- tempfile("liquidSVM.testthat.iris",fileext = ".fsol")
  modelOrig <- mcSVM(Species ~ ., tt$train)
  write.liquidSVM(modelOrig, filename)
  clean(modelOrig)
  
  expect_true(file.exists(filename))
  
  model <- read.liquidSVM(filename)
  
  unlink(filename)
  expect_false(file.exists(filename))
  
  hand_err <- 1-mean(predict(model, tt$test)==tt$test$Species)
  names(hand_err) <- hand_err_name
  test_err <- errors(test(model, tt$test))
  expect_equal(length(test_err),1+3)
  test_err <- test_err[1]
  expect_lt(hand_err,0.3)
  expect_lt(test_err,0.3)
  expect_equal(test_err,hand_err)
})

test_that("save/load model without data",{
  skip_on_cran()
  
  set.seed(123)
  
  #  tt <- liquidData('banana-mc')
  tt <- ttsplit(iris,testSize=30)
  filename <- tempfile("liquidSVM.testthat.iris",fileext = ".sol")
  modelOrig <- mcSVM(Species ~ ., tt$train)
  write.liquidSVM(modelOrig, filename)
  clean(modelOrig)
  
  expect_true(file.exists(filename))
  
  model <- read.liquidSVM(filename, Species ~ ., tt$train)
  
  unlink(filename)
  expect_false(file.exists(filename))
  
  expect_true(is.environment(model))
  
  hand_err <- 1-mean(predict(model, tt$test)==tt$test$Species)
  names(hand_err) <- hand_err_name
  test_err <- errors(test(model, tt$test))
  expect_equal(length(test_err),1+3)
  test_err <- test_err[1]
  expect_lt(hand_err,0.3)
  expect_lt(test_err,0.3)
  expect_equal(test_err,hand_err)
})

test_that("serialize/unserialize model",{
  skip_on_cran()
  
  set.seed(123)
  
  #  tt <- liquidData('banana-mc')
  tt <- ttsplit(iris,testSize=30)
  
  modelOrig <- mcSVM(Species ~ ., tt$train)
  obj <- serialize.liquidSVM(modelOrig)
  clean(modelOrig)
  
  model <- unserialize.liquidSVM(obj)

  expect_true(is.environment(model))
  
  hand_err <- 1-mean(predict(model, tt$test)==tt$test$Species)
  names(hand_err) <- hand_err_name
  test_err <- errors(test(model, tt$test))
  expect_equal(length(test_err),1+3)
  test_err <- test_err[1]
  expect_lt(hand_err,0.3)
  expect_lt(test_err,0.3)
  expect_equal(test_err,hand_err)
})

test_that("serialize/unserialize model using R serialize",{
  return() ## since it is not implemented currently
  
  skip_on_cran()
  
  set.seed(123)
  
  #  tt <- liquidData('banana-mc')
  tt <- ttsplit(iris,testSize=30)
  
  modelOrig <- mcSVM(Species ~ ., tt$train)
  obj <- serialize(modelOrig, NULL, refhook=svmSerializeHook)
  clean(modelOrig)
  
  model <- unserialize(obj, refhook=svmUnserializeHook)
  
  expect_true(is.environment(model)) ## this would fail at the moment...
  
  hand_err <- 1-mean(predict(model, tt$test)==tt$test$Species)
  names(hand_err) <- hand_err_name
  test_err <- errors(test(model, tt$test))
  expect_equal(length(test_err),1+3)
  test_err <- test_err[1]
  expect_lt(hand_err,0.3)
  expect_lt(test_err,0.3)
  expect_equal(test_err,hand_err)
})

test_that("getCover",{
  skip_on_cran()
  
  set.seed(123)
  banana <- liquidData('banana-mc',trainSize=900)
  model <- mcSVM(Y~.,banana$train, voronoi=c(4,300), folds=2)
  # task 4 is predicting 2 vs 3
  cover <- getCover(model,task=4)
  expect_true(all(banana$train$Y[cover$indices] %in% c(2,3)))
  expect_equal(cover$task, 4)
  
  # centers <- cover$samples
  # # we are considering task 4 and hence only show labels 2 and 3:
  # bananaSub <- banana$train[banana$train$Y %in% c(2,3),]
  # distances <- as.matrix(dist(bananaSub[,-1]))
  # cells <- apply(distances[bananaSub %in% c(2,3),cover$indices],1,which.min)
  # # and you can check that the cell sizes are as reported in the training phase for task 4
  # table(cells)
  
})

test_that("getSolution",{
  skip_on_cran()
  
  set.seed(123)
  x <- seq(0,1,by=.01)
  y <- sin(x*10)
  f <- 2
  model <- lsSVM(x,y)
  sol <- getSolution(model, 1,1,f)
  n <- length(sol$sv)
  expect_lte(n,length(x))
  expect_length(sol$sv, n)
  expect_length(sol$coeff, n)
  expect_length(sol$samples, n)
  expect_length(sol$labels, n)
  expect_equal(sol$task, 1)
  expect_equal(sol$cell, 1)
  expect_equal(sol$fold, f)
})

test_that("groupedFolds",{
  set.seed(123)
  tt <- ttsplit(iris, testProb=0.5)
  groups <- sample.int(n=50, size=nrow(tt$train), replace=T)
  
  model <- lsSVM(Species~., tt$train, groupIds=groups)
  result <- test(model, tt$test)
  expect_lte(errors(result), 0.1)
  
  model <- lsSVM(Species~., tt$train, groupIds=factor(groups))
  result <- test(model, tt$test)
  expect_lte(errors(result), 0.1)
  
  model <- mcSVM(Species~., tt$train, groupIds=groups)
  result <- test(model, tt$test)
  expect_lte(errors(result)[1], 0.1)
})


test_that("suboptimal warning",{
  orig <- options(liquidSVM.warn.suboptimal=TRUE)[[1]]
  set.seed(123)
  
  tt <- ttsplit(iris,testSize=30)
  expect_warning(model <- svm(Species ~ ., tt$train,gammas=c(1,10),lambdas=c(.1,1)), 'optimal')
  options(liquidSVM.warn.suboptimal=orig)
})

test_that("compilation info",{
  expect_true(any(nchar(compilationInfo())>0))
})


options(liquidSVM.warn.suboptimal=orig)

