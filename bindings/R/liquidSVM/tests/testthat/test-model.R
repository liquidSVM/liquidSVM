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

orig <- options(liquidSVM.warn.suboptimal=FALSE)[[1]]

hand_err_name <- 'result'

test_that("save/load model",{
  skip_on_cran()
  
  set.seed(123)
  
  #  tt <- liquidData('banana-mc')
  tt <- ttsplit(iris,testSize=30)
  filename <- tempfile("liquidSVM.testthat.iris",fileext = ".fsol")
  modelOrig <- mcSVM(Species ~ ., tt$train,threads=1)
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
  modelOrig <- mcSVM(Species ~ ., tt$train,threads=1)
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
  
  modelOrig <- mcSVM(Species ~ ., tt$train,threads=1)
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
  
  modelOrig <- mcSVM(Species ~ ., tt$train,threads=1)
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

options(liquidSVM.warn.suboptimal=orig)

