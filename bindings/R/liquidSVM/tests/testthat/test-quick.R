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

context("liquidSVM-quick")

orig <- options(liquidSVM.warn.suboptimal=FALSE, threads=1)[[1]]

hand_err_name <- 'result'

test_that("quick iris",{
  set.seed(123)
  
  tt <- ttsplit(iris,testSize=30)
  model <- svm(Species ~ ., tt$train)
  expect_equal(nrow(model$last_result),0)
  hand_err <- 1-mean(predict(model, tt$test)==tt$test$Species)
  names(hand_err) <- hand_err_name
  test_err <- errors(test(model, tt$test))
  expect_equal(length(test_err),4)
  test_err <- test_err[1]
  expect_lt(hand_err,0.3)
  expect_lt(test_err,0.3)
  expect_equal(test_err,hand_err)
})

test_that("quick iris last_result",{
  set.seed(123)
  
  tt <- ttsplit(iris,testSize=30)
  model <- svm(Species ~ ., tt)
  expect_true(nrow(model$last_result)>0)
#  expect_true('last_result' %in% ls(model))
  hand_err <- 1-mean(predict(model, tt$test)==tt$test$Species)
  names(hand_err) <- hand_err_name
  test_err <- errors(test(model, tt$test))
  expect_equal(length(test_err),4)
  test_err <- test_err[1]
  expect_lt(hand_err,0.3)
  expect_lt(test_err,0.3)
  expect_equal(test_err,hand_err)
})

test_that("quick iris no-formula",{
  set.seed(123)
  
  tt <- ttsplit(iris,testSize=30)
  model <- svm(tt$train[,-5], tt$train$Species)
  expect_equal(nrow(model$last_result),0)
  hand_err <- 1-mean(predict(model, tt$test[,-5])==tt$test$Species)
  expect_true(nrow(model$last_result)>0)
  names(hand_err) <- hand_err_name
  test_err <- errors(test(model, tt$test[,-5],tt$test$Species))
  expect_equal(length(test_err),4)
  test_err <- test_err[1]
  expect_lt(hand_err,0.3)
  expect_lt(test_err,0.3)
  expect_equal(test_err,hand_err)
 })

# test_that("quick covtype",{
#   set.seed(123)
#   
#   co <- liquidData('covtype.1000')
#   model <- svm(Y ~ ., co$train)
#   expect_false('last_result' %in% ls(model))
#   expect_gt(mean(predict(model, co$test)==co$test$Y),0.7)
#   expect_true('last_result' %in% ls(model))
# })

test_that("quick quakes",{
  set.seed(123)
  
  tt <- ttsplit(quakes,testSize=600)
  model <- svm(mag ~ ., tt$train)
  expect_equal(nrow(model$last_result),0)
  hand_err <- mean((predict(model, tt$test)-tt$test$mag)^2)
  expect_true(nrow(model$last_result)>0)
  names(hand_err) <- hand_err_name
  test_err <- errors(test(model, tt$test))
  expect_equal(length(test_err),1)
  expect_lt(hand_err,0.2)
  expect_lt(test_err,0.2)
  expect_lt(abs(test_err-hand_err),1e5)
})

test_that("quick 1dim",{
  set.seed(123)
  
  tt <- liquidData('reg-1d',trainSize=400)
  trX <- tt$train$X1
  trY <- tt$train$Y
  tsX <- tt$test$X1
  tsY <- tt$test$Y
  
  expect_null(dim(trX))
  expect_null(dim(trY))
  expect_null(dim(tsX))
  expect_null(dim(tsY))
  
  model <- svm(trX,trY)
  expect_equal(nrow(model$last_result),0)
  hand_err <- mean((predict(model, tsX)-tsY)^2)
  expect_true(nrow(model$last_result)>0)
  names(hand_err) <- hand_err_name
  test_err <- errors(test(model, tsX, tsY))
  expect_equal(length(test_err),1)
  expect_lt(hand_err,0.2)
  expect_lt(test_err,0.2)
  expect_lt(abs(test_err-hand_err),1e5)
})

test_that("quick iris environment",{
  set.seed(123)
  
  tt <- ttsplit(iris,testSize=30)
  
  attach(tt$train)
  model <- svm(Species ~ Sepal.Length+Sepal.Width+Petal.Length+Petal.Width)
  detach(tt$train)
  expect_equal(nrow(model$last_result),0)
  hand_err <- 1-mean(predict(model, tt$test)==tt$test$Species)
  names(hand_err) <- hand_err_name
  test_err <- errors(test(model, tt$test))
  expect_equal(length(test_err),4)
  test_err <- test_err[1]
  expect_lt(hand_err,0.3)
  expect_lt(test_err,0.3)
  expect_equal(test_err,hand_err)
})

test_that("quick data as name",{
  set.seed(123)
  
  tt <- liquidData('banana-bc')
  model <- svm(Y ~ ., 'banana-bc', folds=2, gammas=c(1,2,4,8))
  
  expect_equal(nrow(model$last_result),nrow(tt$test))
  
  result <- predict(model, tt$test)
  hand_err <- 1-mean(result==tt$test$Y)
  # names(hand_err) <- hand_err_name
  test_err <- errors(test(model, tt$test))
  expect_equal(length(test_err),1)
  test_err <- test_err[1]
  expect_lt(hand_err,0.3)
  expect_lt(test_err,0.3)
  expect_equal(test_err,hand_err)
  
  result2 <- predict(model, tt$train)
  result3 <- predict(model, 'banana-bc')
  expect_equal(result2,result3)
})


test_that("quick threads",{
  skip_on_cran()

  set.seed(123)
  tt <- liquidData('banana-bc')
  a <- system.time(model <- svm(Y ~ ., tt$train, do.select=FALSE, folds=2))
  b <- system.time(model <- svm(Y ~ ., tt$train,threads=2, do.select=FALSE, folds=2))
  expect_gt(a['elapsed'],b['elapsed'])
  expect_lt(a['user.self'],b['user.self'])
  expect_gt(b['user.self']/b['elapsed'],1.5)
})

options(liquidSVM.warn.suboptimal=orig)


