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

context("liquidSVM-mlr")

orig <- options(liquidSVM.warn.suboptimal=FALSE, threads=1)[[1]]

hand_err_name <- 'result'

test_that("mlr-regr",{
  set.seed(123)

  skip_if_not(require(mlr))

  ## Define a regression task
  task <- makeRegrTask(id = "trees", data = trees, target = "Volume")
  ## Define the learner
  lrn <- makeLearner("regr.liquidSVM", display=0)
  ## Train the model use mlr::train to get the correct train function
  model <- train(lrn,task)
  pred <- predict(model, task=task)
  expect_lt(performance(pred),10)
})
  
test_that("mlr-class",{
  set.seed(123)
  
  skip_if_not(require(mlr))
  
  ## Define a classification task
  task <- makeClassifTask(id = "iris", data = iris, target = "Species")
  
  ## Define the learner
  lrn <- makeLearner("classif.liquidSVM", display=0)
  model <- train(lrn,task)
  pred <- predict(model, task=task)
  expect_lt(performance(pred),0.05)
})

test_that("mlr-class-prob",{
  set.seed(123)
  
  skip_if_not(require(mlr))

  ## Define a classification task
  task <- makeClassifTask(id = "iris", data = iris, target = "Species")
  
  ## Define the learner
  lrn <- makeLearner("classif.liquidSVM", display=0, predict.type='prob')
  model <- train(lrn,task)
  pred <- predict(model, task=task)
  performance(pred)
  expect_lt(performance(pred),0.05)
})


options(liquidSVM.warn.suboptimal=orig)



