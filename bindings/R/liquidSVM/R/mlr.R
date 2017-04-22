# Copyright 2015-2017 Philipp Thomann
#
# This file is part of liquidSVM.
#
# liquidSVM is free software: you can redistribute it and/or modify
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

#' liquidSVM functions for mlr
#' 
#' Allow for liquidSVM \code{\link{lsSVM}} and \code{\link{mcSVM}}
#' to be used in the \code{mlr} framework.
#' 
#' @note In order that mlr can find our learners liquidSVM has to be loaded
#' using e.g. \code{library(liquidSVM)}
#' \code{model <- train(...)}
#' @name mlr-liquidSVM
#' @param .learner see mlr-Documentation
#' @param .task see mlr-Documentation
#' @param .subset see mlr-Documentation
#' @param .weights see mlr-Documentation
#' @param .model the trained mlr-model, see mlr-Documentation
#' @param .newdata the test features, see mlr-Documentation
#' @param partition_choice the partition choice, see \link{Configuration}
#' @param partition_param a further param for partition choice, see \link{Configuration}
#' @param ... other parameters, see \link{Configuration}
#' @examples
#' \dontrun{
#' if(require(mlr)){
#' library(liquidSVM)
#' 
#' ## Define a regression task
#' task <- makeRegrTask(id = "trees", data = trees, target = "Volume")
#' ## Define the learner
#' lrn <- makeLearner("regr.liquidSVM", display=1)
#' ## Train the model use mlr::train to get the correct train function
#' model <- train(lrn,task)
#' pred <- predict(model, task=task)
#' performance(pred)
#' 
#' ## Define a classification task
#' task <- makeClassifTask(id = "iris", data = iris, target = "Species")
#' 
#' ## Define the learner
#' lrn <- makeLearner("classif.liquidSVM", display=1)
#' model <- train(lrn,task)
#' pred <- predict(model, task=task)
#' performance(pred)
#' 
#' ## or for probabilities
#' lrn <- makeLearner("classif.liquidSVM", display=1, predict.type='prob')
#' model <- train(lrn,task)
#' pred <- predict(model, task=task)
#' performance(pred)
#' 
#' } # end if(require(mlr))
#' }
NULL

commonParamSet <- function() {
  if(!requireNamespace('mlr', quietly=TRUE)) stop("this function needs mlr to be installed")
  if(!requireNamespace('ParamHelpers', quietly=TRUE)) stop("this function needs ParamHelpers to be installed")
  ParamHelpers::makeParamSet(
    ParamHelpers::makeLogicalLearnerParam(id = "scale", default = TRUE),
    ParamHelpers::makeDiscreteLearnerParam(id = "kernel", default = "gauss_rbf",
                                           values = c("gauss_rbf","poisson")),
    ParamHelpers::makeIntegerLearnerParam(id = "partition_choice", default = 0, lower = 0, upper = 6),
    ParamHelpers::makeNumericLearnerParam(id = "partition_param", default = -1,
                                          requires = quote(partition_choice >= 1L)),
    ParamHelpers::makeIntegerLearnerParam(id = "grid_choice", default = 0, lower = -2, upper = 2),
    ParamHelpers::makeIntegerLearnerParam(id = "folds", default = 5, lower = 1),
    ParamHelpers::makeNumericLearnerParam(id = "min_gamma", lower=0),
    ParamHelpers::makeNumericLearnerParam(id = "max_gamma", lower=0,requires = quote(min_gamma <= max_gamma)),
    ParamHelpers::makeIntegerLearnerParam(id = "gamma_steps", lower=0),
    ParamHelpers::makeNumericLearnerParam(id = "min_lambda", lower=0),
    ParamHelpers::makeNumericLearnerParam(id = "max_lambda", lower=0,requires = quote(min_lambda <= max_lambda)),
    ParamHelpers::makeIntegerLearnerParam(id = "lambda_steps", lower=0),
    ParamHelpers::makeDiscreteLearnerParam(id = "retrain_method", default = "select_on_each_fold",
                                           values = c("select_on_entire_train_Set","select_on_each_fold")),
    ParamHelpers::makeLogicalLearnerParam(id = "store_solutions_internally", default = TRUE),
    ParamHelpers::makeIntegerLearnerParam(id = "display", default = 0, lower = 0, upper=7),
    ParamHelpers::makeIntegerLearnerParam(id = "threads", default = 0, lower = -1)
  )
}

#' @export
#' @rdname mlr-liquidSVM
makeRLearner.regr.liquidSVM <- function() {
  if(!requireNamespace('mlr', quietly=TRUE)) stop("this function needs mlr to be installed")
  if(!requireNamespace('ParamHelpers', quietly=TRUE)) stop("this function needs ParamHelpers to be installed")
  mlr::makeRLearnerRegr(
    cl = "regr.liquidSVM",
    package = "liquidSVM",
    par.set = c(commonParamSet(),ParamHelpers::makeParamSet(
      ParamHelpers::makeNumericLearnerParam(id = "clip", lower = -1, default = -1, )
    )),
    #par.vals = list(fit = FALSE),
    properties = c("numerics", "factors"),
    name = "Support Vector Machines",
    short.name = "liquidSVM",
    note = "FIXME make integrated cross-validation more accessible."
  )
}

#' @export
#' @rdname mlr-liquidSVM
trainLearner.regr.liquidSVM <- function(.learner, .task, .subset, .weights = NULL, #scaled, clip, kernel,
                                       partition_choice=0, partition_param=-1, #grid_choice, folds,
                                       ...) {
  if(!requireNamespace('mlr', quietly=TRUE)) stop("this function needs mlr to be installed")
  f = mlr::getTaskFormula(.task)
  if(partition_param > 0) partition_choice <- c(partition_choice, partition_param)
  data <- mlr::getTaskData(.task, .subset)
  liquidSVM::lsSVM(f, data, partition_choice=partition_choice, ...)
}

#' @export
#' @rdname mlr-liquidSVM
predictLearner.regr.liquidSVM <- function(.learner, .model, .newdata, ...) {
  if(!requireNamespace('mlr', quietly=TRUE)) stop("this function needs mlr to be installed")
  predict.liquidSVM(.model$learner.model, newdata = .newdata, ...)#[, 1L]
}


#' @export
#' @rdname mlr-liquidSVM
makeRLearner.classif.liquidSVM <- function() {
  if(!requireNamespace('mlr', quietly=TRUE)) stop("this function needs mlr to be installed")
  if(!requireNamespace('ParamHelpers', quietly=TRUE)) stop("this function needs ParamHelpers to be installed")
  mlr::makeRLearnerClassif(
    cl = "classif.liquidSVM",
    package = "liquidSVM",
    par.set = c(commonParamSet(),ParamHelpers::makeParamSet(
      ParamHelpers::makeDiscreteLearnerParam(id = "mc_type", default = "AvA_hinge",
                              values =  c("AvA_hinge", "OvA_ls", "OvA_hinge", "AvA_ls")),
      ParamHelpers::makeNumericVectorLearnerParam(id = "weights", len = NA_integer_, lower = 0)
    )),
    #par.vals = list(fit = FALSE),
    properties = c("twoclass", "multiclass", "numerics", "factors", "prob", "class.weights"),
    class.weights.param = "weights",
    name = "Support Vector Machines",
    short.name = "liquidSVM",
    note = "FIXME make integrated cross-validation more accessible."
  )
}

#' @export
#' @rdname mlr-liquidSVM
trainLearner.classif.liquidSVM <- function(.learner, .task, .subset, .weights = NULL, #scaled, clip, kernel,
                                            partition_choice=0, partition_param=-1, #grid_choice, folds,
                                            ...) {
  if(!requireNamespace('mlr', quietly=TRUE)) stop("this function needs mlr to be installed")
  if(partition_param > 0) partition_choice <- c(partition_choice, partition_param)
  f <-  mlr::getTaskFormula(.task)
  data <- mlr::getTaskData(.task, .subset)
  predict.prob <- (.learner$predict.type=="prob")
  liquidSVM::mcSVM(f, data, partition_choice=partition_choice, predict.prob=predict.prob, ...)
}

#' @export
#' @rdname mlr-liquidSVM
predictLearner.classif.liquidSVM <- function(.learner, .model, .newdata, ...) {
  if(!requireNamespace('mlr', quietly=TRUE)) stop("this function needs mlr to be installed")
  m <- .model$learner.model
  ret <- predict.liquidSVM(m, newdata = .newdata, ...)
  if(.learner$predict.type=="prob"){
    ret <- as.matrix(ret)
    if(all(ret>=.5))
      warning("")
    ws_type <- getConfig(m, "WS_TYPE")
    if(ws_type==0){ ## binary classification
      colnames(ret) <- .model$task.desc$class.levels
    }else if(ws_type==2){ ## OvA
      colnames(ret) <- .model$task.desc$class.levels
    }else if(ws_type==1){ ## AvA
      warning("You choose mc_type='AvA' which gives not class probabilities but comparison probabilities")
    }
  }
  ret
}


