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

#' liquidSVM for R
#' 
#' Support vector machines (SVMs) and related kernel-based learning algorithms are a well-known
#' class of machine learning algorithms, for non-parametric classification and regression.
#' \pkg{liquidSVM} is an implementation of SVMs whose key features are:
#' \itemize{
#' \item fully integrated hyper-parameter selection,
#' \item extreme speed on both small and large data sets,
#' \item full flexibility for experts, and
#' \item inclusion of a variety of different learning scenarios:
#' \itemize{
#' \item multi-class classification, ROC, and Neyman-Pearson learning, and
#' \item least-squares, quantile, and expectile regression
#' }
#' }
#' \ifelse{latex}{\out{\clearpage}}{}
#' Further information is available in the following vignettes:
#' \tabular{ll}{
#' \code{demo} \tab liquidSVM Demo (source, pdf)
#' \cr\code{documentation} \tab liquidSVM Documentation (source, pdf)\cr
#' }
#' 
#' In \pkg{liquidSVM} an application cycle is divided into a training phase, in which various SVM
#' models are created and validated, a selection phase, in which the SVM models that best
#' satisfy a certain criterion are selected, and a test phase, in which the selected models are
#' applied to test data. These three phases are based upon several components, which can be
#' freely combined using different components: solvers, hyper-parameter selection, working sets.
#' All of these can be configured (see \link{Configuration}) a
#' 
#' For instance multi-class classification with \eqn{k} labels has to be delegated to several binary classifications
#' called \emph{tasks} either using all-vs-all (\eqn{k(k-1)/2} tasks on the corresponding subsets) or
#' one-vs-all (\eqn{k} tasks on the full data set).
#' Every task can be split into \emph{cells} in order to handle larger data sets (for example \eqn{>10000} samples).
#' Now for every task and every cell, several \emph{folds} are created to enable cross-validated hyper-parameter selection.
#' 
#' The following learning scenarios can be used out of the box:
#' \describe{
#' \item{\code{\link{mcSVM}}}{binary and multi-class classification}
#' \item{\code{\link{lsSVM}}}{least squares regression}
#' \item{\code{\link{nplSVM}}}{Neyman-Pearson learning to classify with a specified rate on one type of error}
#' \item{\code{\link{rocSVM}}}{Receivert Operating Characteristic (ROC) curve to solve multiple weighted binary classification problems.}
#' \item{\code{\link{qtSVM}}}{quantile regression}
#' \item{\code{\link{exSVM}}}{expectile regression}
#' \item{\code{\link{bsSVM}}}{bootstrapping}
#' }
#' 
#' To calculate kernel matrices as used by the SVM we also provide for convenience the function
#' \code{\link{kern}}.
#' 
#' \pkg{liquidSVM} can benefit heavily from native compilation, hence we recommend to (re-)install it
#' using the information provided in the \href{../doc/documentation.html#Installation}{installation section}
#' of the documentation vignette.
#' 
#' 
#' @section Known issues:
#' 
#' Interruption (Ctrl-C) of running train/select/test phases is honored, but can leave
#' the C++ library in an inconsistent state, so that it is better to save your work and restart your
#' \R session.
#' 
#' \pkg{liquidSVM} is multi-threaded and is difficult to be multi-threaded externally, see
#' \href{../doc/documentation.html#Using external parallelization}{documentation}
#' 
#' @docType package
#' @name liquidSVM-package
#' @author
#' Ingo Steinwart \email{ingo.steinwart@@mathematik.uni-stuttgart.de},
#' Philipp Thomann \email{philipp.thomann@@mathematik.uni-stuttgart.de}
#' 
#' Maintainer: Philipp Thomann \email{philipp.thomann@@mathematik.uni-stuttgart.de}
#' @references \url{http://www.isa.uni-stuttgart.de}
#' @useDynLib liquidSVM, .registration = TRUE
#' @keywords SVM
#' @aliases liquidSVM
#' @importFrom stats get_all_vars model.frame na.pass terms
#' @importFrom utils ls.str read.table str write.table
#' @seealso \code{\link{init.liquidSVM}}, \code{\link{trainSVMs}}, \code{\link{predict.liquidSVM}}, \code{\link{clean.liquidSVM}}, and \code{\link{test.liquidSVM}}, \link{Configuration};
#' @examples
#' set.seed(123)
#' ## Multiclass classification
#' modelIris <- svm(Species ~ ., iris)
#' y <- predict(modelIris, iris)
#' 
#' ## Least Squares
#' modelTrees <- svm(Height ~ Girth + Volume, trees)
#' y <- predict(modelTrees, trees)
#' plot(trees$Height, y)
#' test(modelTrees, trees)
#' 
#' ## Quantile regression
#' modelTrees <- qtSVM(Height ~ Girth + Volume, trees, scale=TRUE)
#' y <- predict(modelTrees, trees)
#' 
#' ## ROC curve
#' modelWarpbreaks <- rocSVM(wool ~ ., warpbreaks, scale=TRUE)
#' y <- test(modelWarpbreaks, warpbreaks)
#' plotROC(y,warpbreaks$wool)
NULL


#' Convenience function to initialize, train, select, and optionally test an SVM.
#' 
#' The model is inited using the features and labels provided and training and selection is performed.
#' If the labels are given as a \code{factor} classification is performed else least squares regression.
#' If testing data is provided then this is used to calculate predictions and if test labels are provided
#' also the test error and both are saved in \code{$last_result} of the returned \code{svm} object.
#' 
#' The training data can either be provided using a formula and a corresponding \code{data.frame}
#' or the features and the labels are given directly.
#' 
#' \code{svm} has one more difference to \code{\link{lsSVM}} and \code{\link{mcSVM}}
#' because it uses \code{scale=TRUE} by default and the others do not.
#' 
#' @param d level of display information
#' @param do.select can be set to a list to args to be passed to the select phase
#' @param testdata if supplied then also testing is performed.
#'   If this is \code{NULL} but \code{y} is of type \code{liquidData} then
#'   \code{y$test} is used.
#' @param testdata_labels the labels used if testing is also perfomed.
#' @param scale if \code{TRUE} scales the features in the internal representation
#' to values between 0 and 1.
#' @param predict.prob If \code{TRUE} then a LS-svm will be trained and
#'   the conditional probabilities for the binary classification problems will be estimated.
#'   This also restricts the choices of \code{mc_type} to \code{c("OvA_ls","AvA_ls")}.
#' @return an object of type \code{svm}. Depending on the usage this object
#' has also \code{$train_errors}, \code{$select_errors}, and \code{$last_result}
#' properties.
#' @export
#' @inheritParams lsSVM
#' @inheritParams init.liquidSVM.default
#' @seealso \code{\link{lsSVM}}, \code{\link{mcSVM}}, \code{\link{init.liquidSVM}}, \code{\link{trainSVMs}}, \code{\link{selectSVMs}}
#' @examples
#' # since Species is a factor the following performs multiclass classification
#' modelIris <- svm(Species ~ ., iris)
#' # equivalently
#' modelIris <- svm(iris[,1:4], iris$Species)
#' 
#' # since Height is numeric the following performs least-squares regression
#' modelTrees <- svm(Height ~ Girth + Volume, trees)
#' # equivalently
#' modelTrees <- svm(trees[,c(1,3)],trees$Height)
svm <- function(x,y, ..., do.select=TRUE, testdata=NULL, testdata_labels=NULL, scenario=NULL, d=NULL,
                scale=TRUE, predict.prob=FALSE
                # , scenario=NULL ,useCells=F ,threads=1, gpus=0, display=0,folds=5
                ){
  if(predict.prob && is.null(scenario))
    scenario <- c("MC","OvA_ls")
  model <- init.liquidSVM(x, y, ..., scenario=scenario, scale=scale, d=d); # , threads=threads, gpus=gpus, display=d)
  model$predict.prob <- predict.prob
  if(predict.prob)
    model$predict.cols <- -1L
  trainSVMs(model,do.select=do.select)
  return_with_test(model, x, y, ..., testdata=testdata, testdata_labels=testdata_labels)
}



#' Initialize an SVM object.
#' 
#' \strong{Should only be used by experts!}
#' This initializes a \code{svm} object and allocates in C++ an SVM model to which it keeps a reference.
#' 
#' Since it binds heap memory it has to be released using \code{\link{clean.liquidSVM}} which
#' is also performed at garbage collection.
#' 
#' The training data can either be provided using a formula and a corresponding \code{data.frame}
#' or the features and the labels are given directly.
#' 
#' @param d level of display information
#' @inheritParams lsSVM
#' @return an object of type \code{svm}
#' @export
#' @aliases init.liquidSVM.default, init.liquidSVM.formula
#' @seealso \code{\link{svm}}, \code{\link{predict.liquidSVM}}, \code{\link{test.liquidSVM}} and \code{\link{clean.liquidSVM}}
#' @examples
#' modelTrees <- init.liquidSVM(Height ~ Girth + Volume, trees[1:20, ])  # least squares
#' modelIris <- init.liquidSVM(Species ~ ., iris)  # multiclass classification
init.liquidSVM <- function(x, y, ...) UseMethod("init.liquidSVM", x)

#' @describeIn init.liquidSVM Initialize SVM model using a a formula and data
#' @export
init.liquidSVM.formula <- function(x,y, ..., d=NULL){
  formula <- x
  data <- if(missing(y)){ NULL }else{ y }
  if(missing(data)){
    data <- parent.frame()
  }else if(is.character(data)){
    data <- liquidData(data)
  }
  if(inherits(data,"liquidData")){
    data <- data$train
  }
  mf <- model.frame(formula, data)
  labels <- mf[[1]]
  ## if there is only one x variable we still want this to be a data.frame:
  train <- as.data.frame(mf[-1])
  
  all_vars <- colnames(get_all_vars(formula, data))
  explanatory <- attributes(terms(mf))$term.labels
  
  if(is.data.frame(train) && all(explanatory %in% colnames(train))){
    #cat("---Changing train variables", dim(train))
    train <- train[,explanatory]
    #cat(" to", dim(train),fill=T)
  }
  
  model <- init.liquidSVM.default(train, labels, ...,d=d)

  model$formula <- formula
  model$explanatory <- explanatory
  model$all_vars <- all_vars
  
  model
}

#' @param scenario configures the model for a learning scenario:
#' E.g. \code{scenario='ls', scenario='mc', scenario='npl', scenario='qt'} etc.
#' Unlike the specialized functions \code{qtSVM, exSVM, nplSVM} etc.
#' this does not trigger the correct \code{select}
#' @param useCells if \code{TRUE} partitions the problem (equivalent to \code{partition_choice=6})
#' @param sampleWeights vector of weights for every sample or \code{NULL} (default) [currently has no effect]
#' @param groupIds vector of integer group ids for every sample or \code{NULL} (default).
#' If not \code{NULL} this will do group-wise folds, see \code{folds_kind='GROUPED'}.
#' @param ids vector of integer ids for every sample or \code{NULL} (default) [currently has no effect]
#' @export
#' @describeIn init.liquidSVM Initialize SVM model using a data frame and a label vector
init.liquidSVM.default <- function(x,y, scenario=NULL, useCells=NULL, ..., sampleWeights=NULL, groupIds=NULL, ids=NULL, d=NULL){
  train <- x
  labels <- y
#   mf <- model.frame(formula, data)
  lev <- levels(labels)
  if(!is.null(lev)){
    labels <- factorToNumeric(labels)
  }else{
    labels <- as.numeric(labels)
  }
  
  noSamples <- if(is.null(dim(x))) length(x) else nrow(x)
  if(length(labels) != noSamples)
    stop('x and y do not have the same number of samples')
  
  failedAnnotation <- function(x, typeF=function(x) is.integer(x) || is.factor(x)){
    if(is.null(x)) return(FALSE)
    if(!typeF(x)) return(TRUE)
    if(length(x) != length(labels)) return(TRUE)
    if(any(as.numeric(x) < 0)) return(TRUE)
    return(FALSE)
  }
  if(failedAnnotation(sampleWeights, typeF=is.numeric)) stop('sampleWeights has to be NULL or positive numeric of same length as labels.')
  if(failedAnnotation(groupIds)) stop('groupIds has to be NULL or non-negative integers of same length as labels.')
  if(failedAnnotation(ids)) stop('ids has to be NULL or non-negative integers of same length as labels.')
  
  if(any(is.na(labels)))
    warning("Training labels have NA values, removing the corresponding samples.")
  if(any(is.na(train)))
    warning("Training data has NA values, removing the corresponding samples.")
  if(!is.null(sampleWeights) && any(is.na(sampleWeights)))
    warning("sampleWeights has NA values, removing the corresponding samples.")
  if(!is.null(groupIds) && any(is.na(groupIds)))
    warning("groupIds has NA values, removing the corresponding samples.")
  if(!is.null(ids) && any(is.na(ids)))
    warning("ids has NA values, removing the corresponding samples.")
  val_index <- stats::complete.cases(train, labels, sampleWeights, groupIds, ids)
  train <- if(is.null(dim(train)))
    train[val_index]
  else
    train[val_index,]
  labels <- labels[val_index]
  
  if(is.null(dim(train))){
    dim <- 1L
    train_size <- as.integer(length(train))
  }else{
    dim <- as.integer(ncol(train))
    train_size <- as.integer(nrow(train))
  }
  if(FALSE){
  str(as.numeric(labels))
  str(train)
  str(data.matrix(train))
  str(as.numeric(t(data.matrix(train))))
  }
  
  # we do not have to transpose train since __LIQUIDSVM_DATA_BY_COLS = true
  if(any(d>=2))  ### any(), since d is NULL by default
    str(as.numeric(labels))
  cookie <- .Call('liquid_svm_R_init',
            as.numeric(data.matrix(train)), as.numeric(labels),
            as.numeric(sampleWeights), as.integer(groupIds), as.integer(ids),
            PACKAGE='liquidSVM')
  
  stopifnot( is.integer(cookie) & length(cookie)==1 & cookie>=0)

  result <- liquidSVMclass$new(cookie=cookie, solver=-1L, dim=dim, levels=lev,
                          #formula=NULL, explanatory=NULL, all_vars=NULL,
                          train_data=as.data.frame(train), train_labels=labels)
  class(result) <- 'liquidSVM'
  
  # set default values first, they get overwritten by scenario and others
  opts <- names(options())
  for(o in opts[startsWith(opts, 'liquidSVM.default.')]){
    setConfig(result, substring(o, 1+nchar('liquidSVM.default.')), getOption(o))
  }
  
  if(is.null(scenario)){
    if(is.null(lev))
      scenario <- "LS"
    else
      scenario <- "MC"
  }
  
  setConfig(result, "SCENARIO", scenario)
  set_all_params(result, d=d,...)
  if(is.null(useCells))
    useCells <- (train_size > 10000)
  if(useCells)
    setConfig(result, "PARTITION_CHOICE", 6)
  if(any(d>=1))
    str(get_config_line(result, 1))
  
  result
}

#' liquidSVM command line options
#' 
#' \strong{Should only be used by experts!}
#' liquidSVM command line tools \code{svm-train}, \code{svm-select}, and \code{svm-test}
#' can be used by more advanced users to get the most advanced use.
#' These three tools have command line arguments and those can be used from R as well.
#' 
#' @name command-args
#' @examples 
#' \dontrun{
#' reg <- liquidData('reg-1d')
#' model <- init.liquidSVM(Y~., reg$train)
#' trainSVMs(model, command.args=list(L=2, T=2, d=1))
#' selectSVMs(model, command.args=list(R=0,d=2))
#' result <- test(model, reg$test, command.args=list(T=1, d=0))
#' }
NULL


#' Trains an SVM object.
#' 
#' \strong{Should only be used by experts!}
#' This uses the \pkg{liquidSVM} C++ implementation to solve all SVMs on the hyper-parameter grid.
#' 
#' SVMs are solved for all tasks/cells/folds and entries in the hyper-parameter grid
#' and can afterwards be selected using \code{\link{selectSVMs}}.
#' A model even can be retrained using other parameters, reusing the training data.
#' The training phase is usually the most time-consuming phase,
#' and therefore for bigger problems it is recommended to use \code{display=1}
#' to get some progress information.
#' 
#' See \link{command-args} for details.
#' @template template-parameters-svm-train
#' @param model the \code{svm}-model
#' @param ... configuration parameters set before training
#' @param solver solver to use: one of "kernel.rule","ls","hinge","quantile","expectile"
#' @param do.select if not \code{FALSE} then the model is selected.
#'   This parameter can be used as a list of named arguments to be passed to the select phase
#' @param command.args further arguments aranged in a list, corresponding to the arguments
#' of the command line interface to \code{svm-train}, e.g. \code{list(d=2,W=2)}
#' is equivalent to \code{svm-train -d 2 -W 2}.
#' See \link{command-args} for details.
#' @inheritParams init.liquidSVM
#' @return a table giving training and validation errors and more internal statistic
#' for every SVM that was trained.
#' This is also recorded in \code{model$train_errors}.
#' @export
#' @aliases train
#' @seealso \link{command-args}, \code{\link{svm}}, \code{\link{init.liquidSVM}}, \code{\link{selectSVMs}}, \code{\link{predict.liquidSVM}}, \code{\link{test.liquidSVM}} and \code{\link{clean.liquidSVM}}
trainSVMs <- function(model, ... , solver=c("kernel.rule","ls","hinge","quantile"), command.args=NULL,
                      do.select=FALSE,useCells=FALSE, d=NULL
                        # , threads=1, gpus=0, d=0,folds=5,clip=NULL
                      ){
  
  if(!missing(solver)){
    if(!is.numeric(solver)){
      solver <- match.arg(solver)
      solver <- switch(solver, kernel.rule=0, ls=1, hinge=2, quantile=3)
    }
    setConfig(model, "SVM_TYPE", solver)
  }

  set_all_params(model, d=d, ...)

  if(length(model$levels)==2)
    setConfig(model, "WS_TYPE", 0)

  args <- list()
  
  if(is.null(useCells))
    useCells <- (nrow(model$train_data) > 10000)
  if(useCells)
    setConfig(model, "PARTITION_CHOICE", 6)
  
  model$solver <- as.integer(getConfig(model, "SVM_TYPE"))
  
  #cat(get_config_line(model, 1),fill=T)
  
  args <- makeArgs(command.args=command.args, default=args, defaultLine=get_config_line(model, 1))
  #str(args)
  args <- c('liquidSVM-train',
    # '-S',solver,
    # '-T',threads,'-GPU',gpus,'-d',d,
    args)
  
  if(any(d>=2))
    cat(args,fill = T)
  
  ret <- .Call('liquid_svm_R_train',
            as.integer(model$cookie), as.character(args) , PACKAGE='liquidSVM')
  stopifnot( is.numeric(ret) & !is.null(ret))
  
  if(!is.null(ret)){
    errors_labels <- c("task", "cell", "fold", "gamma", "pos_weight", "lambda", "train_error",
      "val_error", "init_iterations", "train_iterations", "val_iterations",
      "gradient_updates", "SVs")
    train_errors <- as.data.frame(matrix(ret, ncol=length(errors_labels), byrow=T))
    colnames(train_errors) <- errors_labels
    model$train_errors <- train_errors
    model$select_errors <- train_errors[NULL,]
    model$gammas <- sort(unique(train_errors$gamma))
    model$lambdas <- sort(unique(train_errors$lambda))
    model$trained <- T
  }else{
    model$train_errors <- NULL
    model$trained <- F
  }
  if(do.select != FALSE){
    if(is.logical(do.select) && do.select == TRUE){
      do.select <- NULL
    }
    selectArgs <- c(list(model, d=d),do.select)
    do.call(selectSVMs,selectArgs)
  }
  invisible(model)
}

#' Selects the best hyper-parameters of all the trained SVMs.
#' 
#' \strong{Should only be used by experts!}
#' This selects for every task and cell the best hyper-parameter based on the
#' validation errors in the folds. This is saved and will afterwards be used
#' in the evaluation of the decision functions.
#' 
#' Some learning scenarios have to perform several selection runs:
#' for instance in quantile regression for every quantile.
#' This is done by specifying \code{weight_number} ranging from 1 to the number of quantiles.
#' 
#' See \link{command-args} for details.
#' @template template-parameters-svm-select
#' @param model the \code{svm}-model
#' @param ... parameters passed to selection phase e.g. \code{retrain_method="select_on_entire_train_set"}
#' @param command.args further arguments aranged in a list, corresponding to the arguments
#' of the command line interface to \code{svm-select}, e.g. \code{list(d=2,R=0)}
#' is equivalent to \code{svm-select -d 2 -R 0}.
#' See \link{command-args} for details.
#' @inheritParams init.liquidSVM
#' @param warn.suboptimal if TRUE this will issue a warning
#' if the boundary of the hyper-parameter grid was hit too many times.
#' The default can be changed by setting \code{options(liquidSVM.warn.suboptimal=FALSE)}.
#' @return a table giving training and validation errors and more internal statistic
#' for all the SVMs that were selected.
#' This is also recorded in \code{model$select_errors}.
#' @export
#' @aliases select
#' @seealso \link{command-args}, \code{\link{svm}}, \code{\link{init.liquidSVM}}, \code{\link{selectSVMs}}, \code{\link{predict.liquidSVM}}, \code{\link{test.liquidSVM}} and \code{\link{clean.liquidSVM}}
selectSVMs <- function(model, command.args=NULL, ..., d=NULL, warn.suboptimal=getOption('liquidSVM.warn.suboptimal',TRUE)){
  
  if(!(model$trained)){
    stop("Model has not been trained yet")
    return(model)
  }
  
  set_all_params(model, d=d, ...)
  args <- makeArgs(command.args=command.args, defaultLine=get_config_line(model, 2))
  
  args <- c('liquidSVM-select',
    # '-S',solver,'-d',d,
    args)
  
  if(any(d>=2))
    cat(args,fill = T)
  
  ret <- .Call('liquid_svm_R_select',
            as.integer(model$cookie), args , PACKAGE='liquidSVM')
  # FIXME can only use this once we figure out how to pass selected train_val_info
  #
  if(length(ret)==1)
    stopifnot( !is.na(ret))

  if(!is.null(ret)){
    errors_labels <- c("task", "cell", "fold", "gamma", "pos_weight", "lambda", "train_error",
                       "val_error", "init_iterations", "train_iterations", "val_iterations",
                       "gradient_updates", "SVs")
    select_errors <- as.data.frame(matrix(ret, ncol=length(errors_labels), byrow=T))
    colnames(select_errors) <- errors_labels
    model$select_errors <- rbind(model$select_errors,select_errors)
    model$selected <- T
  }else{
    model$select_errors <- NULL
    model$selected <- F
  }
  
  if(all(warn.suboptimal)){
    ### give suggestions for boundary value hits
    #folds <- length(unique(model$select_errors$fold))
    alpha <- 1/2 #folds
    enlargeFactor <- 5
  
    msg <- "Solution may not be optimal: try training again using "
    
    if(mean(min(model$lambdas)==model$select_errors$lambda) > alpha)
      warning(msg, "min_lambda=",as.numeric(getConfig(model,"MIN_LAMBDA"))/enlargeFactor)
    
    if(mean(min(model$gammas)==model$select_errors$gamma) > alpha)
      warning(msg, "min_gamma=",as.numeric(getConfig(model,"MIN_GAMMA"))/enlargeFactor)
    
    if(mean(max(model$gammas)==model$select_errors$gamma) > alpha
       | mean(max(model$lambdas)==model$select_errors$lambda) > alpha)
      warning(msg, "max_gamma=",as.numeric(getConfig(model,"MAX_GAMMA"))*enlargeFactor)
  }
  
  invisible(model)
}




#' Predicts labels of new data using the selected SVM.
#' 
#' After training and selection the SVM provides means to compute predictions
#' for new input features.
#' If you have also labels consider using \code{\link{test.liquidSVM}}.
#' 
#' In the multi-result learning scenarios this returns all the predictions
#' corresponding to the different quantiles, expectiles, etc.
#' For multi-class classification, if the model was setup with \code{predict.prob=TRUE}
#' Then this will return only the probability columns and not the prediction.
#' 
#' 
#' @param object the SVM model as returned by \code{\link{init.liquidSVM}}
#' @param newdata data frame of features to predict.
#' If it has all the explanatory variables of \code{formula}, then the respective subset is taken.
#' @param ... other parameters passed to \code{\link{test.liquidSVM}}
#' @return the predicted values of test
#' @export
#' @aliases predict
#' @seealso \code{\link{init.liquidSVM}} and \code{\link{test.liquidSVM}}
#' @examples
#' ## Multiclass classification
#' modelIris <- svm(Species ~ ., iris)
#' y <- predict(modelIris, iris)
#' 
#' ## Least Squares
#' modelTrees <- svm(Height ~ Girth + Volume, trees)
#' y <- predict(modelTrees, trees)
#' plot(trees$Height, y)
predict.liquidSVM <- function(object, newdata, ...){
  result <- test(model=object,newdata=newdata,labels=0,...)
  if(!is.null(dim(result)))
    return(result[,object$predict.cols])
  else
    return(result)
}

#' Tests new data using the selected SVM.
#' 
#' After training and selection the SVM provides means to evaluate labels
#' for new input features.
#' If you do not have labels consider using \code{\link{predict.liquidSVM}}.
#' The errors for all tasks and cells are returned attached to the result (see \code{\link{errors}}).
#' 
#' If the SVM has multiple tasks the result will have corresponding columns.
#' For \code{\link{mcSVM}} the first column gives the global vote
#' and the other columns give the result for the corresponding binary classification problem
#' indicated by the column name.
#' 
#' For convenience the latest result is always saved in \code{model$last_result}.
#' 
#' @param model the SVM model as returned by \code{\link{init.liquidSVM}}
#' @param newdata data frame of features to predict.
#' If it has all the explanatory variables of \code{formula}, then the respective subset is taken.
#' NAs will be removed.
#' @param labels the known labels to test against. If NULL then they are retrieved from newdata using the original formula.
#' @param ... other configuration parameters passed to testing phase
#' @param command.args further arguments aranged in a list, corresponding to the arguments
#' of the command line interface to \code{svm-select}, e.g. \code{list(d=2,R=0)}
#' is equivalent to \code{svm-select -d 2 -R 0}.
#' See \link{command-args} for details.
#' @inheritParams init.liquidSVM
#' @return predictions for all tasks together with errors (see \code{\link{errors}}).
#' This is also recorded in \code{model$last_result}.
#' @inheritParams init.liquidSVM
#' @template template-parameters-svm-test
#' @export
#' @aliases test
#' @seealso \link{command-args}, \code{\link{init.liquidSVM}}, \code{\link{errors}}
#' @examples
#' modelTrees <- svm(Height ~ Girth + Volume, trees[1:10, ])  # least squares
#' result <- test(modelTrees, trees[11:31, ], trees$Height[11:31])
#' errors(result)
test.liquidSVM <- function(model, newdata, labels=NULL, command.args=NULL, ..., d=NULL#, threads=1, gpus=0
                     ){
  if(!(model$selected)){
    stop("Model has not been selected yet")
    return(model)
  }
  
  if(is.character(newdata)){
    newdata <- liquidData(newdata)
  }
  if(inherits(newdata,"liquidData")){
    newdata <- newdata$train
  }

  set_all_params(model, d=d, ...)
  args <- makeArgs(command.args=command.args, defaultLine=get_config_line(model, 3))
  args <- c("liquidSVM-test", # '-T',threads,'-GPU',gpus, '-d',d,
            args)
  
  stopifnot(!is.null(newdata))
  
  if(is.null(dim(newdata))){
    if(any(is.na(newdata)))
      stop("Test data has NA values: remove these using complete.cases(data) and rerun test")
    newdata <- matrix(newdata, ncol=model$dim,byrow=TRUE)
  }
  newdata_size <- as.integer(nrow(newdata))
  
  if(length(model$formula)>0){
    if(all(model$all_vars %in% colnames(newdata))){
      mf <- model.frame(model$formula, newdata, na.action=na.pass)
      newdata <- as.data.frame(mf[,-1])
      if(is.null(labels)){
        #cat("---retrieving labels from newdata\n")
        labels <- mf[[1]]
      }
    }else if(is.data.frame(newdata) && all(model$explanatory %in% colnames(newdata))){
      newdata <- newdata[,model$explanatory]
    }
  }else{
    newdata_dim <- dim(newdata)
    if(ncol(newdata) != model$dim){
      stop("newdata has not correct number of columns for this model.")
    }
  }

  newdata <- data.matrix(newdata)
  
  has_labels <- (length(labels) == newdata_size)
  if(!has_labels){
    #cat("setting all", length(labels), "labels to 0\n")
    labels <- numeric(newdata_size)
    labels <- NULL
  }else if(is.factor(labels)){
    labels <- factorToNumeric(labels, known_levels=model$levels)
  }else
    labels <- as.numeric(labels)
  
  val_index <- stats::complete.cases(newdata, labels)
  newdata <- newdata[val_index,]
  labels <- labels[val_index]

  if(any(d>=2)){
    cat(args,fill = T)
    str(as.numeric(t(newdata)))
    str(as.numeric(labels))
  }
  # we do not have to transpose newdata since __LIQUIDSVM_DATA_BY_COLS = true
  ret <- .Call('liquid_svm_R_test', as.integer(model$cookie), as.character(args),
            as.integer(newdata_size), as.numeric(newdata), as.numeric(labels))

  stopifnot( is.numeric(ret) & !is.null(ret))
  
  task_labels <- NULL
  if(!is.null(model$levels)){
    task_labels <- "result"
    if(getConfig(model, "WS_TYPE")==1){ ## AvA
      for(i in 1:(length(model$levels)-1))
        for(j in (i+1):length(model$levels))
          task_labels <- c(task_labels, paste0(model$levels[i], "vs", model$levels[j]))
    }else if(getConfig(model, "WS_TYPE")==2){  #OvA
      task_labels <- c(task_labels, paste0(model$levels, "vsOthers"))
    }else if(getConfig(model, "WS_TYPE")==0){  # Two-class
    }
  }
  
  error_ret <- attr(ret, 'error_ret');
  dd <- length(ret) / newdata_size
  if(dd==1){
    result <- numeric(newdata_size)
    result[val_index] <- ret
    result[!val_index] <- NA
    if( model$solver == SOLVER$HINGE && !is.null(model$levels)){
      result <- numericToFactor(result, known_levels=model$levels)
    }
    if(model$predict.prob){
      if(min(result) < -1 || max(result) > 1)
        warning('binary classification was done for labels outside -1...1?')
      else if(min(result) >= 0)
        warning('All probabilites are > 0.5, maybe the binary classification was not done for labels -1 and 1?')
      result <- (cbind(-result,result)+1) / 2
      colnames(result) <- if(is.null(model$levels))c(1,-1)else model$levels
      model$predict.cols <- 1:2
    }
  }else{
    #str(ret)
    ret <- matrix(ret, ncol=dd, byrow=T)
    result <- matrix(NA, nrow=newdata_size, ncol=dd)
    result[val_index, ] <- ret
    #if( model$solver == SOLVER$HINGE & !is.null(model$levels)){
    if( !is.null(model$levels)){
      
      raw <- lapply(1:dd, function(i)numericToFactor(result[,i],known_levels=model$levels))
      #str(raw)
      result <- do.call(data.frame,raw)
      colnames(result) <- task_labels # 1:dd
      attributes(result)$task_labels <- task_labels
    }
    if(model$predict.prob){
      result[,model$predict.cols] <- (result[,model$predict.cols]+1) / 2
    }
  }
  model$last_result <- as.data.frame(result)
  if(has_labels){
    err <- matrix(error_ret,ncol = 3,byrow=TRUE,dimnames = list(NULL,c("val_error","pos_val_error","neg_val_error")))
    if(!is.null(model$levels) && length(task_labels) == nrow(err)){
      rownames(err) <- task_labels
    }
    attributes(result)$errors <- err
    attributes(model$last_result)$errors <- err
  }
  result
}

#' Obtain the test errors result.
#' 
#' After calculating the result in \code{\link{test.liquidSVM}} if labels
#' were given \pkg{liquidSVM} also calculates the test error.
#' 
#' Depending on the learning scenario there can be multiple errors: usually there is one  per task,
#' and \code{\link{mcSVM}} adds in front the global classification error.
#' In the latter case the names give an information for what task the error was computed.
#' 
#' For each error also the positive and negative validation error can be shonw using \code{showall}
#' for example in \code{\link{rocSVM}}.
#' 
#' @param y the results of \code{\link{test.liquidSVM}}
#' @param showall show the more detailed errors as well.
#' @return for all tasks the global and optionally also the positive/negative errors.
#' Depending on the learning scenario there can be also a overall error (e.g. in multi-class classification).
#' @export
#' @seealso \code{\link{test.liquidSVM}}
#' @examples
#' modelTrees <- svm(Height ~ Girth + Volume, trees[1:10, ])  # least squares
#' 
#' y <- test(modelTrees,trees[-1:-10,])
#' errors(y)
#' 
#' \dontrun{
#' banana <- liquidData('banana-bc')
#' s_banana <- rocSVM(Y~., banana$test)
#' result <- test(s_banana, banana$train)
#' errors(result, showall=TRUE)
#' }
errors <- function(y, showall=FALSE){
  return(if(showall)attributes(y)$errors else attributes(y)$errors[,1])
}

noop <- function(x){}

#' Force to release the internal memory of the C++ objects
#' associated to this model.
#' 
#' Usually this has not to be done by the user since
#' liquidSVM harnesses garbage collection offered by R.
#' 
#' @param model the SVM model as returned by \code{\link{init.liquidSVM}}
#' @param warn if \code{TRUE} issue warning if the model already was deleted
#' @param ... not used at the moment
#' @export
#' @aliases clean
#' @seealso \code{\link{init.liquidSVM}}
#' @examples
#' ## Multiclass classification
#' modelIris <- svm(Species ~ ., iris)
#' y <- predict(modelIris, iris)
#' 
#' ## Least Squares
#' modelTrees <- svm(Height ~ Girth + Volume, trees)
#' y <- predict(modelTrees, trees)
#' plot(trees$Height, y)
#' test(modelTrees, trees)
#' 
#' clean(modelTrees)
#' clean(modelIris)
#' # now predict(modelTrees, ...) would not be possible any more
clean.liquidSVM <- function(model, warn=TRUE, ...){
  if(length(model$cookie)!=1){
    warning(paste0("Weird cookie: ",model$cookie))
    return(invisible(NULL))
  }
  if(model$cookie < 0){
    if(warn) warning("already deleted")
    return(invisible(NULL))
  }
  .Call('liquid_svm_R_clean',as.integer(model$cookie))
  model$deleted <- TRUE
  model$trained <- FALSE
  model$cookie <- -model$cookie
  invisible(NULL)
}

#' Set display info mode that controls how much information is displayed
#' by liquidSVM C++ routines. Usually you will use \code{display=d} in \code{svm(...)} etc.
#' @param d the display information
setDisplay <-function(d=1){
  .Call('liquid_svm_R_set_info_mode', as.integer(d), PACKAGE='liquidSVM')
  invisible(NULL)
}

# Converts function arguments to command line arguments.
# @param ... the arguments to convert
# @param command.args further arguments aranged in a list, e.g. \code{list(d=2,T=-1,GPUs=1)}
# @param default some default arguments as list which are overriden by \code{...}
# @param defaultLine default arguments as vector which are overriden by \code{...}
makeArgs<-function(...,command.args=NULL,default=NULL, defaultLine=NULL){
  a <- list(...)
  a <- c(a,command.args)
  if(length(a)>0){
  b <- names(a)
  f <- function(i){
    #str(list(i=i,b=b,a=a))
    if(nchar(b[i])==0)
      return(a[[i]])
    else
     return( c(paste0("-",b[i]),a[[i]]))
  }
  l <- c(lapply(1:length(a),f),recursive=T)
  }else{
    l <- NULL
    b <- NULL
  }
  if(!is.null(default)){
    for(name in names(default)){
      if(!(name %in% b)){
        #cat("Trying name",name,fill = T)
        l <- c(l, paste0("-",name), default[[name]])
      }
    }
  }
  return(c(strsplit(defaultLine, " ")[[1]], l))
}

factorToNumeric <- function(f,known_levels=levels(f)){
  stopifnot(is.factor(f))
  numlevels <- suppressWarnings(as.numeric(known_levels))
  if(any(is.na(numlevels))){
    if(!all(levels(f) ==known_levels))
      warning("Test data might have different labelling!")
    return(as.integer(f))
  }else{
    stopifnot(!any(is.na(as.numeric(levels(f)))))
    return(numlevels[as.integer(f)])
  }
}

numericToFactor <- function(x, known_levels){
  numlevels <- suppressWarnings(as.numeric(known_levels))
  if(any(is.na(numlevels))){
    ### So the levels do have non-numerics, hence we did use integer labels before
    if(all(x%in%1:length(known_levels) | is.na(x)))
      return(factor(known_levels[as.integer(x)], levels=known_levels))
    else
      return(x)
  }else{
    ### all levels are numerics so we did send the values before:
    if(all(x %in% known_levels | is.na(x)))
      return(factor(x, levels=known_levels))
    else
      return(x)
  }
}

#' Compilation information: whether the library was compiled using SSE2 or even AVX.
#' @return character with the information.
#' @export
compilationInfo <- function(){
  ret <- .Call('liquid_svm_R_default_params',
               as.integer(-1L), as.integer(0L) , PACKAGE='liquidSVM')
  ret
}

#' @export
test <- function(...){UseMethod("test")}
#' @export
clean <- function(model, ...){UseMethod("clean")}


SOLVER <- list(KERNEL=0, LS=1, HINGE=2, QUANTILE=3)


### To make quick tests the following might be usefull

# doit_iris <- function(formula=Species~.,data=iris,...){
#   doit(formula=formula,data=data,...)
# }
# 
# doit_trees <- function(formula=Height~.,data=trees,...){
#   doit(formula=formula,data=data,...)
# }

doit_sml <- function(formula=Y~.,data=liquidData(sml)$train,sml='covtype.1000',...){
  doit(formula=formula,data=data,...)
}

doit <- function(d=1,threads=1, formula=Y ~ ., data=liquidData("covtype.1000")$train, ...){
  invisible(svm(formula,data, d=d, threads=threads, ...))
}

