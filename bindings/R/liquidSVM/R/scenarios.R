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


# Helper function to automagically test a newly trained model if testdata is available
return_with_test <- function(model, formula, data, ..., testdata=NULL, testdata_labels=NULL,d=NULL){
  if(missing(data)){
    data <- parent.frame()
  }
  if(is.character(data))
    data <- liquidData(data)
  if(is.null(testdata) & inherits(data, "liquidData")){
    testdata <- data$test
  }
  if(!is.null(testdata)){
    test(model, testdata, labels=testdata_labels, d=d)
  }
  return(invisible(model))
}


#' Multiclass Classification
#' 
#' This routine is intended for both binary and multiclass 
#' classification. The binary classification is treated by
#' an SVM solver for the classical hinge loss, and for the
#' multiclass case, one-verus-all and all-versus-all reductions 
#' to binary classification for the hinge and the least 
#' squares loss are provided. The error of the very first
#' task is the overall classification error.
#' \code{svmMulticlass} is a simple alias of \code{mcSVM}
#' 
#' Please look at the demo-vignette (\code{vignette('demo')}) for more examples.
#' 
#' \code{mcSVM} is best used with \code{factor}-labels. If there are just two levels in the factor,
#' or just two unique values if it is \code{numeric} than a binary classification is performed.
#' Else, by using the parameter \code{mc_type} different combinations of all-vs-all (\code{AvA})
#' and one-vs-all (\code{OvA}) and hinge (\code{hinge}) and least squares loss (\code{ls}) can be used.
#' 
#' If a test is performed then not only the final decision is returned but also
#' the results of the intermediate binary classifications. This is indicated in the column names.
#' If the training labels are given by a \code{factor} then the final decision will be encoded
#' in this factor. If this is the case and \code{AvA_hinge} is used,
#' then also the binary classification problems will receive the corresponding label...
#' 
#' @param predict.prob If \code{TRUE} then a LS-svm will be trained and
#'   the conditional probabilities for the binary classification problems will be estimated.
#'   This also restricts the choices of \code{mc_type} to \code{c("OvA_ls","AvA_ls")}.
#' @param mc_type configures the 
#'   the multiclass variants for All-vs-All / One-vs-All and with hinge or least squares loss.
#' @inheritParams lsSVM
#' @return an object of type \code{svm}. Depending on the usage this object
#' has also \code{$train_errors}, \code{$select_errors}, and \code{$last_result}
#' properties.
#' @export
#' @seealso \code{\link{Configuration}}
#' @examples 
#' model <- mcSVM(Species ~ ., iris)
#' model <- mcSVM(Species ~ ., iris, mc_type="OvA")
#' model <- mcSVM(Species ~ ., iris, mc.type="AvA_hi")
#' model <- mcSVM(Species ~ ., iris, predict.prob=TRUE)
#' 
#' \dontrun{
#' ## a worked example can be seen at
#' 
#' vignette("demo",package="liquidSVM")
#' }
#' 
mcSVM <- function(x,y,..., predict.prob=FALSE,mc_type=c("AvA_hinge","OvA_ls","OvA_hinge","AvA_ls"),do.select=TRUE){
  if(predict.prob){
    if(mc_type[1] %in% c("AvA_hinge","OvA_hinge"))
      mc_type <- 'OvA_ls'
  }
  model <- init.liquidSVM(x,y,..., scenario=c("MC",mc_type))
  model$predict.prob <- predict.prob
  if(!predict.prob){
    model$predict.cols <- 1L
  }else{
    model$predict.cols <- -1L
  }
  trainSVMs(model)
  if(do.select)
    selectSVMs(model)
  return_with_test(model, x, y, ...)
}

#' @rdname mcSVM
#' @export
svmMulticlass <- mcSVM

#' Neyman-Pearson-Learning
#' 
#' This routine provides binary classifiers that satisfy a 
#' predefined error rate on one type of error and that 
#' simlutaneously minimize the other type of error. For 
#' convenience some points on the ROC curve around the 
#' predefined error rate are returned.
#' \code{nplNPL} performs Neyman-Pearson-Learning for classification.
#' 
#' Please look at the demo-vignette (\code{vignette('demo')}) for more examples.
#' The labels should only have value \code{c(1,-1)}.
#' 
#' \code{min_weight}, \code{max_weight}, \code{weight_steps}: you might have to define
#' which weighted classification problems will be considered.
#' The choice is usually a bit tricky. Good luck ...
#' 
#' @param class is the normal class (the other class becomes the alarm class)
#' @param constraint gives the false alarm rate which should be achieved
#' @param constraint.factors specifies the factors around \code{constraint}
#' @inheritParams lsSVM
#' @return an object of type \code{svm}. Depending on the usage this object
#' has also \code{$train_errors}, \code{$select_errors}, and \code{$last_result}
#' properties.
#' @export
#' @examples 
#' \dontrun{
#' model <- nplSVM(Y ~ ., 'banana-bc', display=1)
#' 
#' ## a worked example can be seen at
#' vignette("demo",package="liquidSVM")
#' }
#'  
nplSVM <- function(x,y,..., class=1, constraint=0.05, constraint.factors=c(3,4,6,9,12)/6,do.select=TRUE){
  model <- init.liquidSVM(x,y,..., scenario=c("NPL", class))
  model$predict.cols <- 1:length(constraint.factors)
  trainSVMs(model)
  if(do.select)
    for(cf in constraint*constraint.factors)
      selectSVMs(model,npl_class=class, npl_constraint=cf)
  return_with_test(model, x,y,...)
}



#' Receiver Operating Characteristic curve (ROC curve)
#' 
#' This routine provides several points on the ROC curve by
#' solving multiple weighted binary classification problems.
#' It is only suitable to binary classification data.
#' 
#' Please look at the demo-vignette (\code{vignette('demo')}) for more examples.
#' The labels should only have value \code{c(1,-1)}.
#' 
#' \code{min_weight}, \code{max_weight}, \code{weight_steps}: you might have to define
#' which weighted classification problems will be considered.
#' The choice is usually a bit tricky. Good luck ...
#' 
#' @param weight_steps indicates how many weights between \code{min_weight} and
#'   \code{max_weight} will be used
#' @inheritParams lsSVM
#' @return an object of type \code{svm}. Depending on the usage this object
#' has also \code{$train_errors}, \code{$select_errors}, and \code{$last_result}
#' properties.
#' @export
#' @seealso \code{\link{plotROC}}
#' @examples 
#' \dontrun{
#' banana <- liquidData('banana-bc')
#' model <- rocSVM(Y ~ ., banana$train, display=1)
#' plotROC(model,banana$test)
#' 
#' ## a worked example can be seen at
#' vignette("demo",package="liquidSVM")
#' }
#'  
rocSVM <- function(x,y,...,weight_steps=9,do.select=TRUE){
  model <- init.liquidSVM(x,y,..., scenario="ROC",weight_steps=weight_steps)
  model$predict.cols <- 1:weight_steps
  trainSVMs(model)
  if(do.select)
    for(i in 1:weight_steps)
      selectSVMs(model,weight_number=i)
  return_with_test(model, x,y,...)
}

#' Least Squares Regression
#' 
#' This routine performs non-parametric least squares regression
#' using SVMs. The tested estimators are therefore estimating 
#' the conditional means of Y given X.
#' \code{svmRegression} is a simple alias of \code{lsSVM}.
#' 
#' This is the default for \code{\link{svm}} if the labels are not a factor.
#' 
#' @param x either a formula or the features
#' @param y either the data or the labels corresponding to the features \code{x}.
#'   It can be a \code{character} in which case the data is loaded using \code{\link{liquidData}}.
#'   If it is of type \code{liquidData} then after \code{train}ing and \code{select}ion
#'   the model is \code{\link{test}}ed using the testing data (\code{y$test}).
#' @param ... configuration parameters, see \link{Configuration}. Can be \code{threads=2, display=1, gpus=1,} etc.
#' @param do.select if \code{TRUE} also does the whole selection for this model
#' @param clipping absolute value where the estimated labels will be clipped. -1 (the default)
#'   leads to an adaptive clipping value, whereas 0 disables clipping.
#' @return an object of type \code{svm}. Depending on the usage this object
#' has also \code{$train_errors}, \code{$select_errors}, and \code{$last_result}
#' properties.
#' @export
#' @examples 
#' tt <- ttsplit(quakes)
#' model <- lsSVM(mag~., tt$train, display=1)
#' result <- test(model, tt$test)
#' 
#' errors(result) ## is the same as
#' mean( (tt$test$mag-result)^2 )
#'  
lsSVM <- function(x,y,...,clipping=-1.0,do.select=TRUE){
  model <- init.liquidSVM(x,y,..., scenario=c("LS", clipping))
  trainSVMs(model)
  if(do.select)
    selectSVMs(model)
  return_with_test(model, x,y,...)
}

#' @rdname lsSVM
#' @export
svmRegression <- lsSVM

#' Quantile Regression
#' 
#' This routine performs non-parametric and quantile regression using SVMs. 
#' The tested estimators are therefore estimating the conditional tau-quantiles 
#' of Y given X. By default, estimators for five different tau values 
#' are computed.
#' \code{svmQuantileRegression} is a simple alias of \code{qtSVM}.
#' 
#' @param weights the quantiles that should be estimated
#' @inheritParams lsSVM
#' @return an object of type \code{svm}. Depending on the usage this object
#' has also \code{$train_errors}, \code{$select_errors}, and \code{$last_result}
#' properties.
#' @export
#' @examples 
#' \dontrun{
#' tt <- ttsplit(quakes)
#' model <- qtSVM(mag~., tt$train, display=1)
#' result <- test(model, tt$test)
#' 
#' errors(result)[2] ## is the same as
#' mean(ifelse(result[,2]<tt$test$mag, -.1,.9) * (result[,2]-tt$test$mag))
#' }
qtSVM <- function(x,y,...,weights=c(.05,.1,.5,.9,.95),clipping=-1.0,do.select=TRUE){
  model <- init.liquidSVM(x,y,..., scenario=c("QT", clipping),weights=weights)
  model$predict.cols <- 1:length(weights)
  trainSVMs(model)
  if(do.select)
    for(i in 1:length(weights))
      selectSVMs(model,weight_number=i)
  return_with_test(model, x,y,...)
}

#' @rdname qtSVM
#' @export
svmQuantileRegression <- qtSVM

#' Expectile Regression
#' 
#' This routine performs non-parametric, asymmetric least squares 
#' regression using SVMs. The tested estimators are therefore estimating 
#' the conditional tau-expectiles of Y given X. By default, estimators
#' for five different tau values are computed.
#' \code{svmExpectileRegression} is a simple alias of \code{exSVM}.
#' 
#' @param weights the expectiles that should be estimated
#' @inheritParams lsSVM
#' @return an object of type \code{svm}. Depending on the usage this object
#' has also \code{$train_errors}, \code{$select_errors}, and \code{$last_result}
#' properties.
#' @export
#' @examples 
#' \dontrun{
#' tt <- ttsplit(quakes)
#' model <- exSVM(mag~., tt$train, display=1)
#' result <- test(model, tt$test)
#' 
#' errors(result)[2] ## is the same as
#' mean(ifelse(result[,2]<tt$test$mag, .1,.9) * (result[,2]-tt$test$mag)^2)
#' }
exSVM <- function(x,y,...,weights=c(.05,.1,.5,.9,.95),clipping=-1.0,do.select=TRUE){
  model <- init.liquidSVM(x,y,..., scenario=c("EX", clipping),weights=weights)
  model$predict.cols <- 1:length(weights)
  trainSVMs(model)
  if(do.select)
    for(i in 1:length(weights))
      selectSVMs(model,weight_number=i)
  return_with_test(model, x,y,...)
}

#' @rdname exSVM
#' @export
svmExpectileRegression <- exSVM


#' Bootstrap 
#' 
#' This routine performs bootstrap learning for all scenarios except multiclass classification.
#' 
#' @param solver the solver to use. Can be any of \code{KERNEL_RULE}, \code{SVM_LS_2D},
#' \code{SVM_HINGE_2D}, \code{SVM_QUANTILE}, \code{SVM_EXPECTILE_2D}
#' @param ws.number number of working sets to build and train
#' @param ws.size how many samples to draw from the training set for each working set
#' @return an object of type \code{svm}. Depending on the usage this object
#' has also \code{$train_errors}, \code{$select_errors}, and \code{$last_result}
#' properties.
#' @inheritParams lsSVM
#' @export
#' 
bsSVM <- function(x,y,..., solver, ws.number=5, ws.size=500,do.select=TRUE){
  model <- init.liquidSVM(x,y,..., scenario=c("BS", solver, ws.number, ws.size))
  trainSVMs(model)
  if(do.select)
    selectSVMs(model)
  return_with_test(model, x,y,...)
}

#' Plots the ROC curve for a result or model
#' 
#' This can be used either using \code{\link{rocSVM}} or \code{\link{lsSVM}}
#' 
#' @param x either the result from a \code{\link{test}} or a model
#' @param correct either the true values or testing data for the model
#' @param posValue the label marking the positive value.
#' If \code{NULL} (default) then the larger value.
#' @param xlim sets better defaults for \code{\link{plot.default}}
#' @param ylim sets better defaults for \code{\link{plot.default}}
#' @param asp sets better defaults for \code{\link{plot.default}}
#' @param type sets better defaults for \code{\link{plot.default}}
#' @param pch sets better defaults for \code{\link{plot.default}}
#' @param add if `FALSE` (default) produces a new plot and if `TRUE` adds to existing plot.
#' @param ... gets passed to \code{\link{plot.default}}
#' @seealso rocSVM, lsSVM
#' @export
#' @seealso \code{\link{rocSVM}}
#' @importFrom graphics abline plot points
#' @examples
#' \dontrun{
#' banana <- liquidData('banana-bc')
#' model <- rocSVM(Y~.,banana$train)
#' 
#' plotROC(model ,banana$test)
#' # or:
#' result <- test(model, banana$test)
#' plotROC(result, banana$test$Y)
#' 
#' model.ls <- lsSVM(Y~., banana$train)
#' result <- plotROC(model.ls, banana$test)
#' }
plotROC <- function(x, correct, posValue=NULL, xlim=0:1,ylim=0:1,asp=1, type=NULL, pch='x',add=FALSE, ...){
  if(inherits(x,"liquidSVM")){
    model <- x
    x <- test(model,correct)
    # FIXME find the labels using model
    correct <- correct[,1]
  }
  xOrig <- x
  if(is.factor(correct)){
    lev <- levels(correct)
  }else{
    lev <- unique(correct)
  }
  stopifnot(length(lev) %in% c(1,2))
  if(is.null(dim(x)) || all(apply(x,2,is.factor))){
    t <- seq(-1,1,by=.1)
    # or
    # t <- sort(unique(x))
    x <- do.call(cbind, lapply(t, function(tt) x>=tt))
    if(is.null(type)) type <- 'l'
  }
  if(is.null(posValue)) posValue <- max(lev)
  false_positive_rate <- apply(x[correct!=posValue,]==posValue,2,mean)
  detection_rate <- apply(x[correct==posValue,]==posValue,2,mean)
  
  if(add){
    points(false_positive_rate, detection_rate,xlim=xlim,ylim=ylim,asp=asp,type=type,pch=pch,...)
  }else{
    plot(false_positive_rate, detection_rate,xlim=xlim,ylim=ylim,asp=asp,type=type,pch=pch,...)
  }
  abline(0,1,lty=2)
  
  invisible(xOrig)
}
