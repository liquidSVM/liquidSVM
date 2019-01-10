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

# #' @importFrom methods setRefClass

#' A Reference Class to represent a liquidSVM model.
#'
#' @field cookie this is used in C++ to access the model in memory
#' @importFrom methods setRefClass new initRefFields
liquidSVMclass <- setRefClass("liquidSVM",
  fields = list(cookie = "integer", dim="integer", train_data='data.frame', train_labels='numeric',
                all_vars='character', explanatory='character', formula='formula', levels='ANY',
                solver='integer', gammas='numeric', lambdas='numeric',
                train_errors='data.frame', trained='logical',
                select_errors='data.frame', selected='logical',
                last_result='data.frame', deleted='logical',
                predict.prob='logical',predict.cols='integer',
                solution_aux_filename='character'),
  methods = list(
    initialize=function(...){
      trained <<- selected <<- deleted <<- FALSE
      predict.prob <<- FALSE
      predict.cols <<- 1L
      callSuper(...)
    },
    show=function(){
      print.liquidSVM(.self)
    },
    finalize = function(...){
     clean.liquidSVM(.self, warn=FALSE)
    }
  )
)

#' Printing an SVM model.
#' @param x the model to print
#' @param ... other arguments to print.default
#' @export
#' @examples
#' \dontrun{
#' s_iris <- svm(solver='hinge', Species ~ ., iris)  # multiclass classification
#' print(s_iris)
#' }
print.liquidSVM <- function(x,...){
  model <- x
  cat("SVM model")
  cat(" on",model$dim,"features")
  cat(" (cookie=",model$cookie,")",sep="")
  cat("\n")
  if(all(nchar(getConfig(model,"SCENARIO"))>0)){
    cat(" Scenario:",getConfig(model,"SCENARIO"),fill=T)
  }
  if(!is.null(model$formula)){
    cat(" Formula: ",deparse(model$formula), fill=T)
  }
  hyper <- paste0(length(model$gammas),"x",length(model$lambdas))
  if(x$selected){
    cat("  trained and selected on a",hyper,"grid")
  }else if(x$trained){
    cat("  trained on a",hyper,"grid; no solution selected yet")
  }else{
    cat("  not yet trained at all")
  }
  cat("\n")
  if(nrow(model$last_result)>0){
    cat("  has a $last_result because there has been predicting or testing\n")
  }
  if(length(model$solution_aux_filename)>0){
    cat("  solution was loaded from", solution_aux_filename,fill=T)
  }
  if(x$deleted){
    cat("    deleted, please forget me!",fill=T)
  }
}



#' liquidSVM model configuration parameters.
#' 
#' Different parameters configure different aspects of training/selection/testing.
#' The learning scenarios set many parameters to corresponding default values,
#' and these can again be changed by the user. Therefore the order in which they
#' are specified can be important.
#' 
#' @param model the model
#' @param name the name
#' @param value the value
#' @return the value of the configuration parameter
#' @name Configuration
#' @template global-and-grid.md
NULL

#' @export
#' @rdname Configuration
getConfig <- function(model,name){
  .Call('liquid_svm_R_get_param',as.integer(model$cookie),
        as.character(name),
        PACKAGE='liquidSVM')
}

#' @export
#' @rdname Configuration
setConfig <- function(model,name, value){
  #stopifnot(length(value)>0)
  if(length(value)==0) return()
  if(is.numeric(value)){
    value <- sapply(value,format, scientific=F)
  }
  if(is.logical(value)){
    value <- as.numeric(value)
  }
  if(length(value)>1)
    value <- paste(value,collapse=" ")
  name <- toupper(name)
  # cat("setConfig: ", name, "to", value, fill=T)
  result <- .Call('liquid_svm_R_set_param',as.integer(model$cookie),
                  as.character(name), as.character(value),
                  PACKAGE='liquidSVM')
  if(name=='D' || name=="DISPLAY"){
    setDisplay(value)
  }
}

# `[.liquidSVM` <- function(x, name, ..., drop=FALSE){
#   liquidSVM:::getConfig(x, toupper(name))
# }
# `$.liquidSVM` <- function(x, name, ..., drop=FALSE){
#   if(name %in% ls(model))
#     return(model[[name]])
#   liquidSVM:::getConfig(x, toupper(name))
# }
# `$<-.liquidSVM` <- function(x, name, value){
#   if(name %in% ls(model))
#     return(model[[name]]<-value)
#   liquidSVM:::setConfig(x, toupper(name), value)
# }
# `[<-.liquidSVM` <- function(x, name, value){
#   if(name %in% ls(model))
#     return(model[[name]]<-value)
#   liquidSVM:::setConfig(x, toupper(name), value)
# }

get_config_line <- function(model,stage){
  line <- .Call('liquid_svm_R_get_config_line',as.integer(model$cookie),
                as.integer(stage),
                PACKAGE='liquidSVM')
  #str(line)
  line <- paste(line, collapse="")
  #str(line)
  if(substr(line, 1,1)==" ")
    line <- substring(line, 2)
  line
}

# set_scenario <- function(model,scenario, param){
#   if(missing(param))
#     value <- scenario
#   else
#     value <- paste0(scenario, " ", param)
#   setConfig(model, "SCENARIO", value)
# }

set_all_params <- function(model,...){
  otherArgs <- list(...)
  for(name in names(otherArgs)){
    setConfig(model, toupper(name), otherArgs[[name]])
  }
}

#' Get Cover of partitioned SVM
#' 
#' If you use \code{voronoi=3} or \code{voronoi=4} this retrieves the voronoi centers that have been found.
#' 
#' @param model the model
#' @param task the task between 1 and number of tasks
#' @return the indices of the samples in the training data set that were used as Voronoi partition centers.
#' @export
#' @note This is not tested thoroughly so use in production is at your own risk.
#' @examples
#' \dontrun{
#' banana <- liquidData('banana-mc')
#' model <- mcSVM(Y~.,banana$train, voronoi=c(4,500),d=1)
#' # task 4 is predicting 2 vs 3
#' cover <- getCover(model,task=4)
#' centers <- cover$samples
#' # we are considering task 4 and hence only show labels 2 and 3:
#' bananaSub <- banana$train[banana$train$Y %in% c(2,3),]
#' plot(bananaSub[,-1],col=bananaSub$Y)
#' points(centers,pch='x',cex=2)
#' 
#' if(require(deldir)){
#'   voronoi <- deldir::deldir(centers$X1,centers$X2,rw=c(range(bananaSub$X1),range(bananaSub$X2)))
#'   plot(voronoi,wlines="tess",add=TRUE, lty=1)
#'   text(centers$X1,centers$X2,1:nrow(centers),pos=1)
#' }
#' 
#' # let us calculate for every sample in this task which cell it belongs to
#' distances <- as.matrix(dist(model$train_data))
#' cells <- apply(distances[model$train_labels %in% c(2,3),cover$indices],1,which.min)
#' # and you can check that the cell sizes are as reported in the training phase for task 4
#' table(cells)
#' }
getCover <- function(model, task=1){
  if(!(model$trained)){
    stop("Model has not been trained yet")
    return(model)
  }
  indices <- .Call('liquid_svm_R_get_cover',as.integer(model$cookie),
        as.integer(task),
        PACKAGE='liquidSVM')
  indices <- indices + 1
  list(indices=indices, samples=model$train_data[indices,], task=task)
}


#' Retrieve the solution of an SVM
#' 
#' Gives the solution of an SVM that has been trained and selected
#' in an ad-hoc list.
#' 
#' liquidSVM splits all problems into tasks (e.g. for multiclass classification
#' or if using multiple weights), then each task is split into cells (maybe only a global one),
#' and every cell then is trained in one or more folds to yiele a solution.
#' Hence these coordinates have to be specified.
#' 
#' @param model the model
#' @param task the task between 1 and number of tasks
#' @param cell the cell between 1 and number of cells
#' @param fold the fold between 1 and number of folds
#' @return a list with three entries: the offset of the solution (not yet implemented),
#' the indices of the support vectors in the training data set, and
#' the coefficients of the support vectors
#' @export
#' @note This is not tested thoroughly so use in production is at your own risk.
#' @examples
#' \dontrun{
#' # simple example: regression of sinus curve
#' x <- seq(0,1,by=.01)
#' y <- sin(x*10)
#' a <- lapply(1:5, function(i)getSolution(model <- lsSVM(x,y,d=1), 1,1,i))
#' plot(x,y,type='l',ylim=c(-5,5));
#' for(i in 1:5) lines(coeff~samples, data=a[[i]],col=i)
#' 
#' # a more typical example
#' banana <- liquidData('banana-mc')
#' model <- mcSVM(Y~.,banana$train,d=1)
#' # task 4 is predicting 2 vs 3, there is only cell 1 here
#' solution <- getSolution(model,task=4,cell=1,fold=1)
#' supportvecs <- solution$samples
#' # we are considering task 4 and hence only show labels 2 and 3:
#' bananaSub <- banana$train[banana$train$Y %in% c(2,3),]
#' plot(bananaSub[,-1],col=bananaSub$Y)
#' points(supportvecs,pch='x',cex=2)
#' }
getSolution <- function(model,task=1, cell=1, fold=1){
  if(!(model$selected)){
    stop("Model has not been selected yet")
    return(model)
  }
  ret <- .Call('liquid_svm_R_get_solution',as.integer(model$cookie),
               as.integer(task),as.integer(cell),as.integer(fold),
               PACKAGE='liquidSVM')
  ret$sv <- ret$sv + 1
  ret$samples <- if(is.null(dim(model$train_data))) model$train_data[ret$sv] else model$train_data[ret$sv,]
  ret$labels <- model$train_labels[ret$sv]
  ret$task <- task
  ret$cell <- cell
  ret$fold <- fold
  ret
}

#' Read and Write Solution from and to File
#' 
#' Reads or writes the solution from or to a file.
#' The format of the solutions is the same as used in the command line version of liquidSVM.
#' In addition also configuration data is written and by default also the training data.
#' This can be interchanged also with the other bindings.
#' 
#' The command line version of liquidSVM saves solutions
#' after select in files of the name \emph{data}\code{.sol} or \emph{data}\code{.fsol} and
#' uses those in the test-phase.
#' \code{read.liquidSVM} and \code{write.liquidSVM} read and write the same format at the specified path.
#' If you give a \code{filename} using extension \code{.fsol} the training data
#' is written to the file and read from it. On the other hand,
#' if you use the \code{.sol} format, you need to be able to reproduce
#' the same data again once you read the solution.
#' \code{readSolution} creates a new svm object.
#' 
#' @param model the model
#' @param filename the filename to read from/save to. Can be relative to the working directory.
#' @param writeData whether the training data should be serialized in the stream
#' @param obj the data to unserialize
#' @param ... passed to \code{\link{init.liquidSVM}}
#' @inheritParams init.liquidSVM
#' @export
#' @note This is not tested thoroughly so use in production is at your own risk.
#' Furthermore the serialize/unserialize hooks write temporary files.
#' @seealso \code{\link{init.liquidSVM}}, \code{\link{write.liquidSVM}}
#' @examples
#' \dontrun{
#' banana <- liquidData('banana-bc')
#' modelOrig <- mcSVM(Y~., banana$train)
#' write.liquidSVM(modelOrig, "banana-bc.fsol")
#' write.liquidSVM(modelOrig, "banana-bc.sol")
#' clean(modelOrig) # delete the SVM object
#' 
#' # now we read it back from the file
#' modelRead <- read.liquidSVM("banana-bc.fsol")
#' # No need to train/select the data!
#' errors(test(modelRead, banana$test))
#' 
#' # to read the model where no data was saved we have to make sure, we get the same training data:
#' banana <- liquidData('banana-bc')
#' # then we can read it
#' modelDataExternal <- read.liquidSVM("banana-bc.sol", Y~., banana$train)
#' result <- test(modelDataExternal, banana$test)
#' 
#' # to serialize an object use:
#' banana <- liquidData('banana-bc')
#' modelOrig <- mcSVM(Y~., banana$train)
#' # we serialize it into a raw vector
#' obj <- serialize.liquidSVM(modelOrig)
#' clean(modelOrig) # delete the SVM object
#' 
#' # now we unserialize it from that raw vector
#' modelUnserialized <- unserialize.liquidSVM(obj)
#' errors(test(modelUnserialized, banana$test))
#' }
read.liquidSVM <- function(filename, ...){
  if(!( grepl("\\.fsol$",filename) || grepl("\\.sol$",filename) )){
    stop("Filename must have extension .fsol or .sol but is: ",filename)
    return(invisible(NULL))
  }
  stopifnot(length(filename)==1)
  if(!( file.access(filename,mode=4)==0 )){
    stop("Filename is not readable: ", filename)
    return(invisible(NULL))
  }
  if(length(list(...)) != 0){
    model <- init.liquidSVM(...)
  }else{
    model <- liquidSVMclass$new(cookie=-1L, solver=-1L)
  }
  ret <- .Call('liquid_svm_R_read_solution',as.integer(model$cookie),
        as.character(filename),
        PACKAGE='liquidSVM')
  further <- unserialize(ret[[4]])
  for(i in names(further)){
    assign(i, further[[i]], envir=model)
  }
  model$cookie <- ret[[1]]
  model$dim <- ret[[2]]
  #model$size <- ret[[3]]
  model$solution_aux_filename <- filename
  model$trained <- T
  model$selected <- T
  model
}


#' @export
#' @rdname read.liquidSVM
write.liquidSVM <- function(model, filename){
  if(!(model$selected)){
    stop("Model has not been selected yet")
    return(invisible(model))
  }
  if(!( grepl("\\.fsol$",filename) || grepl("\\.sol$",filename) )){
    stop("Filename must have extension .fsol or .sol but is: ",filename)
    return(invisible(model))
  }
  vars <- c("all_vars", "explanatory", "formula", "gammas", 
             "lambdas", "levels", "solver", "predict.cols", "predict.prob")
  further <- lapply(vars,function(x) get(x,envir=model))
  names(further)<-vars
  .Call('liquid_svm_R_write_solution',as.integer(model$cookie),
        as.character(filename),serialize(further,NULL, ascii=T),
        PACKAGE='liquidSVM')
  invisible(model)
}

#' @export
#' @rdname read.liquidSVM
serialize.liquidSVM <- function(model, writeData=TRUE) {
  filename <- tempfile("solution",fileext=if(writeData)".fsol" else ".sol")
  write.liquidSVM(model,filename)
  obj <- readLines(filename)
  unlink(filename)
  obj
}

#' @export
#' @rdname read.liquidSVM
unserialize.liquidSVM <- function(obj,...) {
  fileext <- if(length(list(...))>0)".sol" else ".fsol"
  filename <- tempfile("solution",fileext=fileext)
  writeLines(obj,filename)
  ret <- read.liquidSVM(filename, ...)
  unlink(filename)
  ret$solution_aux_filename <- character(0)
  ret
}

# nocov start

# #' The \code{svmSerializeHook} and \code{svmUnserializeHook} methods can be used
# #' with \code{\link{serialize}} and \code{\link{unserialize}} as seen in the example.
# #' 
# #' @export
# #' @rdname read.liquidSVM
# #' \dontrun{
# #' # to serialize an object usinge serialize/unserialize (unserialize is broken currently!)
# #' banana <- liquidData('banana-bc')
# #' modelOrig <- mcSVM(Y~., banana$train)
# #' # we serialize it into a raw vector
# #' obj <- serialize(object=modelOrig, connection=NULL, refhook=svmSerializeHook)
# #' clean(modelOrig) # delete the SVM object
# #' 
# #' # now we unserialize it from that raw vector
# #' modelUnserialized <- unserialize(connection=obj, refhook=svmUnserializeHook)
# #' errors(test(modelUnserialized, banana$test))
# #' }
svmSerializeHook <- function(x) {
  if(inherits(x,"liquidSVM") || (is.environment(x)&&all(names(liquidSVMclass$fields()) %in% ls(x)))){
    obj <- serialize.liquidSVM(x)
    c("svm.fsol",obj)
  }else
    NULL
}

# #' @export
# #' @rdname read.liquidSVM
svmUnserializeHook <- function(x) {
  if(length(x)>=1 && x[1] == "svm.fsol"){
    unserialize.liquidSVM(x[-1])
  }else
    x
}
# nocov end
