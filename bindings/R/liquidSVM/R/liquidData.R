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


#' Loads or downloads training and testing data
#' 
#' This looks at several locations to find a  \code{\var{name}.train.csv} and \code{\var{name}.test.csv}.
#' If it does then it loads or downloads it, parses it, and returns an \code{liquidData}-object.
#' The files also can be gzipped having names  \code{\var{name}.train.csv.gz} and  \code{\var{name}.test.csv.gz}.
#' 
#' @param name name of the data set. If not given then a list of available names in \code{loc} is returned
#' @param factor_cols list of column numbers that are factors (or list of header names, if \code{header=TRUE})
#' @param header do the data files have headers
#' @param loc vector of locations where the data should be searched for
#' @return if name is specified an liquidData object: an environment with $train and $test datasets as well as $name and optionally $target as name of the target variable.
#' If no name is spacified a character vector of available names in \code{loc}.
#' @examples 
#' banana <- liquidData('banana-mc')
#' 
#' ## to get a smaller sample
#' liquidData('banana-mc',prob=0.2)
#' ## if you disable stratified then there is some variance in the group sizes:
#' liquidData('banana-mc',prob=0.2, stratified=FALSE)
#' 
#' \dontrun{
#' ## to downlad a file from our web directory
#' 
#' liquidData("gisette")
#' 
#' ## To get a list of available names:
#' liquidData()
#' }
#' @seealso \code{\link{ttsplit}}
#' @aliases print.liquidData
#' @export
liquidData <- function(name, factor_cols, header=FALSE,loc=c(".",
                        "~/liquidData",system.file('data',package='liquidSVM'),"../../../data"
                    ), prob=NULL, testSize=NULL, trainSize=NULL, stratified=NULL){
  if(missing(name)){
    ret <- character(0)
    for(i in loc){
      if(substr(i,0,7)=='http://'){
        csvFileNames <- try(({html <- readLines(i)
        pat <- '<a href="([^"]+)\\.train\\.csv(\\.gz)?">'
        gsub(pat, "\\1", regmatches(html, regexpr(pat, html)))
        }), silent=FALSE)
        if(class(csvFileNames) == "try-error")
          csvFileNames <- NULL
        csvGzFileNames <- NULL
      }else{
        filenames <- dir(i,pattern="*.train.csv$")
        csvFileNames <- substring(filenames,0,nchar(filenames)-10)
        filenames <- dir(i,pattern="*.train.csv.gz$")
        csvGzFileNames <- substring(filenames,0,nchar(filenames)-13)
      }
      ret <- c(ret, csvFileNames, csvGzFileNames)
    }
    return(sort(unique(ret)))
  }
  if(missing(factor_cols)){
    tmp <- sub("^([^.]+)\\..*$","\\1",name)
    class_datasets <- c('covtype','gisette','banana-bc','banana-mc','segment',"adult","australian","axa","bank-marketing","breast-cancer","chess-3","chess-6","cod-rna","covtype","covtype-full","diabetes","fourclass","german-numer","heart","higgs","ijcnn1","ionosphere","liver-disorders","magic","NumValPlus5DCleosDiabetis","pendigits","satimage","segment","shuttle","sonar","splice","svhn8vs9","svmguide1","svmguide3","thyroid-ann","vehicle")
    ### the above list contains the names of all data sets in isa-shared/sml/data except bank8fm
    if(any(tmp %in% class_datasets))
       factor_cols <- 1
    else
      factor_cols <- NULL
  }else{
    #factor_cols <- NULL
  }
  
  ret <- new.env()
  class(ret) <- c("liquidData")
  
  for(i in loc){
    trainname <- paste0(i,'/',name,'.train.csv')
    testname <- paste0(i,'/',name,'.test.csv')
    if(substr(i,0,7)=='http://'){
      # at the moment we trust that this works...
    }else{
      if(!file.exists(trainname)){
      trainname <- paste0(i,'/',name,'.train.csv.gz')
      testname <- paste0(i,'/',name,'.test.csv.gz')
      if(!file.exists(trainname)){
        next
      }
    }
    }
    ret$name <- name
    ret$loc <- i
    ret$filename <- trainname
    colClasses <- rep('factor',length(factor_cols))
    if(is.null(factor_cols)){
      colClasses <- NA
    }else if(header){
      names(colClasses) <- factor_cols
    }else{
      names(colClasses) <- paste('V',factor_cols,sep='')
    }
    tryResult <- try({
    ret$train <-read.table(trainname,sep=',',header=header,colClasses=colClasses)
    ret$test <-read.table(testname,sep=',',header=header,colClasses=colClasses)
    }, silent=TRUE)
    if(class(tryResult) == "try-error"){
      trainname <- paste0(trainname, ".gz")
      testname <- paste0(testname, ".gz")
      tryResult <- try({
        ret$train <-read.table(trainname,sep=',',header=header,colClasses=colClasses)
        ret$test <-read.table(testname,sep=',',header=header,colClasses=colClasses)
      }, silent=TRUE)
      if(class(tryResult) == "try-error"){
        next
      }
    }
    for(j in names(colClasses)){
      lev <- c(levels(ret$train[[j]]),levels(ret$train[[j]]))
      lev <- suppressWarnings(as.numeric(lev))
      if(any(is.na(lev)))
        next
      levi <- as.integer(lev)
      if(all(lev == levi))
        lev <- levi
      levels(ret$train[[j]]) <- levels(ret$test[[j]]) <- lev  ### CHECKME: Is this okay?
    }
    ret$target <- 'Y'
    cnames <- c('Y',paste0('X',1:(ncol(ret$test)-1)))
    colnames(ret$train) <- colnames(ret$test) <- cnames
    if(!(is.null(prob) && is.null(trainSize) && is.null(testSize))){
      ret <- sample.liquidData(ret, prob=prob, testSize=testSize, trainSize=trainSize, stratified=stratified)
    }
    return(ret)
  }
  stop(paste("Dataset", name,"not found"))
}

# This is the common part in ttsplit(...) and sample.liquidData(...)
splitIt <- function(data, target, prob, size, stratified){
  stopifnot(ncol(data)>=2)
  stopifnot(length(stratified)<=1)
  
  if(is.null(stratified)){
    if(is.null(target))
      stratified <- FALSE
    else{
      col <- data[,target]
      stratified <- is.factor(col)
    }
  }else if(!is.logical(stratified)){
    target <- stratified
    stratified <- TRUE
  }
  
  if(!is.null(size)){
    prob <- size / nrow(data)
  }else{
    size <- max(round(nrow(data) * prob), 1)
  }
  
  if(size >= nrow(data)){
    # first we force to use Inf
    if(size != Inf && size > nrow(data) && prob>1){
      warning("Trying to sample more data than available; if this is what you want use size=Inf or prob=1")
    }
    # now we just coule do
    #   return(1:nrow(data))
    # but we still want to have it shuffled:
    return(sample(nrow(data)))
  }
  
  if(stratified){
    ## split indices into groups
    groups <- split(1:nrow(data), data[,target])
    ## remove empty groups:
    groups <- groups[sapply(groups,length)>0]
    ## do the stratified sampling
    samples <- lapply(groups, function(x) sample(x, max(round(prob*length(x)),1)))
    samples <- do.call(c,samples)
    ## finally shuffle everything around
    sample(samples)
  }else{
    sample(nrow(data),size)
  }
  
}

cvSmldata <- function(data, folds=5, stratified=NULL, target=NULL){ # nocov start
  stopifnot(length(folds) == 1 && folds >= 2)
  n <- nrow(data)
  stopifnot(n>=folds)
  
  if(is.null(target) && 'target' %in% ls(data))
    target <- data$target
  
  if(is.null(stratified)){
    if(is.null(target)){
      stratified <- FALSE
    }else{
      col <- data[,target]
      stratified <- is.factor(col)
    }
  }else if(!is.logical(stratified)){
    target <- stratified
    stratified <- TRUE
  }else if(stratified){
    target <- 1
  }
  
  if(stratified){
    ## split indices into groups
    groups <- split(1:n, data[,target])
    ## remove empty groups:
    if(min(sapply(groups,length)) < folds)
      stop('some labels have less instances than folds, maybe retry with stratified=FALSE or boost these samples')
    ## do the stratified sampling and reteurn two columns
    ## where first column has index in original data and second the fold it should belong to
    samples <- lapply(groups, function(x){
        I <- sample(rep(1:folds, ceiling(length(x)/folds))[1:length(x)])
        cbind(x,I)
      })
    samples <- do.call(rbind,samples)
    ## resort by first column to get in the second column the folds
    I <- samples[order(samples[,1]),2]
  }else{
    I <- sample(rep(1:folds, ceiling(n/folds))[1:n])
  }
  
  lapply(1:folds, function(i){
    ret <- new.env()
    class(ret) <- 'liquidData'
    ret$name <- deparse(substitute(data))
    ret$orig <- data
    ret$train <- data[I!=i,]
    ret$test <- data[I==i,]
    if(!is.null(target))
      ret$target <- target
    ret
  })
} # nocov end


#' @param data the given data set
#' @param target optional name or index of the target variable.
#' If both this and \code{stratified} are not specified there will be no stratification.
#' @examples
#' ## to produce an liquidData from some dataset
#' ttsplit(iris)
#' # the following will be stratified
#' ttsplit(iris,'Species')
#' 
#' # specify a testSize:
#' ttsplit(trees, testSize=10)
#' @rdname liquidData
#' @export
ttsplit <- function(data, target=NULL, testProb=0.2, testSize=NULL, stratified=NULL){
  ret <- new.env()
  class(ret) <- c("liquidData")
  ret$name <- deparse(substitute(data))
  ret$orig <- data
  ret$target <- target
  
  ret$testIndex <- splitIt(data, target, testProb, testSize, stratified)
  
  ret$train <- data[-ret$testIndex,]
  ret$test <- data[ret$testIndex,]
  return(ret)
}

#' @param liquidData the given liquidData
#' @param prob probability of sample being put into test set
#' @param testProb probability of sample being put into test set
#' @param trainSize size of the train set. If stratified, this will only be approximately fulfilled.
#' @param testSize size of the test set. If stratified, this will only be approximately fulfilled.
#' @param stratified whether sampling should be done separately in every bin defined by
#' the unique values of the target column.
#' Also can be index or name of the column in \code{data} that should be used to define bins.
#' @examples 
#' ## example for sample.liquidData
#' banana <- liquidData('banana-mc')
#' sample.liquidData(banana, prob=0.1)
#' # this is equivalent to
#' liquidData('banana-mc', prob=0.1)
#' @rdname liquidData
#' @export
sample.liquidData <- function(liquidData, prob=0.2, trainSize=NULL, testSize=NULL, stratified=NULL){
  ## first we create a new environment
  ret <- new.env()
  class(ret) <- class(liquidData)
  for(n in ls(liquidData))
    assign(n, get(n, envir=liquidData), envir=ret)
  
  if(is.null(prob)){
    if(is.null(trainSize) && is.null(testSize))
      stop("one of prob, trainSize, testSize hast to be specified!")
    if(is.null(testSize)) testSize <- nrow(ret$test) * trainSize / nrow(ret$train)
    if(is.null(trainSize)) trainSize <- nrow(ret$train) * testSize / nrow(ret$test)
  }
  
  ret$name <- paste0(liquidData$name, " (sample)")
  
  ret$train <- ret$train[splitIt(ret$train, ret$target, prob, trainSize, stratified),]
  ret$test <- ret$test[splitIt(ret$test, ret$target, prob, testSize, stratified),]
  ret
}



#' @param x the model to print
#' @param ... other arguments to print.default
#' @export
#' @rdname liquidData
#' @examples
#' ## example for print
#' banana <- liquidData("banana-mc")
#' print(banana)
print.liquidData <- function(x,...){
  cat('LiquidData "', x$name, '"',sep='')
  cat(" with",nrow(x$train),"train samples and",nrow(x$test),"test samples")
  cat("\n")
  cat("  having",ncol(x$train),"columns")
  cnames <- colnames(x$train)
  if(length(cnames)>=1){
    cat(" named ")
    cat(cnames[1:min(10,length(cnames))],sep=",")
    if(length(cnames)>10) cat(", ...")
  }
  cat('\n')
  if(!is.null(x$target)){
    cat('  target "',x$target,'"',sep='')
    try({
      col <- x$train[,x$target]
      if(is.factor(col)){
        a <- table(col)
        cat(' factor with',length(a),'levels: ')
        b <- paste(names(a),' (',a,' samples)',sep="")
        cat(paste(b[1:min(3,length(b))], sep=', '))
        if(length(b)>3)
          cat(" ...")
      }else{
        cat(' mean ',mean(col),' range [',min(col),',',max(col),']', sep='')
      }
    })
    cat('\n')
  }
  # if(!is.null(x$filename)){
  #   cat('  train file: ',x$filename,sep='')
  #   cat("\n")
  # }
}
#' Write Smldata
#' 
#' Write \code{liquidData} in such a way that it is understood by liquidSVM command line utilities.
#' 
#' @param data the liquidData to write
#' @param location the location to write \code{name.train.csv} and \code{name.test.csv}
#' @param label the column with this index or this name will become the label column,
#'   and be written as the first column.
#' @param name the name of the file. If \code{NULL} (default) then takes the \code{data$name}
#' @param type the format of output. At the moment only \code{"csv"} is supported.
#' @export
write.liquidData <- function(data, location=".", label=1, name=NULL, type="csv"){
  if(!inherits(data, "liquidData"))
    warning("Data is not liquidData")
  if(!all(c("name","train","test")%in%ls(data)))
    stop("Data does note have necessary properties")
  if(is.null(name))
    name <- data$name
  if(is.character(label) & label %in% colnames(data$train))
    label <- which(label == colnames(data$train))
  trainLabs <- data$train[,label]
  testLabs <- data$test[,label]
  if(is.factor(trainLabs)){
    levs <- levels(trainLabs)
    if(all(as.character(as.numeric(levs))==levs)){
      trainLabs <- as.numeric(levs)[trainLabs]
      testLabs <- as.numeric(levels(testLabs))[testLabs]
    }else{
      trainLabs <- as.numeric(trainLabs)
      testLabs <- as.numeric(testLabs)
    }
  }
  if(type=="csv"){
    train <- cbind(trainLabs,data$train[,-label])
    test <- cbind(testLabs,data$test[,-label])
    write.table(train, paste0(location,'/',name,".train.csv"),sep=', ',col.names=F,row.names=F)
    write.table(test, paste0(location,'/',name,".test.csv"),sep=', ',col.names=F,row.names=F)
  }else if(type=="nla"){
    write.table(data$train[,-label], paste0(location,'/',name,".train.nla"),sep=', ',col.names=F,row.names=F)
    write.table(data$test[,-label], paste0(location,'/',name,".test.nla"),sep=', ',col.names=F,row.names=F)
  }else if(type=="libsvm" || type=="lsv"){
    f <- function(x){ paste( (1:length(x))[x!=0],x[x!=0], sep=":", collapse=' ') }
    train <- paste(data$train[,label],apply(data$train[,-label], 1, f))
    test <- paste(data$test[,label],apply(data$test[,-label], 1, f))
    writeLines(train, paste0(location,'/',name,".train.lsv"))
    writeLines(test, paste0(location,'/',name,".test.lsv"))
  }else if(type=="uci"){
    train <- cbind(data$train[,-label],trainLabs)
    test <- cbind(data$test[,-label],testLabs)
    write.table(train, paste0(location,'/',name,".train.uci"),sep=', ',col.names=F,row.names=F)
    write.table(test, paste0(location,'/',name,".test.uci"),sep=', ',col.names=F,row.names=F)
  }else if(type=="GURLS"){
    dir.create(paste0(location,'/',name))
    write.table(as.integer(factor(trainLabs)), paste0(location,'/',name,'/ytr_onecolumn.txt'),sep=', ',col.names=F,row.names=F)
    write.table(as.integer(factor(testLabs)), paste0(location,'/',name,'/yte_onecolumn.txt'),sep=', ',col.names=F,row.names=F)
    write.table(data$train[,-label], paste0(location,'/',name,"/Xtr.txt"),sep=', ',col.names=F,row.names=F)
    write.table(data$test[,-label], paste0(location,'/',name,"/Xte.txt"),sep=', ',col.names=F,row.names=F)
  }else if(type=="m-svm"){
    train <- c(nrow(data$train), ncol(data$train)-1, paste(apply(data$train[,-label],1,paste,collapse=' '),trainLabs, sep=' '))
    test <- c(nrow(data$test), ncol(data$test)-1, paste(apply(data$test[,-label],1,paste,collapse=' '),testLabs, sep=' '))
    writeLines(train, paste0(location,'/',name,".train.txt"))
    writeLines(test, paste0(location,'/',name,".test.txt"))
  }else{
    stop("type '",type,"' not known")
  }
}

# #' \code{covtype.*.train} and \code{covtype.*.test}
# #' 
# #' Binary variant of the classic covertype data set.
# #' Both the train and the test set have about 1000 and 5000 samples, resp.
# #' 
# #' The datasets were compiled from LIBSVM's version of the covertype dataset, which 
# #' in turn was taken from the UCI repository and preprocessed as in [RC02a]. 
# #' Copyright for this dataset is by Jock A. Blackard and Colorado State University.
# #'
# #' @name covtype
# #' @aliases covtype.1000.train covtype.1000.test covtype.5000.train covtype.5000.test
# #' @docType data
# NULL

#' \code{reg-1d.train} and \code{reg-1d.test}
#' 
#' Generated data set having a continuous Y variable and
#' a one-dimensional X variable.
#' 
#' Both the train and the test set have 2000 samples.
#' They were generated by the authors and their collaborators.
#'
#' @name reg-1d
#' @aliases reg-1d.train reg-1d.test
#' @docType data
NULL

#' \code{banana-bc.train}, \code{banana-bc.test}
#' \code{banana-mc.train}, and \code{banana-mc.test}
#' 
#' Generated data set having a binary or 4-level Y variable and
#' a two-dimensional X (first two levels resemble bananas).
#' Both the train and the test set have 2000 samples in the binary case,
#' and 4000 in the multi-class case.
#' They were generated by the authors and their collaborators.
#'
#' @name banana
#' @aliases banana-bc.train banana-bc.test banana-mc.train banana-mc.test
#' @docType data
NULL
