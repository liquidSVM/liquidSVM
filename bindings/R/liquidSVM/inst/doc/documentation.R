## ---- echo = FALSE-------------------------------------------------------
library(liquidSVM)
knitr::opts_chunk$set(collapse = TRUE, comment = "#>", eval=T)
set.seed(123)

options(digits=3)

myOwnCache <- function(name, envir=parent.frame(),vignette_dir="."){
  filename <- paste0(vignette_dir,'/demo_cache/',name,".R")
  if(exists(name, envir=envir)){
    dput(get(name, envir=envir), file=filename)
  }else if(file.exists(filename)){
    #message("Loading")
    assign(name,dget(filename),envir=envir)
  }else{
    warning(paste0("Did not have or load ",name))
  }
}


## ------------------------------------------------------------------------
compilationInfo()

## ----eval=F--------------------------------------------------------------
#  library(parallel)
#  ## how big should the cluster be
#  workers <- 2
#  cl <- makeCluster(workers)
#  ## how many threads should each worker use
#  threads <- 2
#  
#  sml <- liquidData('reg-1d')
#  clusterExport(cl, c("sml","threads","workers"))
#  obj <- parLapply(cl, 1:workers, function(i) {
#    library(liquidSVM)
#    ## to make it interesting use disjoint parts of sml$train
#    data <- sml$train[ seq(i,nrow(sml$train),workers) , ]
#    ## the second argument to threads sets the offset of cores
#    model <- lsSVM(Y~., data, threads=c(threads,threads*(i-1)) )
#    ## finally return the serialized solution
#    serialize.liquidSVM(model)
#  })
#  for(i in 1:workers){
#    ## get the solution in the master session
#    model <- unserialize.liquidSVM(obj[[i]])
#    print(errors(test(model,sml$test)))
#  }
#  #> val_error
#  #>   0.00542
#  #>  val_error
#  #>   0.00583

