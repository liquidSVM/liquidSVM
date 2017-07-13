## ---- echo = FALSE-------------------------------------------------------
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

options(liquidSVM.default.threads=1)

library(liquidSVM)


## ------------------------------------------------------------------------
# Load test and training data
reg <- liquidData('reg-1d')

## ------------------------------------------------------------------------
model <- svm(Y~., reg$train)

## ----ls-reg-plot, eval=T, fig.width=7, fig.height=3----------------------
plot(reg$train$X1, reg$train$Y,pch='.', ylim=c(-.2,.8), ylab='', xlab='', axes=F)
curve(predict(model, x),add=T,col='red')

## ------------------------------------------------------------------------
banana <- liquidData('banana-mc')
banana

## ----mc-banana-plot, echo=TRUE, fig.height=3, fig.width=7----------------
model <- svm(Y~., banana$train)
plot(banana$train$X1, banana$train$X2,pch='o', col=banana$train$Y, ylab='', xlab='', axes=F)
x <- seq(-1,1,.05)
z <- matrix(predict(model,expand.grid(x,x)),length(x))
contour(x,x,z, add=T, levels=1:4,col=1,lwd=4)

## ------------------------------------------------------------------------
modelTrees <- svm(Height ~ Girth + Volume, trees)  # least squares
modelIris <- svm(Species ~ ., iris)  # multiclass classification

## ------------------------------------------------------------------------
predict(modelTrees, trees[21:31, ])
predict(modelIris, iris[3:8, ])

## ------------------------------------------------------------------------
all.equal( predict(modelIris, iris[3:8, ]) , iris$Species[3:8] )

## ------------------------------------------------------------------------
qu <- ttsplit(quakes)

## ------------------------------------------------------------------------
model <- svm(mag ~ ., qu$train)

## ------------------------------------------------------------------------
result <- test(model, qu$test)
errors(result)

## ---- eval=F-------------------------------------------------------------
#  model <- lsSVM(mag ~ . , qu, display=1)
#  #> [...]
#  #> Warning: The best gamma was 0 times at the lower boundary and 5 times at the
#  #> upper boundary of your gamma grid. 5 times a gamma value was selected.
#  #> [...]
#  errors(model$last_result)
#  #> val_error
#  #>     0.109

## ---- eval=F-------------------------------------------------------------
#  errors(lsSVM(mag ~ . , qu, max_gamma=100)$last_result)
#  #> val_error
#  #>    0.0367

## ----eval=F--------------------------------------------------------------
#  banana <- liquidData('banana-bc')
#  modelOrig <- mcSVM(Y~., banana$train)
#  write.liquidSVM(modelOrig, "banana-bc.fsol")
#  write.liquidSVM(modelOrig, "banana-bc.sol")
#  clean(modelOrig) # delete the SVM object
#  
#  # now we read it back from the file
#  modelRead <- read.liquidSVM("banana-bc.fsol")
#  # No need to train/select the data!
#  errors(test(modelRead, banana$test))
#  
#  # to read the model where no data was saved we have to make sure, we get the same training data:
#  banana <- liquidData('banana-bc')
#  # then we can read it
#  modelDataExternal <- read.liquidSVM("banana-bc.sol", Y~., banana$train)
#  result <- test(modelDataExternal, banana$test)
#  
#  # to serialize an object use:
#  banana <- liquidData('banana-bc')
#  modelOrig <- mcSVM(Y~., banana$train)
#  # we serialize it into a raw vector
#  obj <- serialize.liquidSVM(modelOrig)
#  clean(modelOrig) # delete the SVM object
#  
#  # now we unserialize it from that raw vector
#  modelUnserialized <- unserialize.liquidSVM(obj)
#  errors(test(modelUnserialized, banana$test))

## ----cells-banana-plot, eval=T, echo=T, fig.width=4, fig.height=4,results='hide', warning=F, message=F----
banana <- liquidData('banana-bc')
model <- mcSVM(Y~.,banana$train, voronoi=c(4,500))
centers <- getCover(model)$samples
plot(banana$train[,2:3],col=banana$train$Y)
points(centers,pch='x',cex=2,col=3)

if(require(deldir)){
  voronoi <- deldir::deldir(centers$X1,centers$X2,rw=c(range(banana$train$X1),range(banana$train$X2)))
  plot(voronoi,wlines="tess",add=TRUE, lty=1)
  text(centers$X1,centers$X2,1:nrow(centers),pos=1)
}


## ---- eval=F-------------------------------------------------------------
#  co <- liquidData('covtype.10000')
#  system.time(svm(Y~., co$train, threads=3))
#  #>   user  system elapsed
#  #> 28.208   0.124  11.191

## ---- eval=F-------------------------------------------------------------
#  co <- liquidData('covtype.50000')
#  system.time(svm(Y~.,co$train,useCells=TRUE,threads=3))
#  #>    user  system elapsed
#  #> 252.395   1.076  98.119

## ---- eval=F-------------------------------------------------------------
#  co <- liquidData('covtype-full')
#  system.time(svm(Y~.,co$train,useCells=TRUE,threads=3))
#  #>     user   system  elapsed
#  #> 1383.535    4.752  397.559

## ---- eval=F-------------------------------------------------------------
#  gi <- liquidData('gisette')
#  model <- init.liquidSVM(Y~.,gi$train)

## ---- eval=F-------------------------------------------------------------
#  system.time(trainSVMs(model,d=1,gpus=1,threads=1))
#  #>   user  system elapsed
#  #>     57     10       67
#  system.time(trainSVMs(model,d=1,gpus=0,threads=4))
#  #>   user  system elapsed
#  #>    392       1     110

## ---- eval=F-------------------------------------------------------------
#  system.time(trainSVMs(model,d=1,gpus=1,threads=4))
#  #>   user  system elapsed
#  #>     94      42      67
#  system.time(trainSVMs(model,d=1,gpus=0,threads=1))
#  #>   user  system elapsed
#  #>    327       1     329

## ---- eval=F-------------------------------------------------------------
#  folds <- 5
#  co <- liquidData('covtype.1000')
#  
#  system.time(ours <- svm(Y~., co$train, folds=folds, threads=2))
#  #>   user  system elapsed
#  #>  1.525   0.016   0.958

## ---- eval=F-------------------------------------------------------------
#  GAMMA <- 1/(ours$gammas)^2
#  COST <- 1/(2 * (folds-1)/folds * nrow(co$train) * ours$lambdas)

## ---- eval=F-------------------------------------------------------------
#  system.time(e1071::tune.svm(Y~., data=co$train, gamma=GAMMA,cost=COST, scale=F, e1071::tune.control(cross=folds)))
#  #>   user  system elapsed
#  #> 382.364   0.832 385.521

## ---- eval=F-------------------------------------------------------------
#  co <- liquidData('covtype.5000')  # ca. 5000 rows
#  system.time(ours <- svm(Y~., co$train, folds=folds, threads=2))
#  #>   user  system elapsed
#  #> 30.237   0.120  15.676
#  
#  system.time(e1071::tune.svm(Y~., data=co$train, gamma=GAMMA,cost=COST, scale=F, e1071::tune.control(cross=folds)))
#  #>      user    system   elapsed
#  #> 11199.732     4.324 11238.407

## ----eval=F--------------------------------------------------------------
#  co <- liquidData('covtype.10000')  # ca. 10000 rows
#  gamma <- 3.1114822
#  cost <- 0.01654752
#  system.time(ours <- svm(Y ~ ., co$train, g=c("[",gamma,"]"), l=c("[",cost,"]",1),folds=1,threads=4,d=1))
#  #>   user  system elapsed
#  #>  4.836   0.356   2.134
#  
#  system.time(theirs <- e1071::svm(Y~., co$train, gamma=1/gamma^2,cost=cost, scale=F))
#  #>   user    system   elapsed
#  #> 26.502     0.032    26.618

## ----eval=F--------------------------------------------------------------
#  co <- liquidData('covtype.35000')  # ca. 35000 rows
#  system.time(ours <- svm(Y ~ ., co$train, g=c("[",gamma,"]"), l=c("[",cost,"]",1),folds=1,threads=4,d=1))
#  #>    user  system elapsed
#  #>  99.830   4.544  36.949
#  system.time(theirs <- e1071::svm(Y~., co$train, gamma=1/gamma^2,cost=cost, scale=F))
#  #>    user  system elapsed
#  #> 330.557   0.176 331.834

## ----eval=F--------------------------------------------------------------
#  system.time(ours <- svm(Y ~ ., co$train,folds=1,threads=4,d=0))
#  #>    user  system elapsed
#  #> 816.475   5.164 225.934

## ----eval=F--------------------------------------------------------------
#  model <- init.liquidSVM(formula, data)
#  trainSVMs(model, ...)
#  selectSVMs(model)

## ----multiclass-banana, echo=F, eval=T, fig.width=3, fig.height=3--------
banana <- liquidData('banana-mc')
par(mar=rep(0,4))
with(banana$train, plot(X1,X2, col=Y, ylab='', xlab='', axes=F))

## ---- echo=F, eval=F-----------------------------------------------------
#  banana <- liquidData('banana-mc')
#  #banana <- liquidSVM:::sample.liquidData(banana)
#  
#  model <- mcSVM(Y~., banana, mc_type="AvA_hinge")
#  errors(model$last_result)
#  model$last_result[1:3,]
#  
#  model <- mcSVM(Y~., banana, mc_type="OvA_ls")
#  errors(model$last_result)
#  model$last_result[1:3,]
#  
#  model <- mcSVM(Y~., banana, mc_type="AvA_ls")
#  errors(model$last_result)
#  model$last_result[1:3,]
#  
#  # For completeness the following is also possible even though you should not use it:
#  model <- mcSVM(Y~., banana, mc_type="OvA_hinge")
#  errors(model$last_result)
#  model$last_result[1:3,]

## ---- eval=F-------------------------------------------------------------
#  banana <- liquidData('banana-mc')
#  
#  model <- mcSVM(Y~., banana, mc_type="AvA_hinge")
#  errors(model$last_result)
#  #>   result     1vs2     1vs3     1vs4     2vs3     2vs4     3vs4
#  #> 0.217250 0.142083 0.111500 0.092500 0.073500 0.073500 0.000625
#  model$last_result[1:3,]
#  #>   result 1vs2 1vs3 1vs4 2vs3 2vs4 3vs4
#  #> 1      1    1    1    1    2    4    4
#  #> 2      4    1    1    4    2    4    4
#  #> 3      4    1    1    4    2    4    4
#  
#  model <- mcSVM(Y~., banana, mc_type="OvA_ls")
#  errors(model$last_result)
#  #>    result 1vsOthers 2vsOthers 3vsOthers 4vsOthers
#  #>    0.2147    0.1545    0.1227    0.0777    0.0737
#  model$last_result[1:3,]
#  #>   result 1vsOthers 2vsOthers 3vsOthers 4vsOthers
#  #> 1      1   0.99149    -0.964    -0.924    -0.928
#  #> 2      4  -0.45494    -1.000    -0.994     0.387
#  #> 3      1  -0.00657    -0.991    -0.993    -0.111
#  
#  model <- mcSVM(Y~., banana, mc_type="AvA_ls")
#  errors(model$last_result)
#  #>   result     1vs2     1vs3     1vs4     2vs3     2vs4     3vs4
#  #> 0.212500 0.140000 0.107000 0.089500 0.074000 0.074000 0.000625
#  model$last_result[1:3,]
#  #>   result   1vs2   1vs3    1vs4   2vs3  2vs4  3vs4
#  #> 1      1 -0.963 -0.979 -0.9966 -0.605 0.894 0.995
#  #> 2      4 -0.753 -0.998  0.5268 -0.953 1.000 1.000
#  #> 3      1 -0.996 -1.000 -0.0506 -0.894 1.000 1.000
#  
#  # For completeness the following is also possible even though you should not use it:
#  model <- mcSVM(Y~., banana, mc_type="OvA_hinge")
#  errors(model$last_result)
#  #>    result 1vsOthers 2vsOthers 3vsOthers 4vsOthers
#  #>    0.2235    0.1555    0.1275    0.0795    0.0750
#  model$last_result[1:3,]
#  #>   result 1vsOthers 2vsOthers 3vsOthers 4vsOthers
#  #> 1      1     1.000    -0.829    -0.720    -0.981
#  #> 2      4    -0.876    -0.995    -0.740     0.923
#  #> 3      4    -0.202    -0.987    -0.729     0.198

## ----quantile-reg, eval=F------------------------------------------------
#  reg <- liquidData('reg-1d')
#  quantiles_list <- c(0.05, 0.1, 0.5, 0.9, 0.95)
#  
#  model <- qtSVM(Y ~ ., reg$train, weights=quantiles_list)
#  
#  result_qt <- test(model,reg$test)
#  errors(result_qt)
#  #> [1] 0.00714 0.01192 0.02682 0.01251 0.00734

## ---- echo=F-------------------------------------------------------------
## if the previous is not evaluated we still need:
quantiles_list <- c(0.05, 0.1, 0.5, 0.9, 0.95)
reg <- liquidData('reg-1d')
myOwnCache('result_qt')

## ----quantile-reg-plot, eval=T, fig.width=7, fig.height=3----------------
I <- order(reg$test$X1)
par(mar=rep(.1,4))
plot(Y~X1, reg$test[I,],pch='.', ylim=c(-.2,.8), ylab='', xlab='', axes=F)
for(i in 1:length(quantiles_list))
  lines(reg$test$X1[I], result_qt[I,i], col=i+1)

## ----expectile-reg, eval=F-----------------------------------------------
#  reg <- liquidData('reg-1d')
#  expectiles_list <- c(0.05, 0.1, 0.5, 0.9, 0.95)
#  
#  model <- exSVM(Y ~ ., reg$train, weights=expectiles_list)
#  
#  result_ex <- test(model, reg$test)
#  errors(result_ex)

## ---- echo=F-------------------------------------------------------------
## if the previous is not evaluated we still need:
expectiles_list <- c(0.05, 0.1, 0.5, 0.9, 0.95)
reg <- liquidData('reg-1d')
myOwnCache('result_ex')
#> [1] 0.00108 0.00155 0.00270 0.00161 0.00143

## ----expectile-reg-plot, eval=T, fig.width=7, fig.height=3---------------
I <- order(reg$test$X1)
par(mar=rep(.1,4))
plot(Y~X1, reg$test[I,],pch='.', ylim=c(-.2,.8), ylab='', xlab='', axes=F)
for(i in 1:length(expectiles_list))
  lines(reg$test$X1[I], result_ex[I,i], col=i+1)
legend('bottomright', col=6:2, lwd=1, legend=expectiles_list[5:1])

## ----npl, eval=F---------------------------------------------------------
#  banana <- liquidData('banana-bc')
#  npl_constraints <- c(0.025,0.033,0.05,0.075,0.1)
#  
#  # class=-1 specifies the normal class
#  model <- nplSVM(Y ~ ., banana, class=-1, constraint.factor=npl_constraints,threads=0,display=1)
#  
#  result_npl <- model$last_result
#  errors(result_npl)
#  #> [1] 0.437 0.437 0.322 0.308 0.230

## ---- echo=F-------------------------------------------------------------
## if the previous is not evaluated we still need:
banana <- liquidData('banana-bc')
npl_constraints <- c(3,4,6,9,12)/120
myOwnCache('result_npl')

## ----eval=T--------------------------------------------------------------
false_alarm_rate <- apply(result_npl[banana$test$Y==-1,]==1,2,mean)
detection_rate <- apply(result_npl[banana$test$Y==1,]==1,2,mean)
rbind(npl_constraints,false_alarm_rate,detection_rate)

## ----roc, eval=F---------------------------------------------------------
#  banana <- liquidData('banana-bc')
#  
#  model <- rocSVM(Y ~ ., banana$train, threads=0,display=1)
#  
#  result_roc <- test(model, banana$test)

## ---- echo=F-------------------------------------------------------------
## if the previous is not evaluated we still need:
banana <- liquidData('banana-bc')
myOwnCache('result_roc')

## ----roc-banana-plot, eval=T, fig.width=4, fig.height=4------------------
false_positive_rate <- apply(result_roc[banana$test$Y==-1,]==1,2,mean)
detection_rate <- apply(result_roc[banana$test$Y==1,]==1,2,mean)
plot(false_positive_rate, detection_rate, xlim=0:1,ylim=0:1,asp=1, type='b', pch='x')
abline(0,1,lty=2)

## ----roc-ls-banana-plot, eval=T, fig.width=4, fig.height=4---------------
model.ls <- lsSVM(Y~.,banana$train)
plotROC(model.ls, banana$test, xlim=0:1,ylim=0:1,asp=1, type='l')
points(false_positive_rate, detection_rate, pch='x', col='red')

## ----kernel-plot, eval=T, fig.width=4, fig.height=4----------------------
covtype <- liquidData("covtype.1000")$train[1:100,-1]
a <- kern(covtype)
a[1:4,1:4]
image(liquidSVM::kern(covtype, gamma=1.1, type="gauss"))
image(liquidSVM::kern(covtype, gamma=1.1, type="poisson"))

## ----eval=F--------------------------------------------------------------
#  # take 10% of training and testing data
#  liquidData('reg-1d', prob=0.1)
#  # a sample of 400 train samples and the same relative size of test samples
#  liquidData('reg-1d', trainSize=400)
#  # a sample of 400 train samples and all test samples
#  liquidData('reg-1d', trainSize=400, testSize=Inf)

