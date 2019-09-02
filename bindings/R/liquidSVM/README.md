# liquidSVM

`liquidSVM` is a package written in C++ that
provides SVM-type solvers for various classification and regression tasks.
Because of a fully integrated hyper-parameter selection, very carefully implemented solvers,
multi-threading and GPU support,
and several built-in data decomposition strategies  it provides unprecedented speed
for small training sizes as well as for data sets of tens of millions of samples.

You can use it e.g. for multi-class classification, least squares (kernel) regression,
or even quantile regression, etc.:
```r
install.packages("liquidSVM")
library(liquidSVM)

model <- mcSVM(Species ~ ., iris)
predict(model, iris)

model <- lsSVM(Height ~ ., trees)
y <- predict(model, trees)

model <- svmQuantileRegression(Height ~ ., trees)
y <- test(model, trees)
```

If you install build the package to be used on several machines please use the following:
```r
install.packages("liquidSVM", configure.args="generic")
```

For details please look at the vignettes [demo](inst/doc/demo.html) and [documentation](inst/doc/documentation.html).
Also check the help `?liquidSVM` and `?svm`.
For the command-line version and other bindings go to (http://www.isa.uni-stuttgart.de/software/).
