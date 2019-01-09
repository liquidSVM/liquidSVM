# liquidSVM v1.2.2 (Release date: 2019-01-09)

* skipping test that fails due to website reorganization

# liquidSVM v1.2.0 (Release date: 2017-07-15)

* version now up to date with core version
* added predict.prob parameter to activate conditional probability estimation
* added grouped cross validation
* `mlr` support
* added more explicit aliases for learning scenarios:
  `svmRegression` for `lsSVM`, `svmMulticlass` for `mcSVM`,
  `svmQuantileRegression` for `qtSVM`, and `svmExpectileRegression` for `exSVM`.
  The old ones are still valid and the main implementation.
* added configuration defaults, e.g.:
  `options(liquidSVM.default.display=1)`, `options(liquidSVM.default.scale=TRUE)`, ...
* predict now returns the correct number of columns for expectile, quantile, ...
* fixed demo vignette run time issues
* fixed CUDA-compilation issue
* TARGET defaults to "default" on Sparc
* fixed PROTECT-issues (thanks to kalibera/rchk!) and switched from CXX1X to CXX11
* test-coverage over 90%

# liquidSVM v1.0.1 (Release date: 2017-03-01)

* documentation and demo now can now assume that the package is on CRAN
* fixed compilation issues for CRAN farm
* fixed some Valgrind/UBSAN comments
* better README.md

# liquidSVM v1.0.0 (Release date: 2017-02-23)

* added function aliases svmMulticlass, svmRegression, ...
* changed train/select to trainSVMs/selectSVMs
* unit tests do not give warnings for boundary hits
* added citation of arXiv-paper


# liquidSVM v0.9.9 (Release date: 2017-02-05)

* changed name to liquidSVM
* changed class svm to liquidSVM
* changed smldata to liquidData


# liquidSVM v0.9.8 (Release date: 2016-11-18)

* Auto-scaling with scale=TRUE
* warning() about hitting boundary (not only in display=1 output)
* read-/writeSolution
* RC4-Class (Needs R 2.12) and nicer print method
* liquidData now with better sampling interface, also stratified and nicer print method
* basic mlr-support
* plot method for ROC-Curve
* getCover/getSolution interfaces
* many stability changes, sanity checks, bug fixes


# liquidSVM v0.9 (Release date: 2016-04-18)

* Adding learning scenarios: the functionality of `scripts/{ls,mc,qt,ex,npl,roc,bs}-svm.sh`
  now is reproduced in `{ls,mc,qt,ex,npl,roc,bs}SVM`.
* Configuration of the SVM with higher level parameters (e.g. `grid_choice=1, max_gamma=100, kernel="poisson"`).
* The new ISA-repsitory under http://www.isa.uni-stuttgart.de/software/R allows for easy installation.
* Usage without formulas: `svm(iris[,-5],iris$Species)`
* Automagically usage of testing data if available.
* New build configurations (`native, generic, debug, empty`).

* Warning: the command line arguments, which could be passed to `svm(...)` have now to be passed
    as `command.args=list(...)`. Better: use the new configuration!
