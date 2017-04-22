
## General Information

Support vector machines (SVMs) and related kernel-based learning algorithms are
a well-known class of machine learning algorithms, for non-parametric
classification and regression. **liquidSVM** is an implementation of
SVMs whose key features are:

* fully integrated hyper-parameter selection,
* extreme speed on both small and large data sets,
* Bindings for [R](#R), [Python](#python), [MATLAB / Octave](#matlab-octave), [Java](#java), and [Spark](#spark),
* full flexibility for experts, and
* inclusion of a variety of different learning scenarios:
    - multi-class classification, ROC, and Neyman-Pearson learning,
    - least-squares, quantile, and expectile regression.


For questions and comments just contact us via
[mail](http://www.uni-stuttgart.de/cgi-bin/mail.cgi?liquidSVM=mathematik.uni-stuttgart.de).
There you also can ask to be registerd to our mailing list.

liquidSVM is licensed under [AGPL 3.0](http://www.gnu.org/licenses/agpl-3.0.html). In case you need another license, please contact [me](http://www.isa.uni-stuttgart.de/Steinwart/).

## Command Line interface

[Installation instructions](http://www.isa.uni-stuttgart.de/software/install.txt) for the command line versions.

|                                      |                                                                                    |
|--------------------------------------|------------------------------------------------------------------------------------|
| Terminal version for Linux/OS X      | [liquidSVM.tar.gz](http://www.isa.uni-stuttgart.de/software/liquidSVM.tar.gz)      |
| Terminal version for Windows (64bit) | avx2: [liquidSVM.zip](http://www.isa.uni-stuttgart.de/software/avx2/liquidSVM.zip) |
|                                      | avx:  [liquidSVM.zip](http://www.isa.uni-stuttgart.de/software/avx/liquidSVM.zip)  |
|                                      | sse2: [liquidSVM.zip](http://www.isa.uni-stuttgart.de/software/sse2/liquidSVM.zip) |
| Previous versions                    | [v1.1](v1.1) (June 2016), [v1.0](v1.0) (January 2016)                              |

On Linux and Mac on the terminal `liquidSVM` can be used in the following way:
```bash
wget www.isa.uni-stuttgart.de/software/liquidSVM.tar.gz
tar xzf liquidSVM.tar.gz
cd liquidSVM
make all
scripts/mc-svm.sh banana-mc 1 2
```

## R

Read the [demo vignette](http://www.isa.uni-stuttgart.de/software/R/demo.html) for a tutorial on installing liquidSVM-package and how to use it and the [documentation vignette](http://www.isa.uni-stuttgart.de/software/R/documentation.html) for more advanced installation options and usage.

An easy usage is:
```r
install.packages("liquidSVM")
library(liquidSVM)
banana <- liquidData('banana-mc')
model <- mcSVM( Y~. , banana$train, display=1, threads=2)
result <- test(model, banana$test)
errors(result)
```

## Python

Read the [demo notebook](http://www.isa.uni-stuttgart.de/software/python/demo.html) for a tutorial on installing liquidSVM-package and how to use it and the [homepage](bindings/python/) for more advanced installation options and usage.

To install use:
```bash
pip install --user liquidSVM
```
and then in Python you can use it e.g. like:
```python
from liquidSVM import *
banana = LiquidData('banana-mc')
model = mcSVM(banana.train, display=1, threads=2)
result, err = model.test(banana.test)
```


## MATLAB / Octave

The [MATLAB bindings](bindings/matlab/) are currently getting a better interface,
and this is a preview version.

> It does currently not work on Windows.

For installation download the Toolbox
[liquidSVM.mltbx](http://www.isa.uni-stuttgart.de/software/matlab/liquidSVM.mltbx)
and install it in MATLAB by double clicking it.
To compile and add paths issue:
```matlab
makeliquidSVM native
```
Then you can use it like:
```matlab
banana = liquidData('banana-mc');
model = svm_mc(banana.train, 'DISPLAY', 1, 'THREADS', 2);
[result, err] = model.test(banana.test);
```

Most of the code also works in \texttt{Octave}
if you use [liquidSVM-octave.zip](http://www.isa.uni-stuttgart.de/software/matlab/liquidSVM-octave.zip).


## Java
The main homepage is [here](bindings/java/).
For installation download [liquidSVM-java.zip](http://www.isa.uni-stuttgart.de/software/java/liquidSVM-java.zip) and unzip it.
The classes are all in package `de.uni_stuttgart.isa.liquidsvm` and an easy example is:
```java
LiquidData banana = new LiquidData("banana-mc");
SVM model = new MC(banana.train, new Config().display(1).threads(2));
ResultAndErrors result = model.test(banana.test);
```
If this is implemented in the file `Example.java` this can be compiled and run using
```bash
# if you want to compile the JNI-native library:
make lib
# compile your Java-Code
javac -classpath liquidSVM.jar Example.java
# and run it
java -Djava.library.path=. -cp .:liquidSVM.jar Example
```

## Spark
This is a preview version, see [Spark](bindings/spark/) for more details.
Download [liquidSVM-spark.zip](http://www.isa.uni-stuttgart.de/software/spark/liquidSVM-spark.zip) and unzip it.
Assume you have `Spark` installed in `$SPARK_HOME` you can issue:
```bash
make lib
export LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH
$SPARK_HOME/bin/spark-submit \
  --class de.uni_stuttgart.isa.liquidsvm.spark.App \
  liquidSVM-spark.jar banana-mc
```
If you have configured `Spark` to be used on a cluster with `Hadoop` use:
```bash
hdfs dfs -put data/covtype-full.train.csv data/covtype-full.test.csv .
make lib
$SPARK_HOME/bin/spark-submit --files ../libliquidsvm.so \
  --conf spark.executor.extraLibraryPath=. \
  --conf spark.driver.extraLibraryPath=. \
  --class de.uni_stuttgart.isa.liquidsvm.spark.App \
  --num-executors 14 liquidSVM-spark.jar covtype-full
```



Extra Datasets for the Demo
---------------------------

[covertype data set with 35.090 training and 34.910 test samples](http://www.isa.uni-stuttgart.de/software/covtype.35000.zip)

[covertype data set with 522.909 training and 58.103 test samples](http://www.isa.uni-stuttgart.de/software/covtype.zip)

Both datasets were compiled from [LIBSVM's version of the covertype dataset](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html), which
in turn was taken from the [UCI repository](http://mlr.cs.umass.edu/ml/datasets/Covertype) and preprocessed as in [\[RC02a\].](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/ref.html#RC02a)
Copyright for this dataset is by Jock A. Blackard and Colorado State University.

Citation
--------

If you use liquidSVM, please cite it as:

> I. Steinwart and P. Thomann.
> *liquidSVM: A fast and versatile SVM package.*
> [*ArXiv e-prints 1702.06899*](http://arxiv.org/abs/1702.06899), February 2017.
