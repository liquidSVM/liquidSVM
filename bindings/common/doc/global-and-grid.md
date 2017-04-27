## Overview of Configuration Parameters

`display`
  : This parameter determines the amount of output of
    you see at the screen: The larger its value is,
    the more you see. This can help as a progress indication.

`scale`
  : If set to a true value then for every feature in the training data
    a scaling is calculated so that its values lie in the interval $[0,1]$.
    The training then is performed using these scaled values
    and any testing data is scaled transparently as well.
    
    Because SVMs are not scale-invariant any data should be scaled
    for two main reasons: First that all features have the same weight,
    and second to assure that the default gamma parameters that liquidSVM
    provide remain meaningful.
    
    If you do not have scaled the data previously this is an easy option.

`threads`
  : This parameter determines the number of cores
    used for computing the kernel matrices, the
    validation error, and the test error.
    
    * `threads=0` (default) means that all physical cores of your CPU run one thread. 
    * `threads=-1` means that all but one physical cores of your CPU run one thread.

`partition_choice`
:   This parameter determines the way the input space
    is partitioned. This allows larger data sets for which
    the kernel matrix does not fit into memory.
    
    * `partition_choice=0` (default) disables partitioning.
    * `partition_choice=6` gives usually highest speed.
    * `partition_choice=5` gives usually the best test error.

`grid_choice`
:   This parameter determines the size of the hyper-
    parameter grid used during the training phase.
    Larger values correspond to larger grids. By
    default, a 10x10 grid is used. Exact descriptions are given in the next section.
    
`adaptivity_control`
:   This parameter determines, whether an adaptive
    grid search heuristic is employed. Larger values
    lead to more aggressive strategies. The default
    `adaptivity_control = 0` disables the heuristic.
    
`random_seed`
:   This parameter determines the seed for the random
    generator. `random_seed` = -1 uses the internal
    timer create the seed. All other values lead to
    repeatable behavior of the svm.
    
`folds`
  : How many folds should be used.

## Specialized configuration parameters

Parameters for regression (least-squares, quantile, and expectile)

`clipping`
  : This parameter determines whether the decision
    functions should be clipped at the specified
    value. The value `clipping` = -1.0 leads to
    an adaptive clipping value, whereas `clipping` = 0
    disables clipping.

Parameter for multiclass classification determine the multiclass strategy:
`mc-type=0`
  : AvA with hinge loss.
`mc-type=1`
  : OvA with least squares loss.
`mc-type=2`
  : OvA with hinge loss.
`mc-type=3`
  : AvA with least squares loss.

Parameters for Neyman-Pearson Learning

`class`
  : The class, the `constraint` is enforced on.

`constraint`
  : The constraint on the false alarm rate. The script
    actually considers a couple of values around the
    value of `constraint` to give the user an informed
    choice.



## Hyperparameter Grid

For Support Vector Machines two hyperparameters need to be determined:

* `gamma` the bandwith of the kernel 
* `lambda` has to be chosen such that neither over- nor underfitting happen.
    lambda values are the classical regularization parameter in front of the norm term.

liquidSVM has a built-in a cross-validation scheme to calculate validation errors for
many values of these hyperparameters and then to choose the best pair.
Since there are two parameters this means we consider a two-dimensional grid.

For both parameters either specific values can be given or a geometrically spaced grid can be specified.

`gamma_steps`, `min_gamma`, `max_gamma`
  : specifies in the interval between `min_gamma` and `max_gamma` there should be `gamma_steps` many values

`gammas`
  : e.g. `gammas=c(0.1,1,10,100)` will do these four gamma values

`lambda_steps`, `min_lambda`, `max_lambda`
  : specifies in the interval between `min_lambda` and `max_lambda` there should be `lambda_steps` many values

`lambdas`
  : e.g. `lambdas=c(0.1,1,10,100)` will do these four lambda values

`c_values`
  : the classical term in front of the empirical error term,
    e.g. `c_values=c(0.1,1,10,100)` will do these four cost values (basically inverse of `lambdas`)

Note the min and max values are 
scaled according the the number of samples, the dimensionality
of the data sets, the number of folds used, and the estimated 
diameter of the data set.

Using `grid_choice` allows for some general choices of these parameters

|`grid_choice`  | 0     | 1      |     2   |
|---------------|-------|--------|---------|
|`gamma_steps`  | 10    | 15     | 20      |
|`lambda_steps` | 10    | 15     | 20      | 
|`min_gamma`    | 0.2   | 0.1    | 0.05    |
|`max_gamma`    | 5.0   | 10.0   | 20.0    |
|`min_lambda`   | 0.001 | 0.0001 | 0.00001 |
|`max_lambda`   | 0.01  | 0.01   | 0.01    |


Using negative values of `grid_choice` we create a grid with listed gamma and lambda values:

| `grid_choice` |  -1     |
|----------|-----------------|
|`gammas`  | `c(10.0, 5.0, 2.0, 1.0, 0.5, 0.25, 0.1, 0.05)` |
|`lambdas` | `c(1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001)` |

|`grid_choice` | -2          |
|----------|------------------|
|`gammas`  | `c(10.0, 5.0, 2.0, 1.0, 0.5, 0.25, 0.1, 0.05)` |
|`c_values`| `c(0.01, 0.1, 1, 10, 100, 1000, 10000)` |


## Adaptive Grid

An adaptive grid search can be activated. The higher the values
of `MAX_LAMBDA_INCREASES` and `MAX_NUMBER_OF_WORSE_GAMMAS` are set
the more conservative the search strategy is. The values can be 
freely modified.

|`ADAPTIVITY_CONTROL`         |  1 | 2 |
|-----------------------------|----|---|
|`MAX_LAMBDA_INCREASES`       |  4 | 3 |
|`MAX_NUMBER_OF_WORSE_GAMMAS` |  4 | 3 |


## Cells

A major issue with SVMs is that for larger sample sizes the kernel matrix
does not fit into the memory any more.
Classically this gives an upper limit for the class of problems that traditional
SVMs can handle without significant runtime increase.
Furthermore also the time complexity is at least $O(n^2)$.

liquidSVM implements two major concepts to circumvent these issues.
One is random chunks which is known well in the literature.
However we prefer the new alternative of splitting the space into
spatial cells and use local SVMs on every cell.

If you specify `useCells=TRUE` then the sample space $X$ gets partitioned into
a number of cells.
The training is done first for cell 1 then for cell 2 and so on.
Now, to predict the label for a value $x\in X$ liquidSVM first finds out
to which cell this $x$ belongs and then uses the SVM of that cell to predict
a label for it.

> If you run into memory issues turn cells on: `useCells=TRUE`

This is quite performant, since the complexity in both
time and memore are both $O(\mbox{CELLSIZE} \times n)$
and this holds both for training as well as testing!
It also can be shown that the quality of the solution is comparable,
at least for moderate dimensions.



The cells can be configured using the `partition_choice`:

1) This gives a partition into random
    chunks of size 2000
    
    `VORONOI=c(1, 2000)`
    
2)  This gives a partition into 10
    random chunks
    
    `VORONOI=c(2, 10)`
    
3)  This gives a Voronoi partition into cells with radius 
    not larger than 1.0. For its creation a subsample containing
    at most 50.000 samples is used. 
    
	  `VORONOI=c(3, 1.0, 50000)`
	  
4)  This gives a Voronoi partition into cells with at most 2000 
    samples (approximately). For its creation a subsample containing
    at most 50.000 samples is used. A shrinking heuristic is used 
    to reduce the number of cells.
    
	  `VORONOI=c(4, 2000, 1, 50000)`
	  
5)  This gives a overlapping regions with at most 2000 samples
    (approximately). For its creation a subsample containing
    at most 50.000 samples is used. A stopping heuristic is used 
    to stop the creation of regions if 0.5 * 2000 samples have
    not been assigned to a region, yet. 
    
    `VORONOI=c(5, 2000, 0.5, 50000, 1)`
	  
6)  This splits the working sets into Voronoi like with `PARTITION_TYPE=4`.
    Unlike that case, the centers for the Voronoi partition are
    found by a recursive tree approach, which in many cases may be
    faster.
    
	 `VORONOI=c(6, 2000, 1, 50000, 2.0, 20, 4,)`

The first parameter values correspond to `NO_PARTITION`, `RANDOM_CHUNK_BY_SIZE`, `RANDOM_CHUNK_BY_NUMBER`, `VORONOI_BY_RADIUS`, `VORONOI_BY_SIZE`, `OVERLAP_BY_SIZE`

## Weights

* qt, ex:
  Here the number of considered tau-quantiles/expectiles as well as the 
  considered tau-values are defined. You can freely change these
  values but notice that the list of tau-values is space-separated!
  
* npl, roc:
  Here, you define, which weighted classification problems will be considered.
  The choice is usually a bit tricky. Good luck ...

```r
NPL:
WEIGHT_STEPS=10
MIN_WEIGHT=0.001
MAX_WEIGHT=0.5
GEO_WEIGHTS=1

ROC:
WEIGHT_STEPS=9
MAX_WEIGHT=0.9
MIN_WEIGHT=0.1
GEO_WEIGHTS=0
```


## More Advanced Parameters

The following parameters should only employed by experienced users and are self-explanatory for these:

`KERNEL`
  : specifies the kernel to use, at the moment either `GAUSS_RBF` or `POISSON`
    
`RETRAIN_METHOD`
  : After training on grids and folds there are only solutions on folds.
    In order to construct a global solution one can either retrain on the whole
    training data (`SELECT_ON_ENTIRE_TRAIN_SET`) or
    the (partial) solutions from the training are
    kept and combined using voting (`SELECT_ON_EACH_FOLD` default)
    
`store_solutions_internally`
  : If this is true (default in all applicable cases) then the solutions of the train phase
    are stored and can be just reused in the select phase.
    If you slowly run out of memory during the train phase maybe disable this.
    However then in the select phase the best models have to be trained again.
    
For completeness here are some values that usually get set by the learning scenario

`SVM_TYPE`
  : `KERNEL_RULE`, `SVM_LS_2D`, `SVM_HINGE_2D`, `SVM_QUANTILE`, `SVM_EXPECTILE_2D`, `SVM_TEMPLATE`
    
`LOSS_TYPE`
  : `CLASSIFICATION_LOSS`, `MULTI_CLASS_LOSS`, `LEAST_SQUARES_LOSS`, `WEIGHTED_LEAST_SQUARES_LOSS`, `PINBALL_LOSS`, `TEMPLATE_LOSS`
    
`VOTE_SCENARIO`
  : `VOTE_CLASSIFICATION`, `VOTE_REGRESSION`, `VOTE_NPL`
    
`KERNEL_MEMORY_MODEL`
  : `LINE_BY_LINE`, `BLOCK`, `CACHE`, `EMPTY`
    
`FOLDS_KIND`
  : `BLOCKS`, `ALTERNATING`, `RANDOM`, `STRATIFIED`, `RANDOM_SUBSET`
    
`WS_TYPE`
  : `FULL_SET`, `MULTI_CLASS_ALL_VS_ALL`, `MULTI_CLASS_ONE_VS_ALL`, `BOOT_STRAP`
