
 %% LiquidSVM_Model (Constructor)
 % Constructor of the class LiquidSVM_Model
 %% Syntax
 %   dataSourceHandleObj = LiquidSVM_DataSourceHandle(source,type,key,value);
 %% Description
 % LiquidSVM_Model create a new SVM and initialize with training
 % data x and labels y.
 %% Input Arguments
 %
 %% required
 % |x|: numeric matrix containing the feature Samples of the
 % traininng data
 %
 %%
 % |y|: numeric or categorical vector of same length as x
 % containing the training labels
 %% optional
 % *key:* |'readSol'|
 % *value:* In case a model was trained, selected written to a
 % 'sol'-file this option allows to load the model.
 % Default: |''|
 %%
 % *key:* |'display'|
 % *value:* Has to be provided as a scalar double. This parameter
 % determines the amount of information displayed to the screen:
 % The larger its value is, the more you see.
 %  Default: |0|
 %%
 % *key:* |threads|
 % *value:* This parameter determines the number of cores
 % used for computing the kernel matrices, the
 % validation error, and the test error.
 %%
 % |threads =  0| (default) means that all physical cores of your CPU run one thread.
 % |threads = -1|  means that all but one physical cores of your CPU run one thread.
 %%
 % *key:* |partition_choice|
 % *value:* This parameter determines the way the input space
 % is partitioned. This allows larger data sets for which
 % the kernel matrix does not fit into memory.
 %%
 % * |partition_choice=0| (default) disables partitioning.
 % * |partition_choice=4| gives usually highest speed.
 % * |partition_choice=6| gives usually the best test error.
 %%
 % *key:* |grid_choice|
 % *value:* This parameter determines the size of the hyper-
 % parameter grid used during the training phase.
 % Larger values correspond to larger grids. By
 % default, a 10x10 grid is used. Exact descriptions are given
 % in the 'Hyperparameter Grid' section (see below).
 %%
 % *key:* |adaptivity_control|
 % *value:* This parameter determines, whether an adaptive
 % grid search heuristic is employed. Larger values
 % lead to more aggressive strategies. The default
 % |adaptivity_control = 0| disables the heuristic.
 %%
 % *key:* |random_seed|
 % *value:* This parameter determines the seed for the random
 % generator. |random_seed| = -1 uses the internal
 % timer create the seed. All other values lead to
 % repeatable behavior of the svm.
 %%
 % *key:* |folds|
 % *value:* How many folds should be used.
 %
 %% Hyperparameter Grid
 % For Support Vector Machines two hyperparameters have to be figured out:
 %
 % * |gamma| the bandwith of the kernel
 % * |lambda| has to be chosen such that neither over- nor underfitting happen.
 % lambda values are the classical regularization parameter in front of the norm term.
 %
 % LiquidSVM has build in a cross-validation scheme to calculate validation errors for
 % many values of these hyperparameters and then to choose the best pair.
 % Since there are two parameters this means we consider a two-dimensional grid.
 %
 % For both parameters either specific values can be given or a geometrically spaced grid can be specified.
 %
 % *key:* |gamma_steps|, |min_gamma|, |max_gamma|
 % *value:* specifies in the interval between |min_gamma| and |max_gamma| there
 % should be |gamma_steps| many values.
 %
 % *key:* |gammas|
 % *value:* e.g. |gammas = [0.1 1 10 100]| will do these four gamma values
 %
 % *key:* |lambda_steps|, |min_lambda|, |max_lambda|
 % *value:* specifies in the interval between |min_lambda| and |max_lambda| there should be |lambda_steps| many values
 %
 % *key:* |lambdas|
 % *value:*  e.g. |lambdas = [0.1,1,10,100]| will do these four lambda values
 %
 % *key:* |c_values|
 % *value:*  the classical term in front of the empirical error term,
 % e.g. |c_values = [0.1,1,10,100]| will do these four cost values (basically inverse of |lambdas|)
 %
 % *Note:* the min and max values are
 % scaled according to the number of samples, the dimensionality
 % of the data sets, the number of folds used, and the estimated
 % diameter of the data set.
 %%
 % Using |grid_choice| allows for some general choices of these parameter.
 %%
 % <html>
 % <table border=1>
 % <tr><td>grid_choice</td><td>0</td><td>1</td><td>2</td></tr>
 % <tr><td>gamma_steps</td><td>10</td><td>15</td><td>20</td></tr>
 % <tr><td>lambda_steps_steps</td><td>10</td><td>15</td><td>20</td></tr>
 % <tr><td>min_gamma</td><td>0.2</td><td>0.2</td><td>0.05</td></tr>
 % <tr><td>max_gamma</td><td>5</td><td>10</td><td>20</td></tr>
 % <tr><td>min_lambda</td><td>0.001</td><td>0.0001</td><td>0.00001</td></tr>
 % <tr><td>max_lambda</td><td>0.01</td><td>0.01</td><td>0.01</td></tr>
 % </table>
 % </html>
 %%
 % Using negative values of |grid_choice| we create a grid with listed gamma and lambda values:
 %%
 % <html>
 % <table border=1>
 % <tr><td>grid_choice</td><td>-1</td><td>-2</td></tr>
 % <tr><td>gamma</td><td>[10.0, 5.0, 2.0, 1.0, 0.5, 0.25, 0.1, 0.05]</td><td>[10.0, 5.0, 2.0, 1.0, 0.5, 0.25, 0.1, 0.05]</td></tr>
 % <tr><td>lambdas</td><td>[1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001]</td><td>-</td></tr>
 % <tr><td>c_values</td><td>-</td><td>[0.01, 0.1, 1, 10, 100, 1000, 10000]</td></tr>
 % </table>
 % </html>
 %% Adaptive Grid
 %
 % An adaptive grid search can be activated. The higher the values
 % of |MAX_LAMBDA_INCREASES| and |MAX_NUMBER_OF_WORSE_GAMMAS| are set
 % the more conservative the search strategy is. The values can be
 % freely modified.
 %
 %   {'ADAPTIVITY_CONTROL',1} = {'ADAPTIVE_SEARCH', 1, 'MAX_LAMBDA_INCREASES', 4, 'MAX_NUMBER_OF_WORSE_GAMMAS', 4}
 %   {'ADAPTIVITY_CONTROL',2} = {'ADAPTIVE_SEARCH', 1, 'MAX_LAMBDA_INCREASES', 3, 'MAX_NUMBER_OF_WORSE_GAMMAS', 3}
 %
 %% Cells
 %
 % A major issue with SVMs is that for larger sample sizes the kernel matrix
 % does not fit into the memory any more.
 % Classically this gives an upper limit for the class of problems that traditional
 % SVMs can handle without significant runtime increase.
 % Furthermore also the time complexity is $O(n^2)$.
 %
 % LiquidSVM implements two major concepts to circumvent these issues.
 % One is random chunks which is known well in the literature.
 % However we prefer the new alternative of splitting the space into
 % spatial cells and use local SVMs on every cell.
 %
 %%
 % *key:* |'useCells'|
 % *value:*
 % If you specify |true| then the sample space $X$ gets partitioned into
 % a number of cells.
 % The training is done first for cell 1 then for cell 2 and so on.
 % Now, to predict the label for a value $x\in X$ LiquidSVM first finds out
 % to which cell this $x$ belongs and then uses the SVM of that cell to predict
 % a label for it.
 %
 % _If you run into memory issues turn cells on:_ |'useCells',true|
 %
 % This is quite performant, since the complexity in both
 % time and memory are both $O(\mbox{CELLSIZE} \times n)$
 % and this holds both for training as well as testing!
 % It also can be shown that the quality of the solution is comparable,
 % at least for moderate dimensions.
 %
 %%
 % *key:* |'partition_choice'|
 % *value:*
 % The cells can be configured using the |partition_choice|:
 %
 % * This gives a partition into random chunks of size 2000
 %
 %   {'PARTITION_CHOICE',1} = {'VORONOI','1 2000'}
 %
 % * This gives a partition into 10 random chunks
 %
 %   {'PARTITION_CHOICE',2} = {'VORONOI','2 10'}
 %
 % * This gives a Voronoi partition into cells with radius
 % not larger than 1.0. For its creation a subsample containing
 % at most 50.000 samples is used.
 %
 %   {'PARTITION_CHOICE',3} = {'VORONOI','3 1.0 50000'}
 %
 % * This gives a Voronoi partition into cells with at most 2000
 % samples (approximately). For its creation a subsample containing
 % at most 50.000 samples is used. A shrinking heuristic is used
 % to reduce the number of cells.
 %
 %   {'PARTITION_CHOICE',4} = {'VORONOI', '4 2000 1 50000'}
 %
 % * This gives a overlapping regions with at most 2000 samples
 % (approximately). For its creation a subsample containing
 % at most 50.000 samples is used. A stopping heuristic is used
 % to stop the creation of regions if 0.5 * 2000 samples have
 % not been assigned to a region, yet.
 %
 %   {'PARTITION_CHOICE',5} = {'VORONOI','5 2000 0.5 50000 1'}
 %
 % * This splits the working sets into Voronoi like with |PARTITION_TYPE=4|.
 % Unlike that case, the centers for the Voronoi partition are
 % found by a recursive tree approach, which in many cases may be
 % faster. This is the same as |{'useCells',true}|
 %
 %   {'PARTITION_CHOICE',6} = {'VORONOI','6 2000 1 50000 2.0 20 4'}
 %% Weights
 %
 % * qtSVM, exSVM:
 %   Here the number of considered tau-quantiles as well as the
 %   considered tau-values are defined. You can freely change these
 %   values but notice that the list of tau-values is space-separated!
 %
 % * nplSVM, rocSVM:
 %   Here, you define, which weighted classification problems will be considered.
 %   The meaning of the values can be found by typing svm-train -w
 %   The choice is usually a bit tricky. Good luck ...
 %
 % |||r
 % NPL:
 % WEIGHT_STEPS=10
 % MIN_WEIGHT=0.001
 % MAX_WEIGHT=0.5
 % GEO_WEIGHTS=1
 %
 % ROC:
 % WEIGHT_STEPS=9
 % MAX_WEIGHT=0.9
 % MIN_WEIGHT=0.1
 % GEO_WEIGHTS=0
 % |||
 %
 %% More Advanced Parameters
 %
 % The following parameters should only employed by experienced users and are self-explanatory for these:
 %
 % |SVM_TYPE|
 %   : "KERNEL_RULE", "SVM_LS_2D", "SVM_HINGE_2D", "SVM_QUANTILE", "SVM_EXPECTILE_2D", "SVM_TEMPLATE"
 %
 % |LOSS_TYPE|
 %   : "CLASSIFICATION_LOSS", "MULTI_CLASS_LOSS", "LEAST_SQUARES_LOSS", "WEIGHTED_LEAST_SQUARES_LOSS", "PINBALL_LOSS", "TEMPLATE_LOSS"
 %
 % |VOTE_SCENARIO|
 %   : "VOTE_CLASSIFICATION", "VOTE_REGRESSION", "VOTE_NPL"
 %
 % |KERNEL|
 %   : "GAUSS_RBF", "POISSON"
 %
 % |KERNEL_MEMORY_MODEL|
 %   : "LINE_BY_LINE", "BLOCK", "CACHE", "EMPTY"
 %
 % |RETRAIN_METHOD|
 %   : "SELECT_ON_ENTIRE_TRAIN_SET", "SELECT_ON_EACH_FOLD"
 %
 % |FOLDS_KIND|
 %   : "FROM_FILE", "BLOCKS", "ALTERNATING", "RANDOM", "STRATIFIED", "RANDOM_SUBSET"
 %
 % |WS_TYPE|
 %   : "FULL_SET", "MULTI_CLASS_ALL_VS_ALL", "MULTI_CLASS_ONE_VS_ALL", "BOOT_STRAP"
 %
 % |PARTITION_KIND|
 %   : "NO_PARTITION", "RANDOM_CHUNK_BY_SIZE", "RANDOM_CHUNK_BY_NUMBER", "VORONOI_BY_RADIUS", "VORONOI_BY_SIZE", "OVERLAP_BY_SIZE"
 %% Output:
 % |model| a new object of class LiquidSVM_Model
 %% Example:
 %%
 % url source
 dataSources1 = LiquidSVM_DataSourceHandle('http://www.isa.uni-stuttgart.de/liquidData','url');
 %%
 % local source
 dataSources2 = LiquidSVM_DataSourceHandle(pwd,'local');
 %% See also
 % LiquidSVM_DataFile, LiquidSVM_Data
