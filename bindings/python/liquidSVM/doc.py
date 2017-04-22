### This file is generated do not edit it by hand!


def configuration():
    """    Overview of Configuration Parameters
    ------------------------------------
    
    ``display``
        This parameter determines the amount of output of you see at the
        screen: The larger its value is, the more you see. This can help as
        a progress indication.
    
    ``scale``
        If set to a true value then for every feature in the training data a
        scaling is calculated so that its values lie in the interval
        :math:`[0,1]`. The training then is performed using these scaled
        values and any testing data is scaled transparently as well.
    
        Because SVMs are not scale-invariant any data should be scaled for
        two main reasons: First that all features have the same weight, and
        second to assure that the default gamma parameters that liquidSVM
        provide remain meaningful.
    
        If you do not have scaled the data previously this is an easy
        option.
    
    ``threads``
        This parameter determines the number of cores used for computing the
        kernel matrices, the validation error, and the test error.
    
        -  ``threads=0`` (default) means that all physical cores of your CPU
           run one thread.
        -  ``threads=-1`` means that all but one physical cores of your CPU
           run one thread.
    
    ``partition_choice``
        This parameter determines the way the input space is partitioned.
        This allows larger data sets for which the kernel matrix does not
        fit into memory.
    
        -  ``partition_choice=0`` (default) disables partitioning.
        -  ``partition_choice=6`` gives usually highest speed.
        -  ``partition_choice=5`` gives usually the best test error.
    
    ``grid_choice``
        This parameter determines the size of the hyper- parameter grid used
        during the training phase. Larger values correspond to larger grids.
        By default, a 10x10 grid is used. Exact descriptions are given in
        the next section.
    
    ``adaptivity_control``
        This parameter determines, whether an adaptive grid search heuristic
        is employed. Larger values lead to more aggressive strategies. The
        default ``adaptivity_control = 0`` disables the heuristic.
    
    ``random_seed``
        This parameter determines the seed for the random generator.
        ``random_seed`` = -1 uses the internal timer create the seed. All
        other values lead to repeatable behavior of the svm.
    
    ``folds``
        How many folds should be used.
    
    Specialized configuration parameters
    ------------------------------------
    
    Parameters for regression (least-squares, quantile, and expectile)
    
    ``clipping``
        This parameter determines whether the decision functions should be
        clipped at the specified value. The value ``clipping`` = -1.0 leads
        to an adaptive clipping value, whereas ``clipping`` = 0 disables
        clipping.
    
    Parameter for multiclass classification determine the multiclass
    strategy: ``mc-type=0`` : AvA with hinge loss. ``mc-type=1`` : OvA with
    least squares loss. ``mc-type=2`` : OvA with hinge loss. ``mc-type=3`` :
    AvA with least squares loss.
    
    Parameters for Neyman-Pearson Learning
    
    ``class``
        The class, the ``constraint`` is enforced on.
    
    ``constraint``
        The constraint on the false alarm rate. The script actually
        considers a couple of values around the value of ``constraint`` to
        give the user an informed choice.
    
    Hyperparameter Grid
    -------------------
    
    For Support Vector Machines two hyperparameters need to be determined:
    
    -  ``gamma`` the bandwith of the kernel
    -  ``lambda`` has to be chosen such that neither over- nor underfitting
       happen. lambda values are the classical regularization parameter in
       front of the norm term.
    
    liquidSVM has a built-in a cross-validation scheme to calculate
    validation errors for many values of these hyperparameters and then to
    choose the best pair. Since there are two parameters this means we
    consider a two-dimensional grid.
    
    For both parameters either specific values can be given or a
    geometrically spaced grid can be specified.
    
    ``gamma_steps``, ``min_gamma``, ``max_gamma``
        specifies in the interval between ``min_gamma`` and ``max_gamma``
        there should be ``gamma_steps`` many values
    
    ``gammas``
        e.g. ``gammas=[0.1,1,10,100]`` will do these four gamma values
    
    ``lambda_steps``, ``min_lambda``, ``max_lambda``
        specifies in the interval between ``min_lambda`` and ``max_lambda``
        there should be ``lambda_steps`` many values
    
    ``lambdas``
        e.g. ``lambdas=[0.1,1,10,100]`` will do these four lambda values
    
    ``c_values``
        the classical term in front of the empirical error term, e.g.
        ``c_values=[0.1,1,10,100]`` will do these four cost values
        (basically inverse of ``lambdas``)
    
    Note the min and max values are scaled according the the number of
    samples, the dimensionality of the data sets, the number of folds used,
    and the estimated diameter of the data set.
    
    Using ``grid_choice`` allows for some general choices of these
    parameters
    
    +--------------------+---------+----------+-----------+
    | ``grid_choice``    | 0       | 1        | 2         |
    +====================+=========+==========+===========+
    | ``gamma_steps``    | 10      | 15       | 20        |
    +--------------------+---------+----------+-----------+
    | ``lambda_steps``   | 10      | 15       | 20        |
    +--------------------+---------+----------+-----------+
    | ``min_gamma``      | 0.2     | 0.1      | 0.05      |
    +--------------------+---------+----------+-----------+
    | ``max_gamma``      | 5.0     | 10.0     | 20.0      |
    +--------------------+---------+----------+-----------+
    | ``min_lambda``     | 0.001   | 0.0001   | 0.00001   |
    +--------------------+---------+----------+-----------+
    | ``max_lambda``     | 0.01    | 0.01     | 0.01      |
    +--------------------+---------+----------+-----------+
    
    Using negative values of ``grid_choice`` we create a grid with listed
    gamma and lambda values:
    
    +-------------------+----------------------------------------------------------------------+
    | ``grid_choice``   | -1                                                                   |
    +===================+======================================================================+
    | ``gammas``        | ``[10.0, 5.0, 2.0, 1.0, 0.5, 0.25, 0.1, 0.05]``                     |
    +-------------------+----------------------------------------------------------------------+
    | ``lambdas``       | ``[1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001]``   |
    +-------------------+----------------------------------------------------------------------+
    
    +-------------------+----------------------------------------------------+
    | ``grid_choice``   | -2                                                 |
    +===================+====================================================+
    | ``gammas``        | ``[10.0, 5.0, 2.0, 1.0, 0.5, 0.25, 0.1, 0.05]``   |
    +-------------------+----------------------------------------------------+
    | ``c_values``      | ``[0.01, 0.1, 1, 10, 100, 1000, 10000]``          |
    +-------------------+----------------------------------------------------+
    
    Adaptive Grid
    -------------
    
    An adaptive grid search can be activated. The higher the values of
    ``MAX_LAMBDA_INCREASES`` and ``MAX_NUMBER_OF_WORSE_GAMMAS`` are set the
    more conservative the search strategy is. The values can be freely
    modified.
    
    +----------------------------------+-----+-----+
    | ``ADAPTIVITY_CONTROL``           | 1   | 2   |
    +==================================+=====+=====+
    | ``MAX_LAMBDA_INCREASES``         | 4   | 3   |
    +----------------------------------+-----+-----+
    | ``MAX_NUMBER_OF_WORSE_GAMMAS``   | 4   | 3   |
    +----------------------------------+-----+-----+
    
    Cells
    -----
    
    A major issue with SVMs is that for larger sample sizes the kernel
    matrix does not fit into the memory any more. Classically this gives an
    upper limit for the class of problems that traditional SVMs can handle
    without significant runtime increase. Furthermore also the time
    complexity is at least :math:`O(n^2)`.
    
    liquidSVM implements two major concepts to circumvent these issues. One
    is random chunks which is known well in the literature. However we
    prefer the new alternative of splitting the space into spatial cells and
    use local SVMs on every cell.
    
    If you specify ``useCells=TRUE`` then the sample space :math:`X` gets
    partitioned into a number of cells. The training is done first for cell
    1 then for cell 2 and so on. Now, to predict the label for a value
    :math:`x\in X` liquidSVM first finds out to which cell this :math:`x`
    belongs and then uses the SVM of that cell to predict a label for it.
    
        If you run into memory issues turn cells on: ``useCells=TRUE``
    
    This is quite performant, since the complexity in both time and memore
    are both :math:`O(\mbox{CELLSIZE} \times n)` and this holds both for
    training as well as testing! It also can be shown that the quality of
    the solution is comparable, at least for moderate dimensions.
    
    The cells can be configured using the ``partition_choice``:
    
    1) This gives a partition into random chunks of size 2000
    
       ``VORONOI=[1, 2000]``
    
    2) This gives a partition into 10 random chunks
    
       ``VORONOI=[2, 10]``
    
    3) This gives a Voronoi partition into cells with radius not larger than
       1.0. For its creation a subsample containing at most 50.000 samples
       is used.
    
       ``VORONOI=[3, 1.0, 50000]``
    
    4) This gives a Voronoi partition into cells with at most 2000 samples
       (approximately). For its creation a subsample containing at most
       50.000 samples is used. A shrinking heuristic is used to reduce the
       number of cells.
    
       ``VORONOI=[4, 2000, 1, 50000]``
    
    5) This gives a overlapping regions with at most 2000 samples
       (approximately). For its creation a subsample containing at most
       50.000 samples is used. A stopping heuristic is used to stop the
       creation of regions if 0.5 \* 2000 samples have not been assigned to
       a region, yet.
    
       ``VORONOI=[5, 2000, 0.5, 50000, 1]``
    
    6) This splits the working sets into Voronoi like with
       ``PARTITION_TYPE=4``. Unlike that case, the centers for the Voronoi
       partition are found by a recursive tree approach, which in many cases
       may be faster.
    
       ``VORONOI=[6, 2000, 1, 50000, 2.0, 20, 4,]``
    
    The first parameter values correspond to ``NO_PARTITION``,
    ``RANDOM_CHUNK_BY_SIZE``, ``RANDOM_CHUNK_BY_NUMBER``,
    ``VORONOI_BY_RADIUS``, ``VORONOI_BY_SIZE``, ``OVERLAP_BY_SIZE``
    
    Weights
    -------
    
    -  qt, ex: Here the number of considered tau-quantiles/expectiles as
       well as the considered tau-values are defined. You can freely change
       these values but notice that the list of tau-values is
       space-separated!
    
    -  npl, roc: Here, you define, which weighted classification problems
       will be considered. The choice is usually a bit tricky. Good luck ...
    
    .. code:: r
    
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
    
    More Advanced Parameters
    ------------------------
    
    The following parameters should only employed by experienced users and
    are self-explanatory for these:
    
    ``KERNEL``
        specifies the kernel to use, at the moment either ``GAUSS_RBF`` or
        ``POISSON``
    
    ``RETRAIN_METHOD``
        After training on grids and folds there are only solutions on folds.
        In order to construct a global solution one can either retrain on
        the whole training data (``SELECT_ON_ENTIRE_TRAIN_SET``) or the
        (partial) solutions from the training are kept and combined using
        voting (``SELECT_ON_EACH_FOLD`` default)
    
    ``store_solutions_internally``
        If this is true (default in all applicable cases) then the solutions
        of the train phase are stored and can be just reused in the select
        phase. If you slowly run out of memory during the train phase maybe
        disable this. However then in the select phase the best models have
        to be trained again.
    
    For completeness here are some values that usually get set by the
    learning scenario
    
    ``SVM_TYPE``
        ``KERNEL_RULE``, ``SVM_LS_2D``, ``SVM_HINGE_2D``, ``SVM_QUANTILE``,
        ``SVM_EXPECTILE_2D``, ``SVM_TEMPLATE``
    
    ``LOSS_TYPE``
        ``CLASSIFICATION_LOSS``, ``MULTI_CLASS_LOSS``,
        ``LEAST_SQUARES_LOSS``, ``WEIGHTED_LEAST_SQUARES_LOSS``,
        ``PINBALL_LOSS``, ``TEMPLATE_LOSS``
    
    ``VOTE_SCENARIO``
        ``VOTE_CLASSIFICATION``, ``VOTE_REGRESSION``, ``VOTE_NPL``
    
    ``KERNEL_MEMORY_MODEL``
        ``LINE_BY_LINE``, ``BLOCK``, ``CACHE``, ``EMPTY``
    
    ``FOLDS_KIND``
        ``FROM_FILE``, ``BLOCKS``, ``ALTERNATING``, ``RANDOM``,
        ``STRATIFIED``, ``RANDOM_SUBSET``
    
    ``WS_TYPE``
        ``FULL_SET``, ``MULTI_CLASS_ALL_VS_ALL``,
        ``MULTI_CLASS_ONE_VS_ALL``, ``BOOT_STRAP``
    
    
    """


def trainArgs():
    '''Arguments for SVM.train
    -  ``f=c(<kind>,<number>,[<train_fraction>],[<neg_fraction>])``
    
       Selects the fold generation method and the number of folds. If < 1.0,
       then the folds for training are generated from a subset with the
       specified size and the remaining samples are used for validation.
       Meaning of specific values: = 1 => each fold is a contiguous block =
       2 => alternating fold assignmend = 3 => random = 4 => stratified
       random = 5 => random subset ( and required)
    
       Allowed values: : integer between 1 and 5 : integer >= 1 : float >
       0.0 and <= 1.0 : float > 0.0 and < 1.0
    
       Default values: = 3 = 5 = 1.00
    
    -  ``g=c(<size>,<min_gamma>,<max_gamma>,[<scale>])``
    -  ``g=<gamma_list>``
    
       The first variant sets the size of the gamma grid and its endpoints
       and . The second variant uses for the gamma grid.
    
       Meaning of specific values: Flag indicating whether and are scaled
       based on the sample size, the dimension, and the diameter.
    
       Allowed values: : integer >= 1 : float > 0.0 : float > 0.0 : bool
    
       Default values: = 10 = 0.200 = 5.000 = 1
    
    -  ``GPU=<gpus>``
    
       Sets the number of GPUs that are going to be used. Currently, there
       is no checking whether your system actually has many GPUs. In
       addition, the number of used threads is reduced to .
    
       Allowed values: : integer between 0 and ???
    
       Default values: = 0
    
       Unfortunately, this option is not activated for the binaries you are
       currently using. Install CUDA and recompile to activate this option.
    
    -  ``h=[<level>]``
    
       Displays all help messages.
    
       Meaning of specific values: = 0 => short help messages = 1 =>
       detailed help messages
    
       Allowed values: : 0 or 1
    
       Default values: = 0
    
    -  ``i=c(<cold>,<warm>)``
    
       Selects the cold and warm start initialization methods of the solver.
       In general, this option should only be used in particular situations
       such as the implementation and testing of a new solver or when using
       the kernel cache.
    
       Meaning of specific values: For values between 0 and 6, both and have
       the same meaning taken from Steinwart et al, 'Training SVMs without
       offset', JMLR 2011. These are: 0 Sets all coefficients to zero. 1
       Sets all coefficients to C. 2 Uses the coefficients of the previous
       solution. 3 Multiplies all coefficients by C\_new/C\_old. 4
       Multiplies all unbounded SVs by C\_new/C\_old. 5 Multiplies all
       coefficients by C\_old/C\_new. 6 Multiplies all unbounded SVs by
       C\_old/C\_new.
    
       Allowed values: Depends on the solver, but the range of is always a
       subset of the range of .
    
       Default values: Depending on the solver, the (hopefully) most
       efficient method is chosen.
    
    -  ``k=c(<type>,[aux-file],[<Tr_mm_Pr>,[<size_P>],<Tr_mm>,[<size>],<Va_mm_Pr>,<Va_mm>])``
    
       Selects the type of kernel and optionally the memory model for the
       kernel matrices.
    
       Meaning of specific values: = 0 => Gaussian RBF = 1 => Poisson = 2 =>
       Experimental hierarchical Gauss kernel => Name of the file that
       contains additional information for the hierarchical Gauss kernel.
       Only this kernel type requires this option. = 0 => not contiguously
       stored matrix = 1 => contiguously stored matrix = 2 => cached matrix
       = 3 => no matrix stored => size of kernel cache in MB Here, X=Tr
       stands for the training matrix and X=Va for the validation matrix. In
       both cases, Y=Pr stands for the pre-kernel matrix, which stores the
       distances between the samples. If is set, then the other three flags
       need to be set, too. The values must only be set if a cache is
       chosen. NOTICE: Not all possible combinations are allowed.
    
       Allowed values: : integer between 0 and 2 : integer between 0 and 3 :
       integer not smaller than 1
    
       Default values: = 0 = 1 = 1024 = 512
    
    -  ``l=c(<size>,<min_lambda>,<max_lambda>,[<scale>])``
    -  ``l=c(<lambda_list>,[<interpret_as_C>])``
    
       The first variant sets the size of the lambda grid and its endpoints
       and . The second variant uses , after ordering, for the lambda grid.
    
       Meaning of specific values: Flag indicating whether is internally
       devided by the average number of samples per fold. Flag indicating
       whether the lambda list should be interpreted as a list of C values
    
       Allowed values: : integer >= 1 : float > 0.0 : float > 0.0 : bool :
       bool
    
       Default values: = 10 = 0.001 = 0.100 = 1 = 0
    
    -  ``L=c(<loss>,[<clipp>],[<neg_weight>,<pos_weight>])``
    
       Sets the loss that is used to compute empirical errors. The optional
       value specifies where the predictions are clipped during validation.
       The optional weights can only be set if specifies a loss that has
       weights.
    
       Meaning of specific values: = 0 => binary classification loss = 2 =>
       least squares loss = 3 => weighted least squares loss = 4 => pinball
       loss = 5 => your own template loss = -1.0 => clipp at smallest
       possible value (depends on labels) = 0.0 => no clipping is applied
    
       Allowed values: : values listed above : float >= -1.0 : float > 0.0 :
       float > 0.0
    
       Default values: = native loss of solver chosen by option -S = -1.000
       = set by option -W = set by option -W
    
    -  ``P=c(1,[<size>])``
    -  ``P=c(2,[<number>])``
    -  ``P=c(3,[<radius>],[<subset_size>])``
    -  ``P=c(4,[<size>],[<reduce>],[<subset_size>])``
    -  ``P=c(5,[<size>],[<ignore_fraction>],[<subset_size>],[<covers>])``
    
       Selects the working set partition method.
    
       Meaning of specific values: = 0 => do not split the working sets = 1
       => split the working sets in random chunks using maximum of each
       chunk. Default values are: = 2000 = 2 => split the working sets in
       random chunks using of chunks. Default values are: = 10 = 3 => split
       the working sets by Voronoi subsets using . If [subset\_size] is set,
       a subset of this size is used to faster create the Voronoi partition.
       If subset\_size == 0, the entire data set is used. Default values
       are: = 1.000 = 0 = 4 => split the working sets by Voronoi subsets
       using . The optional controls whether a heuristic to reduce the
       number of cells is used. If [subset\_size] is set, a subset of this
       size is used to faster create the Voronoi partition. If subset\_size
       == 0, the entire data set is used. Default values are: = 2000 = 1 =
       20000 = 5 => devide the working sets into overlapping regions of size
       . The process of creating regions is stopped when \* samples have not
       been assigned to a region. These samples will then be assigned to the
       closest region. If is set, a subset of this size is used to find the
       regions. If subset\_size == 0, the entire data set is used. Finally,
       controls the number of times the process of finding regions is
       repeated. Default values are:. = 2000 = 0.5 = 20000 = 1
    
       Allowed values: : integer between 0 and 5 : positive integer :
       positive integer : positive real : positive integer : bool : positive
       integer
    
       Default values: = 0
    
    -  ``r=<seed>``
    
       Initializes the random number generator with .
    
       Meaning of specific values: = -1 => a random seed based on the
       internal timer is used
    
       Allowed values: : integer between -1 and 2147483647
    
       Default values: = -1
    
    -  ``s=c(<clipp>,[<stop_eps>])``
    
       Sets the value at which the loss is clipped in the solver to . The
       optional parameter sets the threshold in the stopping criterion of
       the solver.
    
       Meaning of specific values: = -1.0 => Depending on the solver type
       clipp either at the smallest possible value (depends on labels), or
       do not clipp. = 0.0 => no clipping is applied
    
       Allowed values: : -1.0 or float >= 0.0. In addition, if > 0.0, then
       must not be smaller than the largest absolute value of the samples. :
       float > 0.0
    
       Default values: = -1.0 = 0.0010
    
    -  ``S=c(<solver>,[<NNs>])``
    
       Selects the SVM solver and the number of nearest neighbors used in
       the working set selection strategy (2D-solvers only).
    
       Meaning of specific values: = 0 => kernel rule for classification = 1
       => LS-SVM with 2D-solver = 2 => HINGE-SVM with 2D-solver = 3 =>
       QUANTILE-SVM with 2D-solver = 4 => EXPECTILE-SVM with 2D-solver = 5
       => Your SVM solver implemented in template\_svm.\*
    
       Allowed values: : integer between 0 and 5 : integer between 0 and 100
    
       Default values: = 2 = depends on the solver
    
    -  ``T=<threads>``
    
       Sets the number of threads that are going to be used. Each thread is
       assigned to a logical processor on the system, so that the number of
       allowed threads is bounded by the number of logical processors. On
       systems with activated hyperthreading each physical core runs one
       thread, if does not exceed the number of physical cores. Since hyper-
       threads on the same core share resources, using more threads than
       cores does usually not increase the performance significantly, and
       may even decrease it.
    
       Meaning of specific values: = 0 => 4 threads are used (all physical
       cores run one thread) = -1 => 3 threads are used (all but one of the
       physical cores run one thread)
    
       Allowed values: : integer between -1 and 4
    
       Default values: = 0
    
    -  ``w=c(<neg_weight>,<pos_weight>)``
    -  ``w=c(<min_weight>,<max_weight>,<size>,[<geometric>,<swap>])``
    -  ``w=c(<weight_list>,[<swap>])``
    
       Sets values for the weights, solvers should be trained with. For
       solvers that do not have weights this option is ignored. The first
       variants sets a pair of values. The second variant computes a
       sequence of weights of length . The third variant takes the list of
       weights.
    
       Meaning of specific values: = 1 => is the negative weight and is the
       positive weight. > 1 => many pairs are computed, where the positive
       weights are between and and the negative weights are 1 - pos\_weight.
       Flag indicating whether the intermediate positive weights are
       geometrically or arithmetically distributed. Flag indicating whether
       the role of the positive and negative weights are interchanged.
    
       Allowed values: <... weight ...>: float > 0.0 and < 1.0 : integer > 0
       : bool : bool
    
       Default values: = 1.0 = 1.0 = 1 = 0 = 0
    
    -  ``W=<type>``
    
       Selects the working set selection method.
    
       Meaning of specific values: = 0 => take the entire data set = 1 =>
       multiclass 'all versus all' = 2 => multiclass 'one versus all' = 3 =>
       bootstrap with resamples of size
    
       Allowed values: : integer between 0 and 3
    
       Default values: = 0
    
    
    '''


def selectArgs():
    '''Arguments for SVM.select
    -  ``h=[<level>]``
    
       Displays all help messages.
    
       Meaning of specific values: = 0 => short help messages = 1 =>
       detailed help messages
    
       Allowed values: : 0 or 1
    
       Default values: = 0
    
    -  ``N=c(<class>,<constraint>)``
    
       Replaces the best validation error in the search for the best
       hyper-parameter combination by an NPL criterion, in which the best
       detection rate is searched for given the false alarm constraint on
       class .
    
       Allowed values: : -1 or 1 : float between 0.0 and 1.0
    
       Default values: Option is deactivated.
    
    -  ``R=<method>``
    
       Selects the method that produces decision functions from the
       different folds.
    
       Meaning of specific values: = 0 => select for best average validation
       error = 1 => on each fold select for best validation error
    
       Allowed values: : integer between 0 and 1
    
       Default values: = 1
    
    -  ``W=<number>``
    
       Restrict the search for the best hyper-parameters to weights with the
       number .
    
       Meaning of specific values: = 0 => all weights are considered.
    
       Default values: = 0
    
    
    '''


def testArgs():
    '''Arguments for SVM.test
    -  ``GPU=<gpus>``
    
       Sets the number of GPUs that are going to be used. Currently, there
       is no checking whether your system actually has many GPUs. In
       addition, the number of used threads is reduced to .
    
       Allowed values: : integer between 0 and ???
    
       Default values: = 0
    
       Unfortunately, this option is not activated for the binaries you are
       currently using. Install CUDA and recompile to activate this option.
    
    -  ``h=[<level>]``
    
       Displays all help messages.
    
       Meaning of specific values: = 0 => short help messages = 1 =>
       detailed help messages
    
       Allowed values: : 0 or 1
    
       Default values: = 0
    
    -  ``L=c(<loss>,[<neg_weight>,<pos_weight>])``
    
       Sets the loss that is used to compute empirical errors. The optional
       weights can only be set, if specifies a loss that has weights.
    
       Meaning of specific values: = 0 => binary classification loss = 1 =>
       multiclass class = 2 => least squares loss = 3 => weighted least
       squares loss = 5 => your own template loss
    
       Allowed values: : integer between 0 and 2 : float > 0.0 : float > 0.0
    
       Default values: = 0 = 1.0 = 1.0
    
    -  ``T=<threads>``
    
       Sets the number of threads that are going to be used. Each thread is
       assigned to a logical processor on the system, so that the number of
       allowed threads is bounded by the number of logical processors. On
       systems with activated hyperthreading each physical core runs one
       thread, if does not exceed the number of physical cores. Since hyper-
       threads on the same core share resources, using more threads than
       cores does usually not increase the performance significantly, and
       may even decrease it.
    
       Meaning of specific values: = 0 => 4 threads are used (all physical
       cores run one thread) = -1 => 3 threads are used (all but one of the
       physical cores run one thread)
    
       Allowed values: : integer between -1 and 4
    
       Default values: = 0
    
    -  ``v=c(<weighted>,<scenario>,[<npl_class>])``
    
       Sets the weighted vote method to combine decision functions from
       different folds. If = 1, then weights are computed with the help of
       the validation error, otherwise, equal weights are used. In the
       classification scenario, the decision function values are first
       transformed to -1 and +1, before a weighted vote is performed, in the
       regression scenario, the bare function values are used in the vote.
       In the weighted NPL scenario, the weights are computed according to
       the validation error on the samples with label , the rest is like in
       the classification scenario. can only be set for the NPL scenario.
    
       Meaning of specific values: = 0 => classification = 1 => regression =
       2 => NPL
    
       Allowed values: : 0 or 1 : integer between 0 and 2 : -1 or 1
    
       Default values: = 1 = 0 = 1
    
    -  ``o=<display_roc_style>``
    
       Sets a flag that decides, wheather classification errors are
       displayed by true positive and false positives.
    
       Allowed values: : 0 or 1
    
       Default values: : Depends on option -v
    
    
    '''


