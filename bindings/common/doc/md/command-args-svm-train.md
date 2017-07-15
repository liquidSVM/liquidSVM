
    
* `f=c(<kind>,<number>,[<train_fraction>],[<neg_fraction>])`
    
    Selects the fold generation method and the number of folds. If <train_fraction>
    < 1.0, then the folds for training are generated from a subset with the
     specified size and the remaining samples are used for validation.
    Meaning of specific values:
    <kind> = 1  =>  each fold is a contiguous block
    <kind> = 2  =>  alternating fold assignmend
    <kind> = 3  =>  random
    <kind> = 4  =>  stratified random
    <kind> = 5  =>  random respecting group information of samples
    <kind> = 6  =>  random subset (<train_fraction> and <neg_fraction> required)
    
    Allowed values:
    <kind>:           integer between 1 and 6
    <number>:         integer >= 1
    <train_fraction>: float > 0.0 and <= 1.0
    <neg_fraction>:   float > 0.0 and < 1.0
    
    Default values:
    <kind>           = 3
    <number>         = 5
    <train_fraction> = 1.00
    
    
* `g=c(<size>,<min_gamma>,<max_gamma>,[<scale>])`
* `g=<gamma_list>`
    
    The first variant sets the size <size> of the gamma grid and its endpoints
    <min_gamma> and <max_gamma>.
    The second variant uses <gamma_list> for the gamma grid.
    
    Meaning of specific values:
    <scale>       Flag indicating whether <min_gamma> and <max_gamma> are scaled
                  based on the sample size, the dimension, and the diameter.
    
    Allowed values:
    <size>:       integer >= 1
    <min_gamma>:  float > 0.0
    <max_gamma>:  float > 0.0
    <scale>:      bool
    
    Default values:
    <size>        = 10
    <min_gamma>   = 0.200
    <max_gamma>   = 5.000
    <scale>       = 1
    
    
* `GPU=c(<use_gpus>,[<GPU_offset>])`
    
    Flag controlling whether the GPU support is used. If <use_gpus> = 1, then each
    CPU thread gets a thread on a GPU. In the case of multiple GPUs, these threads
    are uniformly distributed among the available GPUs. The optional <GPU_offset>
    is added to the CPU thread number before the GPU is added before distributing
    the threads to the GPUs. This makes it possible to avoid that two or more
    independent processes use the same GPU, if more than one GPU is available.
    
    Allowed values:
    <use_gpus>:   bool
    <use_offset>: non-negative integer.
    
    Default values:
    <gpus>       = 0
    <gpu_offset> = 0
    
    Unfortunately, this option is not activated for the binaries you are currently
    using. Install CUDA and recompile to activate this option.
    
    
* `h=[<level>]`
    
    Displays all help messages.
    
    Meaning of specific values:
    <level> = 0  =>  short help messages
    <level> = 1  =>  detailed help messages
    
    Allowed values:
    <level>: 0 or 1
    
    Default values:
    <level> = 0
    
    
* `i=c(<cold>,<warm>)`
    
    Selects the cold and warm start initialization methods of the solver. In
    general, this option should only be used in particular situations such as the
    implementation and testing of a new solver or when using the kernel cache.
    
    Meaning of specific values:
    For values between 0 and 6, both <cold> and <warm> have the same meaning as
    in Steinwart et al, 'Training SVMs without offset', JMLR 2011. These are:
     0      Sets all coefficients to zero.
     1      Sets all coefficients to C.
     2      Uses the coefficients of the previous solution.
     3      Multiplies all coefficients by C_new/C_old.
     4      Multiplies all unbounded SVs by C_new/C_old.
     5      Multiplies all coefficients by C_old/C_new.
     6      Multiplies all unbounded SVs by C_old/C_new.
    
    Allowed values:
    Depends on the solver, but the range of <cold> is always a subset of the range
    of <warm>.
    
    Default values:
    Depending on the solver, the (hopefully) most efficient method is chosen.
    
    
* `k=c(<type>,[aux-file],[<Tr_mm_Pr>,[<size_P>],<Tr_mm>,[<size>],<Va_mm_Pr>,<Va_mm>])`
    
    Selects the type of kernel and optionally the memory model for the kernel matrices.
    
    Meaning of specific values:
    <type>   = 0  =>   Gaussian RBF
    <type>   = 1  =>   Poisson
    <type>   = 3  =>   Experimental hierarchical Gauss kernel
    <aux_file>    =>   Name of the file that contains additional information for the
                       hierarchical Gauss kernel. Only this kernel type requires this option.
    <X_mm_Y> = 0  =>   not contiguously stored matrix
    <X_mm_Y> = 1  =>   contiguously stored matrix
    <X_mm_Y> = 2  =>   cached matrix
    <X_mm_Y> = 3  =>   no matrix stored
    <size_Y>      =>   size of kernel cache in MB
    Here, X=Tr stands for the training matrix and X=Va for the validation matrix. In
    both cases, Y=Pr stands for the pre-kernel matrix, which stores the distances
    between the samples. If <Tr_mm_Pr> is set, then the other three flags <X_mm_Y>
    need to be set, too. The values <sizeY> must only be set if a cache is chosen.
    NOTICE: Not all possible combinations are allowed.
    
    Allowed values:
    <type>:          integer between 0 and 3
    <X_mm_Y>:        integer between 0 and 3
    <size_Y>:        integer not smaller than 1
    
    Default values:
    <type>           = 0
    <X_mm_Y>         = 1
    <size_Y>         = 1024
    <size>           = 512
    
    
* `l=c(<size>,<min_lambda>,<max_lambda>,[<scale>])`
* `l=c(<lambda_list>,[<interpret_as_C>])`
    
    The first variant sets the size <size> of the lambda grid and its endpoints
    <min_lambda> and <max_lambda>.
    The second variant uses <lambda_list>, after ordering, for the lambda grid.
    
    Meaning of specific values:
    <scale>             Flag indicating whether <min_lambda> is internally
                        devided by the average number of samples per fold.
    <interpret_as_C>    Flag indicating whether the lambda list should be
                        interpreted as a list of C values
    
    Allowed values:
    <size>:             integer >= 1
    <min_lambda>:       float > 0.0
    <max_lambda>:       float > 0.0
    <scale>:            bool
    <interpret_as_C>:   bool
    
    Default values:
    <size>              = 10
    <min_lambda>        = 0.001
    <max_lambda>        = 0.100
    <scale>             = 1
    <scale>             = 0
    
    
* `L=c(<loss>,[<clipp>],[<neg_weight>,<pos_weight>])`
    
    Sets the loss that is used to compute empirical errors. The optional <clipp> value
    specifies where the predictions are clipped during validation. The optional weights
    can only be set if <loss> specifies a loss that has weights.
    
    Meaning of specific values:
    <loss> = 0  =>   binary classification loss
    <loss> = 2  =>   least squares loss
    <loss> = 3  =>   weighted least squares loss
    <loss> = 4  =>   pinball loss
    <loss> = 5  =>   hinge loss
    <loss> = 6  =>   your own template loss
    <clipp> = -1.0  =>   clipp at smallest possible value (depends on labels)
    <clipp> =  0.0  =>   no clipping is applied
    
    Allowed values:
    <loss>:       values listed above
    <neg_weight>: float >= -1.0
    <neg_weight>: float > 0.0
    <pos_weight>: float > 0.0
    
    Default values:
    <loss>       = native loss of solver chosen by option -S
    <clipp>      = -1.000
    <neg_weight> = <weight1> set by option -W
    <pos_weight> = <weight2> set by option -W
    
    
* `P=c(1,[<size>])`
* `P=c(2,[<number>])`
* `P=c(3,[<radius>],[<subset_size>])`
* `P=c(4,[<size>],[<reduce>],[<subset_size>])`
* `P=c(5,[<size>],[<ignore_fraction>],[<subset_size>],[<covers>])`
* `P=c(6,[<size>],[<reduce>],[<subset_size>],[<covers>],[<shrink_factor>])`
           [<max_width>] [<max_depth>]
    
    Selects the working set partition method.
    
    Meaning of specific values:
    <type> = 0  =>  do not split the working sets
    <type> = 1  =>  split the working sets in random chunks using maximum <size> of
                    each chunk.
                    Default values are:
                    <size>            = 2000
    <type> = 2  =>  split the working sets in random chunks using <number> of
                    chunks.
                    Default values are:
                    <size> = 10
    <type> = 3  =>  split the working sets into Voronoi subsets of radius <radius>.
                    If [subset_size] is set, a subset of this size is used to faster
                    create the Voronoi partition. If subset_size == 0, the entire
                    data set is used, otherwise, the radius is only approximately
                    ensured.
                    Default values are:
                    <radius>          = 1.000
                    <subset_size>     = 0
    <type> = 4  =>  split the working sets into Voronoi subsets of maximal size
                    <size>. The optional flag <reduce> controls whether a heuristic
                    to reduce the number of cells is used. If [subset_size] is set,
                    a subset of this size is used to faster create the Voronoi
                    partition. If subset_size == 0, the entire data set is used, 
                    otherwise, the maximal size is only approximately ensured.
                    Default values are:
                    <size>            = 2000
                    <reduce>          = 1
                    <subset_size>     = 50000
    <type> = 5  =>  devide the working sets into overlapping regions of maximal
                    size <size>. The process of creating regions is stopped when
                    <size> * <ignore_fraction> samples have not been assigned to
                    a region. These samples will then be assigned to the closest
                    region. If <subset_size> is set, a subset of this size is
                    used to find the regions. If subset_size == 0, the entire
                    data set is used. Finally, <covers> controls the number of
                    times the process of finding regions is repeated.
                    Default values are:.
                    <size>            = 2000
                    <ignore_fraction> = 0.5
                    <subset_size>     = 50000
                    <covers>          = 1
    <type> = 6  =>  split the working sets into Voronoi subsets of maximal size
                    <size>. The optional flag <reduce> controls whether a heuristic
                    to reduce the number of cells is used. If [subset_size] is set,
                    a subset of this size is used to faster create the Voronoi
                    partition. If subset_size == 0, the entire data set is used, 
                    otherwise, the maximal size is only approximately ensured.
                    Unlike for <type> = 4, the centers for the Voronoi partition are
                    found by a recursive tree approach, which in many cases may be
                    faster. <shrink_factor> describes by which factor the number of
                    samples should at least be decreased. The recursion is stoppend
                    when either <max_width> * <size> is greater than the current
                    working subset or the <max_tree_depth> is reached. For both
                    parameters, a value of 0 means that the corresponding condition
                    above is ignored.
                    Default values (so far, they are only a brave guess) are:
                    <size>            = 2000
                    <reduce>          = 1
                    <subset_size>     = 50000
                    <shrink_factor>   = 2.0000
                    <max_width>       = 20
                    <max_tree_depth>  = 4
    
    Allowed values:
    <type>:            integer between 0 and 6
    <size>:            positive integer
    <number>:          positive integer
    <radius>:          positive real
    <subset_size>:     non-negative integer
    <reduce>:          bool
    <covers>:          positive integer
    <shrink_factor>:   real > 1.0
    <max_width>:       non-negative integer
    <max_tree_depth>:  non-negative integer
    
    Default values:
    <type>             = 0
    
    
* `r=<seed>`
    
    Initializes the random number generator with <seed>.
    
    Meaning of specific values:
    <seed> = -1  =>  a random seed based on the internal timer is used
    
    Allowed values:
    <seed>: integer between -1 and 2147483647
    
    Default values:
    <seed> = -1
    
    
* `s=c(<clipp>,[<stop_eps>])`
    
    Sets the value at which the loss is clipped in the solver to <value>. The
    optional parameter <stop_eps> sets the threshold in the stopping criterion
    of the solver.
    
    Meaning of specific values:
    <clipp> = -1.0  =>   Depending on the solver type clipp either at the
                         smallest possible value (depends on labels), or
                         do not clipp.
    <clipp> = 0.0   =>   no clipping is applied
    
    Allowed values:
    <clipp>:    -1.0 or float >= 0.0.
                In addition, if <clipp> > 0.0, then <clipp> must not be smaller
                than the largest absolute value of the samples.
    <stop_eps>: float > 0.0
    
    Default values:
    <clipp>     = -1.0
    <stop_eps>  = 0.0010
    
    
* `S=c(<solver>,[<NNs>])`
    
    Selects the SVM solver <solver> and the number <NNs> of nearest neighbors used
    in the working set selection strategy (2D-solvers only).
    
    Meaning of specific values:
    <solver> = 0  =>  kernel rule for classification
    <solver> = 1  =>  LS-SVM with 2D-solver
    <solver> = 2  =>  HINGE-SVM with 2D-solver
    <solver> = 3  =>  QUANTILE-SVM with 2D-solver
    <solver> = 4  =>  EXPECTILE-SVM with 2D-solver
    <solver> = 5  =>  Your SVM solver implemented in template_svm.*
    
    Allowed values:
    <solver>: integer between 0 and 5
    <NNs>:    integer between 0 and 100
    
    Default values:
    <solver> = 2
    <NNs>    = depends on the solver
    
    
* `T=c(<threads>,[<thread_id_offset>])`
    
    Sets the number of threads that are going to be used. Each thread is
    assigned to a logical processor on the system, so that the number of
    allowed threads is bounded by the number of logical processors. On
    systems with activated hyperthreading each physical core runs one thread,
    if <threads> does not exceed the number of physical cores. Since hyper-
    threads on the same core share resources, using more threads than cores
    does usually not increase the performance significantly, and may even
    decrease it. The optional <thread_id_offset> is added before distributing
    the threads to the cores. This makes it possible to avoid that two or more
    independent processes use the same physical cores.
    Example: To run 2 processes with 3 threads each on a 6-core system call
    the first process with -T 3 0 and the second one with -T 3 3 .
    
    Meaning of specific values:
    <threads> =  0   =>   4 threads are used (all physical cores run one thread)
    <threads> = -1   =>   3 threads are used (all but one of the physical cores
                                              run one thread)
    
    Allowed values:
    <threads>:          integer between -1 and 4
    <thread_id_offset>: integer between  0 and 4
    
    Default values:
    <threads>          = 0
    <thread_id_offset> = 0
    
    
* `w=c(<neg_weight>,<pos_weight>)`
* `w=c(<min_weight>,<max_weight>,<size>,[<geometric>,<swap>])`
* `w=c(<weight_list>,[<swap>])`
    
    Sets values for the weights, solvers should be trained with. For solvers
    that do not have weights this option is ignored.
    The first variants sets a pair of values.
    The second variant computes a sequence of weights of length <size>.
    The third variant takes the list of weights.
    
    Meaning of specific values:
    size> = 1      =>  <weight1> is the negative weight and <weight2> is the
                       positive weight.
    <size> > 1     =>  <size> many pairs are computed, where the positive
                       weights are between <min_weight> and <max_weight> and
                       the negative weights are 1 - pos_weight.
    <geometric>        Flag indicating whether the intermediate positive
                       weights are geometrically or arithmetically distributed.
    <swap>             Flag indicating whether the role of the positive and
                       negative weights are interchanged.
    
    Allowed values:
    <... weight ...>:  float > 0.0 and < 1.0
    <size>:      integer > 0
    <geometric>: bool
    <swap>:      bool
    
    Default values:
    <weight1>   = 1.0
    <weight2>   = 1.0
    <size>      = 1
    <geometric> = 0
    <swap>      = 0
    
    
* `W=<type>`
    
    Selects the working set selection method.
    
    Meaning of specific values:
    <type> = 0  =>  take the entire data set
    <type> = 1  =>  multiclass 'all versus all'
    <type> = 2  =>  multiclass 'one versus all'
    <type> = 3  =>  bootstrap with <number> resamples of size <size>
    
    Allowed values:
    <type>: integer between 0 and 3
    
    Default values:
    <type>    = 0
    
    
