#'@section Documentation for command-line parameters of svm-test:
#'The following parameters can be used as well:
#'\itemize{
#'\ifelse{latex}{\out{\medskip}}{\cr}
#'\ifelse{latex}{\out{\medskip}}{\cr}
#'\item \code{GPU=c(<use_gpus>,[<GPU_offset>])}\cr
#'\ifelse{latex}{\out{\medskip}}{\cr}
#'Flag controlling whether the GPU support is used. If <use_gpus> = 1, then each\cr
#'CPU thread gets a thread on a GPU. In the case of multiple GPUs, these threads\cr
#'are uniformly distributed among the available GPUs. The optional <GPU_offset>\cr
#'is added to the CPU thread number before the GPU is added before distributing\cr
#'the threads to the GPUs. This makes it possible to avoid that two or more\cr
#'independent processes use the same GPU, if more than one GPU is available.\cr
#'\ifelse{latex}{\out{\medskip}}{\cr}
#'Allowed values:\cr
#'<use_gpus>:   bool\cr
#'<use_offset>: non-negative integer.\cr
#'\ifelse{latex}{\out{\medskip}}{\cr}
#'Default values:\cr
#'<gpus>       = 0\cr
#'<gpu_offset> = 0\cr
#'\ifelse{latex}{\out{\medskip}}{\cr}
#'Unfortunately, this option is not activated for the binaries you are currently\cr
#'using. Install CUDA and recompile to activate this option.\cr
#'\ifelse{latex}{\out{\medskip}}{\cr}
#'\ifelse{latex}{\out{\medskip}}{\cr}
#'\item \code{h=[<level>]}\cr
#'\ifelse{latex}{\out{\medskip}}{\cr}
#'Displays all help messages.\cr
#'\ifelse{latex}{\out{\medskip}}{\cr}
#'Meaning of specific values:\cr
#'<level> = 0  =>  short help messages\cr
#'<level> = 1  =>  detailed help messages\cr
#'\ifelse{latex}{\out{\medskip}}{\cr}
#'Allowed values:\cr
#'<level>: 0 or 1\cr
#'\ifelse{latex}{\out{\medskip}}{\cr}
#'Default values:\cr
#'<level> = 0\cr
#'\ifelse{latex}{\out{\medskip}}{\cr}
#'\ifelse{latex}{\out{\medskip}}{\cr}
#'\item \code{L=c(<loss>,[<neg_weight>,<pos_weight>])}\cr
#'\ifelse{latex}{\out{\medskip}}{\cr}
#'Sets the loss that is used to compute empirical errors. The optional weights can\cr
#'only be set, if <loss> specifies a loss that has weights.\cr
#'\ifelse{latex}{\out{\medskip}}{\cr}
#'Meaning of specific values:\cr
#'<loss> = 0  =>   binary classification loss\cr
#'<loss> = 1  =>   multiclass class\cr
#'<loss> = 2  =>   least squares loss\cr
#'<loss> = 3  =>   weighted least squares loss\cr
#'<loss> = 6  =>   your own template loss\cr
#'\ifelse{latex}{\out{\medskip}}{\cr}
#'Allowed values:\cr
#'<loss>: integer between 0 and 2\cr
#'<neg_weight>: float > 0.0\cr
#'<pos_weight>: float > 0.0\cr
#'\ifelse{latex}{\out{\medskip}}{\cr}
#'Default values:\cr
#'<loss> = 0\cr
#'<neg_weight> = 1.0\cr
#'<pos_weight> = 1.0\cr
#'\ifelse{latex}{\out{\medskip}}{\cr}
#'\ifelse{latex}{\out{\medskip}}{\cr}
#'\item \code{T=c(<threads>,[<thread_id_offset>])}\cr
#'\ifelse{latex}{\out{\medskip}}{\cr}
#'Sets the number of threads that are going to be used. Each thread is\cr
#'assigned to a logical processor on the system, so that the number of\cr
#'allowed threads is bounded by the number of logical processors. On\cr
#'systems with activated hyperthreading each physical core runs one thread,\cr
#'if <threads> does not exceed the number of physical cores. Since hyper-\cr
#'threads on the same core share resources, using more threads than cores\cr
#'does usually not increase the performance significantly, and may even\cr
#'decrease it. The optional <thread_id_offset> is added before distributing\cr
#'the threads to the cores. This makes it possible to avoid that two or more\cr
#'independent processes use the same physical cores.\cr
#'Example: To run 2 processes with 3 threads each on a 6-core system call\cr
#'the first process with -T 3 0 and the second one with -T 3 3 .\cr
#'\ifelse{latex}{\out{\medskip}}{\cr}
#'Meaning of specific values:\cr
#'<threads> =  0   =>   4 threads are used (all physical cores run one thread)\cr
#'<threads> = -1   =>   3 threads are used (all but one of the physical cores\cr
#'run one thread)\cr
#'\ifelse{latex}{\out{\medskip}}{\cr}
#'Allowed values:\cr
#'<threads>:          integer between -1 and 4\cr
#'<thread_id_offset>: integer between  0 and 4\cr
#'\ifelse{latex}{\out{\medskip}}{\cr}
#'Default values:\cr
#'<threads>          = 0\cr
#'<thread_id_offset> = 0\cr
#'\ifelse{latex}{\out{\medskip}}{\cr}
#'\ifelse{latex}{\out{\medskip}}{\cr}
#'\item \code{v=c(<weighted>,<scenario>,[<npl_class>])}\cr
#'\ifelse{latex}{\out{\medskip}}{\cr}
#'Sets the weighted vote method to combine decision functions from different\cr
#'folds. If <weighted> = 1, then weights are computed with the help of the\cr
#'validation error, otherwise, equal weights are used. In the classification\cr
#'scenario, the decision function values are first transformed to -1 and +1,\cr
#'before a weighted vote is performed, in the regression scenario, the bare\cr
#'function values are used in the vote. In the weighted NPL scenario, the weights\cr
#'are computed according to the validation error on the samples with label\cr
#'<npl_class>, the rest is like in the classification scenario.\cr
#'<npl_class> can only be set for the NPL scenario.\cr
#'\ifelse{latex}{\out{\medskip}}{\cr}
#'Meaning of specific values:\cr
#'<scenario> = 0  =>   classification\cr
#'<scenario> = 1  =>   regression\cr
#'<scenario> = 2  =>   NPL\cr
#'\ifelse{latex}{\out{\medskip}}{\cr}
#'Allowed values:\cr
#'<weighted>: 0 or 1\cr
#'<scenario>: integer between 0 and 2\cr
#'<npl_class>: -1 or 1\cr
#'\ifelse{latex}{\out{\medskip}}{\cr}
#'Default values:\cr
#'<weighted> = 1\cr
#'<scenario> = 0\cr
#'<npl_class> = 1\cr
#'\ifelse{latex}{\out{\medskip}}{\cr}
#'\ifelse{latex}{\out{\medskip}}{\cr}
#'\item \code{o=<display_roc_style>}\cr
#'\ifelse{latex}{\out{\medskip}}{\cr}
#'Sets a flag that decides, wheather classification errors are displayed by\cr
#'true positive and false positives.\cr
#'\ifelse{latex}{\out{\medskip}}{\cr}
#'Allowed values:\cr
#'<display_roc_style>: 0 or 1\cr
#'\ifelse{latex}{\out{\medskip}}{\cr}
#'Default values:\cr
#'<display_roc_style>: Depends on option -v\cr
#'\ifelse{latex}{\out{\medskip}}{\cr}
#'\ifelse{latex}{\out{\medskip}}{\cr}
#'}
