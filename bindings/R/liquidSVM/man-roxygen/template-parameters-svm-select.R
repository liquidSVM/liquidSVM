#'@section Documentation for command-line parameters of svm-select:
#'The following parameters can be used as well:
#'\itemize{
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
#'\item \code{N=c(<class>,<constraint>)}\cr
#'\ifelse{latex}{\out{\medskip}}{\cr}
#'Replaces the best validation error in the search for the best hyper-parameter\cr
#'combination by an NPL criterion, in which the best detection rate is searched\cr
#'for given the false alarm constraint <constraint> on class <class>.\cr
#'\ifelse{latex}{\out{\medskip}}{\cr}
#'Allowed values:\cr
#'<class>: -1 or 1\cr
#'<constraint>: float between 0.0 and 1.0\cr
#'\ifelse{latex}{\out{\medskip}}{\cr}
#'Default values:\cr
#'Option is deactivated.\cr
#'\ifelse{latex}{\out{\medskip}}{\cr}
#'\ifelse{latex}{\out{\medskip}}{\cr}
#'\item \code{R=<method>}\cr
#'\ifelse{latex}{\out{\medskip}}{\cr}
#'Selects the method that produces decision functions from the different folds.\cr
#'\ifelse{latex}{\out{\medskip}}{\cr}
#'Meaning of specific values:\cr
#'<method> = 0  =>   select for best average validation error\cr
#'<method> = 1  =>   on each fold select for best validation error\cr
#'\ifelse{latex}{\out{\medskip}}{\cr}
#'Allowed values:\cr
#'<method>: integer between 0 and 1\cr
#'\ifelse{latex}{\out{\medskip}}{\cr}
#'Default values:\cr
#'<method> = 1\cr
#'\ifelse{latex}{\out{\medskip}}{\cr}
#'\ifelse{latex}{\out{\medskip}}{\cr}
#'\item \code{W=<number>}\cr
#'\ifelse{latex}{\out{\medskip}}{\cr}
#'Restrict the search for the best hyper-parameters to weights with the number\cr
#'<number>.\cr
#'\ifelse{latex}{\out{\medskip}}{\cr}
#'Meaning of specific values:\cr
#'<number> = 0  =>   all weights are considered.\cr
#'\ifelse{latex}{\out{\medskip}}{\cr}
#'Default values:\cr
#'<number> = 0\cr
#'\ifelse{latex}{\out{\medskip}}{\cr}
#'\ifelse{latex}{\out{\medskip}}{\cr}
#'}
