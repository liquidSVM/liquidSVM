
    
* `h=[<level>]`
    
    Displays all help messages.
    
    Meaning of specific values:
    <level> = 0  =>  short help messages
    <level> = 1  =>  detailed help messages
    
    Allowed values:
    <level>: 0 or 1
    
    Default values:
    <level> = 0
    
    
* `N=c(<class>,<constraint>)`
    
    Replaces the best validation error in the search for the best hyper-parameter
    combination by an NPL criterion, in which the best detection rate is searched
    for given the false alarm constraint <constraint> on class <class>.
    
    Allowed values:
    <class>: -1 or 1
    <constraint>: float between 0.0 and 1.0
    
    Default values:
    Option is deactivated.
    
    
* `R=<method>`
    
    Selects the method that produces decision functions from the different folds.
    
    Meaning of specific values:
    <method> = 0  =>   select for best average validation error
    <method> = 1  =>   on each fold select for best validation error
    
    Allowed values:
    <method>: integer between 0 and 1
    
    Default values:
    <method> = 1
    
    
* `W=<number>`
    
    Restrict the search for the best hyper-parameters to weights with the number
    <number>.
    
    Meaning of specific values:
    <number> = 0  =>   all weights are considered.
    
    Default values:
    <number> = 0
    
    
