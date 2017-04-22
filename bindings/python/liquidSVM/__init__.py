# Copyright 2015-2017 Philipp Thomann
#
# This file is part of liquidSVM.
#
# liquidSVM is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# liquidSVM is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
# 
# You should have received a copy of the GNU Affero General Public License
# along with liquidSVM. If not, see <http://www.gnu.org/licenses/>.
#
'''liquidSVM for Python

liquidSVM is a package written in C++ that
provides SVM-type solvers for various classification and regression tasks.
Because of a fully integrated hyper-parameter selection, very carefully implemented solvers,
multi-threading and GPU support,
and several built-in data decomposition strategies  it provides unprecedented speed
for small training sizes as well as for data sets of tens of millions of samples.

To install use

> pip install --user --upgrade liquidSVM

Then you can use it like:

>>> from liquidSVM import *
>>> model = mcSVM(iris, iris_labs, display=1,threads=2)
>>> result, err = model.test(iris, iris_labs)
>>> result = model.predict(iris)

For more information see the README and the demo notebook.

@author: Ingo Steinwart and Philipp Thomann
'''

import numpy as np
import numpy.ctypeslib as npct
import ctypes as ct
import os, sysconfig, glob
import pkg_resources



__all__ = ["SVM","lsSVM","mcSVM","qtSVM","exSVM","nplSVM","rocSVM","iris","iris_labs","LiquidData","doc"]

iris = np.array([5.1,4.9,4.7,4.6,5,5.4,4.6,5,4.4,4.9,5.4,4.8,4.8,4.3,5.8,5.7,5.4,5.1,5.7,5.1,5.4,5.1,4.6,5.1,4.8,5,5,5.2,5.2,4.7,4.8,5.4,5.2,5.5,4.9,5,5.5,4.9,4.4,5.1,5,4.5,4.4,5,5.1,4.8,5.1,4.6,5.3,5,7,6.4,6.9,5.5,6.5,5.7,6.3,4.9,6.6,5.2,5,5.9,6,6.1,5.6,6.7,5.6,5.8,6.2,5.6,5.9,6.1,6.3,6.1,6.4,6.6,6.8,6.7,6,5.7,5.5,5.5,5.8,6,5.4,6,6.7,6.3,5.6,5.5,5.5,6.1,5.8,5,5.6,5.7,5.7,6.2,5.1,5.7,6.3,5.8,7.1,6.3,6.5,7.6,4.9,7.3,6.7,7.2,6.5,6.4,6.8,5.7,5.8,6.4,6.5,7.7,7.7,6,6.9,5.6,7.7,6.3,6.7,7.2,6.2,6.1,6.4,7.2,7.4,7.9,6.4,6.3,6.1,7.7,6.3,6.4,6,6.9,6.7,6.9,5.8,6.8,6.7,6.7,6.3,6.5,6.2,5.9,3.5,3,3.2,3.1,3.6,3.9,3.4,3.4,2.9,3.1,3.7,3.4,3,3,4,4.4,3.9,3.5,3.8,3.8,3.4,3.7,3.6,3.3,3.4,3,3.4,3.5,3.4,3.2,3.1,3.4,4.1,4.2,3.1,3.2,3.5,3.6,3,3.4,3.5,2.3,3.2,3.5,3.8,3,3.8,3.2,3.7,3.3,3.2,3.2,3.1,2.3,2.8,2.8,3.3,2.4,2.9,2.7,2,3,2.2,2.9,2.9,3.1,3,2.7,2.2,2.5,3.2,2.8,2.5,2.8,2.9,3,2.8,3,2.9,2.6,2.4,2.4,2.7,2.7,3,3.4,3.1,2.3,3,2.5,2.6,3,2.6,2.3,2.7,3,2.9,2.9,2.5,2.8,3.3,2.7,3,2.9,3,3,2.5,2.9,2.5,3.6,3.2,2.7,3,2.5,2.8,3.2,3,3.8,2.6,2.2,3.2,2.8,2.8,2.7,3.3,3.2,2.8,3,2.8,3,2.8,3.8,2.8,2.8,2.6,3,3.4,3.1,3,3.1,3.1,3.1,2.7,3.2,3.3,3,2.5,3,3.4,3,1.4,1.4,1.3,1.5,1.4,1.7,1.4,1.5,1.4,1.5,1.5,1.6,1.4,1.1,1.2,1.5,1.3,1.4,1.7,1.5,1.7,1.5,1,1.7,1.9,1.6,1.6,1.5,1.4,1.6,1.6,1.5,1.5,1.4,1.5,1.2,1.3,1.4,1.3,1.5,1.3,1.3,1.3,1.6,1.9,1.4,1.6,1.4,1.5,1.4,4.7,4.5,4.9,4,4.6,4.5,4.7,3.3,4.6,3.9,3.5,4.2,4,4.7,3.6,4.4,4.5,4.1,4.5,3.9,4.8,4,4.9,4.7,4.3,4.4,4.8,5,4.5,3.5,3.8,3.7,3.9,5.1,4.5,4.5,4.7,4.4,4.1,4,4.4,4.6,4,3.3,4.2,4.2,4.2,4.3,3,4.1,6,5.1,5.9,5.6,5.8,6.6,4.5,6.3,5.8,6.1,5.1,5.3,5.5,5,5.1,5.3,5.5,6.7,6.9,5,5.7,4.9,6.7,4.9,5.7,6,4.8,4.9,5.6,5.8,6.1,6.4,5.6,5.1,5.6,6.1,5.6,5.5,4.8,5.4,5.6,5.1,5.1,5.9,5.7,5.2,5,5.2,5.4,5.1,0.2,0.2,0.2,0.2,0.2,0.4,0.3,0.2,0.2,0.1,0.2,0.2,0.1,0.1,0.2,0.4,0.4,0.3,0.3,0.3,0.2,0.4,0.2,0.5,0.2,0.2,0.4,0.2,0.2,0.2,0.2,0.4,0.1,0.2,0.2,0.2,0.2,0.1,0.2,0.2,0.3,0.3,0.2,0.6,0.4,0.3,0.2,0.2,0.2,0.2,1.4,1.5,1.5,1.3,1.5,1.3,1.6,1,1.3,1.4,1,1.5,1,1.4,1.3,1.4,1.5,1,1.5,1.1,1.8,1.3,1.5,1.2,1.3,1.4,1.4,1.7,1.5,1,1.1,1,1.2,1.6,1.5,1.6,1.5,1.3,1.3,1.3,1.2,1.4,1.2,1,1.3,1.2,1.3,1.3,1.1,1.3,2.5,1.9,2.1,1.8,2.2,2.1,1.7,1.8,1.8,2.5,2,1.9,2.1,2,2.4,2.3,1.8,2.2,2.3,1.5,2.3,2,2,1.8,2.1,1.8,1.8,1.8,2.1,1.6,1.9,2,2.2,1.5,1.4,2.3,2.4,1.8,1.8,2.1,2.4,2.3,1.9,2.3,2.5,2.3,1.9,2,2.3,1.8]).reshape(150,4)
iris_labs = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3])

# Load the library as _libliquidSVM.
_filepath = os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
#print(_filepath)
# Why the underscore (_) in front of _libliquidSVM below?
# To mimimise namespace pollution -- see PEP 8 (www.python.org).
#_libliquidSVM = npct.load_library('impl.so', _filepath)
#_libliquidSVM = npct.load_library('libliquidsvm', _filepath)
for loc in [_filepath,
            "/home/thomapp/liquidSVM/bindings/python/venvs/py3/lib/python3.4/site-packages/liquidSVM-0.5-py3.4-linux-x86_64.egg",
            "/home/thomapp/opt/anaconda/lib/python2.7/site-packages/liquidSVM-0.5-py2.7-linux-x86_64.egg",
            "/home/thomapp/opt/anaconda/envs/anacondaPy3/lib/python3.6/site-packages/liquidSVM-0.5-py3.6-linux-x86_64.egg"]:
    try:
        # print('Trying to load from: '+loc)
        thenames = glob.glob(loc + '/liquidSVM*'+sysconfig.get_config_var('SO'))
        if len(thenames) == 0:
            continue
        _libliquidSVM = npct.load_library(os.path.basename(thenames[0]), loc)
        # print('Found it!')
        break
    except:
        # print("Could not load!")
        pass


TdoubleP = npct.ndpointer(dtype = np.double)

_set_param = _libliquidSVM.liquid_svm_set_param
_set_param.argtypes = [ct.c_int, ct.c_char_p, ct.c_char_p]

_get_param = _libliquidSVM.liquid_svm_get_param
_get_param.argtypes = [ct.c_int, ct.c_char_p]
_get_param.restype = ct.c_char_p

_get_config_line = _libliquidSVM.liquid_svm_get_config_line
_get_config_line.argtypes = [ct.c_int, ct.c_int]
_get_config_line.restype = ct.c_char_p

_libliquidSVM.liquid_svm_init.argtypes = [TdoubleP, ct.c_int, ct.c_int, TdoubleP]
_libliquidSVM.liquid_svm_init.restype  = ct.c_int

_libliquidSVM.liquid_svm_train.argtypes = [ct.c_int, ct.c_int, ct.POINTER(ct.c_char_p)]
_libliquidSVM.liquid_svm_train.restype  = ct.POINTER(ct.c_double)

_libliquidSVM.liquid_svm_select.argtypes = [ct.c_int, ct.c_int, ct.POINTER(ct.c_char_p)]
_libliquidSVM.liquid_svm_select.restype  = ct.POINTER(ct.c_double)

_libliquidSVM.liquid_svm_test.argtypes = [ct.c_int, ct.c_int, ct.POINTER(ct.c_char_p), TdoubleP, ct.c_int, ct.c_int, TdoubleP, ct.POINTER(ct.POINTER(ct.c_double))]
_libliquidSVM.liquid_svm_test.restype  = ct.POINTER(ct.c_double)

_libliquidSVM.liquid_svm_clean.argtypes = [ct.c_int]

def makeArgs(kwargs, default={}, defaultLine=None):
    """This is an internal helper function to put all arguments needed
    for the train/select/test phases in core-liquidSVM.

    Parameters
    ----------
    kwargs : dict
        the configurations that shall be added.
    default :
        (Default value = {})
    defaultLine :
        (Default value = None)

    Returns
    -------

    
    """
    ret = ["liquidSVM.py"]
    ret.extend(defaultLine.split(' ')[1:])
    for name in default:
        if not name in kwargs: kwargs[name] = default[name]
    for name in kwargs:
        ret.append("-"+name)
        value = kwargs[name]
        if isinstance(value, (list,tuple)):
            ret.extend(value)
        else:
            ret.append(value)
    return [str(x).encode("utf-8") for x in ret]

def convertTable(arr):
    """This is an internal helper function to convert the ad-hoc matrices into numpy matrices.

    Parameters
    ----------
    arr : np.array
        the flat array to convert

    Returns
    -------

    
    """
    rows = int(arr[0])
    cols = int(arr[1])
    raw = npct.as_array(arr, (2+rows*cols,))
    return raw[2:(2+rows*cols)].reshape( (rows,cols) )

class SVM:
    """The base class for all SVM learning scenarios.
    This should usually not be used directly.

    If no scenario is specified, it will be set to LS.

    Parameters
    ----------
    data : np.array or LiquidData or Bunch or str
        the data to train on. If it is an `np.array` then
        `labs` have to be provided as well.
        If it is an instance of `LiquidData` then data.train.data
        and data.train.target will be used and data.test will be
        automagically be tested after training and selection.
        If it is a `str` then `LiquidData(data)` will be used.
        If it is a Bunch, then `data.data` and `data.target` will be used.
    labs : np.array (1-dim)
        the labels. If this is `None` then the labels
        have to be provided in the `data` argument.
    **kwargs : dict
        Configuration arguments, can be `threads=1`, `display=1`, `grid_choice=1`, etc.
        For more information see: `?doc.configuration`


    Attributes
    ----------
    cookie : int
        the internal cookie of the C++ SVM
    lastResult : (np.array , np.array)
        after each `test` the result and errors will be kept here.


    """

    def __init__(self, data, labs=None, **kwargs):
        if labs is None:
            if isinstance(data, str):
                data = LiquidData(data)
            if isinstance(data, LiquidData):
                self.autoTestData = data.test
                data = data.train
            if hasattr(data, 'target') and hasattr(data, 'data'):
                labs = data.target
                data = data.data
            else:
                raise ValueError('No labels have been specified!')
        n = data.shape[0]
        self.dim = data.shape[1] if len(data.shape)>=2 else 1
        self.data = np.asarray(data, dtype=np.double).copy()
        self.labs = np.asarray(labs, dtype=np.double).copy()
        self.predictCols = 0
        
        self.cookie = _libliquidSVM.liquid_svm_init(self.data, n, self.dim, self.labs)
        for name in kwargs:
            self.set(name, kwargs[name])
        if not len(self.get("SVM_TYPE")):
            self.set("SCENARIO", "LS")
    
    def train(self, **kwargs):
        """Trains all SVMs for all tasks/cells/hyper-parameters.
        This should only be used by experts.

        Parameters
        ----------
        **kwargs :dict
            The command-line parameters of svm-train can be given here
            in dictionary form, e.g. `d=1` instead of `-d 1`
            For detailed information see: `?doc.trainArgs`

        Returns
        -------
        np.array
            all validation errors and technical details of the training phase
        
        """
        argv = makeArgs(kwargs, defaultLine=self.configLine(1))
        #print(argv)
        TypeArgv = ct.c_char_p * len(argv)
        argv_s = TypeArgv(*argv)
        err =_libliquidSVM.liquid_svm_train(ct.c_int(self.cookie), len(argv), argv_s)
        if not err:
            raise Exception("Problem with training of a liquidSVM model.")
        self.errTrain = convertTable(err)
        return self.errTrain
    
    def select(self, **kwargs):
        """Selects the best of all SVMs for all tasks/cells..
        This should only be used by experts.

        Parameters
        ----------
        **kwargs : dict
            The command-line parameters of svm-select can be given here
            in dictionary form, e.g. `d=1` instead of `-d 1`
            For detailed information see: `?doc.selectArgs`

        Returns
        -------
        np.array
            all selected validation errors and technical details of the training phase

        """
        argv = makeArgs(kwargs, defaultLine=self.configLine(2))
        TypeArgv = ct.c_char_p * len(argv)
        argv_s = TypeArgv(*argv)
        err = _libliquidSVM.liquid_svm_select(ct.c_int(self.cookie), len(argv), argv_s)
        if not err:
            raise Exception("Problem with selecting of a liquidSVM model.")
        self.errSelect = convertTable(err)
        if hasattr(self, 'autoTestData'):
            self.test(self.autoTestData)
        return self.errSelect
    
    def test(self, test_data, test_labs=None, **kwargs):
        """Predicts labels for `test_data` and if applicable compares to test_labs.

        Parameters
        ----------
        test_data : np.array or LiquidData or Bunch
            the data to predict for. If it is an `np.array` then
            `labs` have to be provided as well.
            If it is an instance of `LiquidData` then data.test.data
            and data.test.target will be used.
            If it is a Bunch, then `data.data` and `data.target` will be used.
        test_labs : np.array (1-dim)
            the ground truth labels. If this is `None` and the labels
            are not provided in the `data` argument then only prediction will
            be performed.
            (Default value = None)
        **kwargs : dict
            These only should be used by experts.
            The command-line parameters of svm-test can be given here
            in dictionary form, e.g. `d=1` instead of `-d 1`.
            For detailed information see: `?doc.testArgs`


        Returns
        -------
        ( np.array , np.array)
            The first return argument is an array of the results.
            The number of columns depends on the learning scenario.
            The second return argument gives the errors if labels were provided,
            or is empty else. The number of rows depends on the learning scenario.
        """
        if test_labs is None:
            if isinstance(test_data, LiquidData):
                test_data = test_data.test
            if hasattr(test_data, 'target') and hasattr(test_data, 'data'):
                test_labs = test_data.target
                test_data = test_data.data
        n = test_data.shape[0]
        dim = test_data.shape[1] if len(test_data.shape)>=2 else 1
        test_data = np.asarray(test_data, dtype=np.double).copy()
        test_labs = np.asarray(test_labs, dtype=np.double).copy()
        argv = makeArgs(kwargs, defaultLine=self.configLine(3))
        TypeArgv = ct.c_char_p * len(argv)
        argv_s = TypeArgv(*argv)
        
        errors_ret = ct.pointer(ct.POINTER(ct.c_double)())
        
        result = _libliquidSVM.liquid_svm_test(ct.c_int(self.cookie), len(argv), argv_s, test_data, n, dim, test_labs, errors_ret)
        if not result:
            raise Exception("Problem with testing of a liquidSVM model.")
        self.lastResult = convertTable(result), convertTable(errors_ret.contents)
        return self.lastResult
        
    def predict(self, test_data, **kwargs):
        """Predicts labels for `test_data`.

        Parameters
        ----------
        test_data : np.array
            The features for which prediction should be estimated.
            
        **kwargs : dict
            Passed to `liquidSVM.test`

        Returns
        -------
        np.array
            the predictions for every test sample.
            It has the same number of rows as `test_data`.
            The number of columns depends on the learning scenario.
        
        """
        return self.test(test_data, np.zeros(test_data.shape[0]), **kwargs)[0][:,self.predictCols]
    
    def get(self, name):
        """Gets the value of a liquidSVM-configuration parameter
        For more information see: `?doc.configuration`

        Parameters
        ----------
        name : str
            

        Returns
        -------

        
        """
        return _libliquidSVM.liquid_svm_get_param(ct.c_int(self.cookie), ct.c_char_p(name.upper().encode('UTF-8')))
    def set(self, name, value):
        """Sets the value of a liquidSVM-configuration parameter
        For more information see: `?doc.configuration`

        Parameters
        ----------
        name : str
            
        value : any
            This will be converted into a str, joining by spaces if needed.

        Returns
        -------
        self
        
        """
        if isinstance(value, (list,tuple)):
            value = " ".join(map(str,value))
        if isinstance(value, bool):
            value = int(value)
        if name == "useCells":
            name = 'PARTITION_CHOICE'
            value = 6 if value else 0
        _libliquidSVM.liquid_svm_set_param(ct.c_int(self.cookie), ct.c_char_p(name.upper().encode('UTF-8')), ct.c_char_p(str(value).encode('UTF-8')))
        return self
    def configLine(self, stage):
        """Internal function to get the command-line like parameters for the different stages.

        Parameters
        ----------
        stage : int
            

        Returns
        -------
        str
        
        """
        return _libliquidSVM.liquid_svm_get_config_line(ct.c_int(self.cookie), ct.c_int(stage)).decode("utf-8")
    
    def clean(self):
        """Force to release internal C++ memory. After that, this SVM cannot be used any more."""
        if self.cookie >= 0:
            _libliquidSVM.liquid_svm_clean(ct.c_int(self.cookie))
        self.cookie = -1
    def __del__(self):
        self.clean()

class lsSVM(SVM):
    """This class performs non-parametric least squares regression using SVMs.
    The tested estimators are therefore estimating the conditional means of Y given X.

    Parameters
    ----------
    see `?SVM` and `?doc.configuration`
    """
    def __init__(self, data, labs=None, **kwargs):
        SVM.__init__(self, data, labs, scenario="LS", **kwargs)
        self.train()
        self.select()

class mcSVM(SVM):
    """This class is intended for both binary and multiclass classification.
    The binary classification is treated by an SVM solver for the classical hinge loss, and
    for the multiclass case, one-verus-all and all-versus-all reductions to binary classification
    for the hinge and the least squares loss are provided.
    The error of the very first task is the overall classification error.

    Parameters
    ----------
    mcType : str or int
        The multi-class classification scheme: "AvA_hinge","OvA_ls","OvA_hinge", or "AvA_ls".

    others: see `?SVM` and `?doc.configuration`
    """
    def __init__(self, data, labs=None, mcType="AvA_hinge", **kwargs):
        SVM.__init__(self, data, labs, scenario="MC "+mcType, **kwargs)
        self.train()
        self.select()

class qtSVM(SVM):
    """This class performs non-parametric and quantile regression using SVMs. The tested estimators are therefore
    estimating the conditional tau-quantiles of Y given X. By default, estimators for five different tau values are
    computed.

    Parameters
    ----------
    weights : arr of doubles
        The list of quantiles that should be estimated.
        (Default value = [0.05,0.1,0.5,0.9,0.95])

    others: see `?SVM` and `?doc.configuration`

    """
    def __init__(self, data, labs=None, weights=[0.05,0.1,0.5,0.9,0.95], **kwargs):
        SVM.__init__(self, data, labs, scenario="QT", **kwargs)
        self.set("WEIGHTS",weights)
        self.weights = weights
        self.predictCols = range(len(weights))
        self.train()
        self.select()
    def select(self, **kwargs):
        for i in range(len(self.weights)):
            self.set("WEIGHT_NUMBER", i+1)
            SVM.select(self,**kwargs)

class exSVM(SVM):
    """This class performs non-parametric, asymmetric least squares regression using SVMs. The tested estimators are
    therefore estimating the conditional tau-e;xpectiles of Y given X. By default, estimators for five different tau
    values are computed.

    Parameters
    ----------
    weights : arr of doubles
        The list of expectiles that should be estimated.
        (Default value = [0.05,0.1,0.5,0.9,0.95])

    others: see `?SVM` and `?doc.configuration`

    """
    def __init__(self, data, labs=None, weights=[0.05,0.1,0.5,0.9,0.95], **kwargs):
        SVM.__init__(self, data, labs, scenario="EX", **kwargs)
        self.set("WEIGHTS",weights)
        self.weights = weights
        self.predictCols = range(len(weights))
        self.train()
        self.select()
    def select(self, **kwargs):
        for i in range(len(self.weights)):
            self.set("WEIGHT_NUMBER", i+1)
            SVM.select(self,**kwargs)

class rocSVM(SVM):
    """This class provides several points on the ROC curve by solving multiple weighted binary classification problems.
    It is only suitable to binary classification data.

    Parameters
    ----------
    weightsteps : int
        The number of weights that should be used.
        (Default value = 9)

    others: see `?SVM` and `?doc.configuration`
    """
    def __init__(self, data, labs=None, weightsteps=9, **kwargs):
        SVM.__init__(self, data, labs, scenario="ROC", **kwargs)
        self.set("WEIGHT_STEPS",weightsteps)
        self.weightsteps = weightsteps
        self.predictCols = range(weightsteps)
        self.train()
        self.select()
    def select(self, **kwargs):
        for i in range(self.weightsteps):
            self.set("WEIGHT_NUMBER", i+1)
            SVM.select(self,**kwargs)

class nplSVM(SVM):
    """This class provides binary classifiers that satisfy a predefined error rate on one type of error and that
    simlutaneously minimize the other type of error. For convenience some points on the ROC curve around the
    predefined error rate are returned. nplNPL performs Neyman-Pearson-Learning for classification.

    Parameters
    ----------
    constraint : double
        The constraint around which different values should be found.
        (Default value = 0.05)
    constraintFactors : arr of doubles
        The factors to multiply `constraint` with.
        (Default value = [1/2,2/3,1,3/2,2])

    others: see `?SVM` and `?doc.configuration`
    """
    def __init__(self, data, labs=None, nplClass=1, constraint=0.05, constraintFactors=[1/2,2/3,1,3/2,2], **kwargs):
        SVM.__init__(self, data, labs, scenario="NPL "+str(nplClass), **kwargs)
        self.nplClass = nplClass
        self.constraint = constraint
        self.constraintFactors = constraintFactors
        self.predictCols = range(len(constraintFactors))
        self.train()
        self.select()
    def select(self, **kwargs):
        for cf in self.constraintFactors:
            self.set("NPL_CLASS", self.nplClass)
            self.set("NPL_CONSTRAINT", self.constraint * cf)
            SVM.select(self,**kwargs)

class LiquidData:
    """This class helps to organize train/test splits and targets. It has a train and a test attribute, of which each
    has a data, target, and DESCR attribte as known from sklearn. This looks at several locations to find a
    name.train.csv and name.test.csv. If it does then it loads or downloads it, parses it, and returns an
    liquidData-object. The files also can be gzipped having names name.train.csv.gz and name.test.csv.gz.
    
    Included in the package are 'banana-bc', 'banana-mc', 'covtype.1000', and 'reg-1d'.

    Parameters
    ----------
    name : str
        The base name of a train/test splitted data set to load.

    Attributes
    ----------
    name : str
        The name of the data sets
    train : Bunch
        The training data set.
    test : Bunch
        The test set including labels
    loc : str
        The location where the data was found
    
    """
    def __init__(self, name):
        train_data = None
        test_data = None
        theLoc = None
        for loc in ["./",os.path.expanduser('~/liquidData/'),
                    pkg_resources.resource_filename(__name__, 'data/'),
                    'http://www.isa.uni-stuttgart.de/liquidData/']:
            #print('Trying to load from ' + loc)
            for ext in ['.csv','.csv.gz']:
                try:
                    #print(loc+name+'.train'+ext)
                    train_data = np.genfromtxt(loc+name+'.train'+ext, delimiter=", ")
                    test_data = np.genfromtxt(loc+name+'.test'+ext, delimiter=", ")
                    theLoc = loc
                    break
                except:
                    pass
            if theLoc != None:
                break
        if train_data is None or test_data is None:
            raise FileNotFoundError

        self.name = name
        self.train = self.__bunch(train_data, name+" (train)")
        self.test = self.__bunch(test_data, name+" (test)")
        self.loc = theLoc
    
    def __bunch(self,data, descr):
      #ret = []
      #ret.data = data[:,1:]
      #ret.target = data[:,0]
      #ret.DESCR = descr
      return Bunch(data=data[:,1:] , target=data[:,0] , DESCR=descr)
      #return ret
    
    def fromData(self, train_x, train_y, test_x,test_y):
        """Creates a LiquidData from given `np.array`s

        Parameters
        ----------
        train_x : np.array
            Train features.
        train_y : np.array
            Train labels
        test_x : np.array
            Test features.
        test_y : np.array
            Test labels.

        Returns
        -------
        LiquidData
        """
        self.train = Bunch(data=train_x , target=train_y , DESCR='')
        self.test = Bunch(data=test_x , target=test_y , DESCR='')
class Bunch(dict):
    """ """
    def __init__(self, *args, **kwds):
        super(Bunch, self).__init__(*args, **kwds)
        self.__dict__ = self

if __name__ == '__main__':
    print("Hello to liquidSVM (python)")

    mcSVM('banana-mc',display=1, mcType="OvA_ls")
