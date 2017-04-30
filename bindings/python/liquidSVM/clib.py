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
'''c-library interface to core liquidSVM

liquidSVM is a package written in C++ and this module provides access to that
by the use of ctypes.
@author: Ingo Steinwart and Philipp Thomann
'''

import numpy as np
import numpy.ctypeslib as npct
import ctypes as ct
import os
import sysconfig
import glob

__all__ = ['_set_param', '_get_param', '_get_config_line', '_libliquidSVM']


# Load the library as _libliquidSVM.
_filepath = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# print(_filepath)
# Why the underscore (_) in front of _libliquidSVM below?
# To mimimise namespace pollution -- see PEP 8 (www.python.org).
# _libliquidSVM = npct.load_library('impl.so', _filepath)
# _libliquidSVM = npct.load_library('libliquidsvm', _filepath)
_locations = [_filepath,
              _filepath + "/build/*/",
              # "/home/thomapp/liquidSVM/bindings/python/venvs/py3/lib/python3.4/site-packages/liquidSVM-0.5-py3.4-linux-x86_64.egg",
              # "/home/thomapp/opt/anaconda/lib/python2.7/site-packages/liquidSVM-0.5-py2.7-linux-x86_64.egg",
              # "/home/thomapp/opt/anaconda/envs/anacondaPy3/lib/python3.6/site-packages/liquidSVM-0.5-py3.6-linux-x86_64.egg"
              ]
for loc in _locations:
    _libliquidSVM = None
    try:
        # print('Trying to load from: '+loc)
        _thenames = glob.glob(
            loc + '/liquidSVM*' + sysconfig.get_config_var('SO'))
        for n in _thenames:
            # print('Trying to load from: '+n)
            # print('Trying to load at: '+os.path.dirname(n))
            _libliquidSVM = npct.load_library('liquidSVM', os.path.dirname(n))
            # _libliquidSVM = npct.load_library(os.path.basename(n), os.path.dirname(n))
            # print(_libliquidSVM)
            break
    except BaseException as a:
        # print("Could not load!"+a)
        pass
    except:
        # print("Could not load!")
        pass
    if _libliquidSVM:
        break
else:
    raise Exception('Did not find native library, expected at ' + _filepath)


TdoubleP = npct.ndpointer(dtype=np.double)


def nullable_from_param(cls, obj):
    '''To pass NULL for some numpy arrays, the workaround is from
    http://stackoverflow.com/questions/32120178
    '''
    if obj is None:
        return obj
    return TdoubleP.from_param(obj)


TdoublePnullable = type(
    'TdoublePnullable',
    (TdoubleP,),
    {'from_param': classmethod(nullable_from_param)}
)

_set_param = _libliquidSVM.liquid_svm_set_param
_set_param.argtypes = [ct.c_int, ct.c_char_p, ct.c_char_p]

_get_param = _libliquidSVM.liquid_svm_get_param
_get_param.argtypes = [ct.c_int, ct.c_char_p]
_get_param.restype = ct.c_char_p

_get_config_line = _libliquidSVM.liquid_svm_get_config_line
_get_config_line.argtypes = [ct.c_int, ct.c_int]
_get_config_line.restype = ct.c_char_p

_libliquidSVM.liquid_svm_init.argtypes = [
    TdoubleP, ct.c_int, ct.c_int, TdoubleP]
_libliquidSVM.liquid_svm_init.restype = ct.c_int

_libliquidSVM.liquid_svm_train.argtypes = [
    ct.c_int, ct.c_int, ct.POINTER(ct.c_char_p)]
_libliquidSVM.liquid_svm_train.restype = ct.POINTER(ct.c_double)

_libliquidSVM.liquid_svm_select.argtypes = [
    ct.c_int, ct.c_int, ct.POINTER(ct.c_char_p)]
_libliquidSVM.liquid_svm_select.restype = ct.POINTER(ct.c_double)

_libliquidSVM.liquid_svm_test.argtypes = [
    ct.c_int, ct.c_int, ct.POINTER(ct.c_char_p),
    TdoubleP, ct.c_int, ct.c_int, TdoublePnullable,
    ct.POINTER(ct.POINTER(ct.c_double))]
_libliquidSVM.liquid_svm_test.restype = ct.POINTER(ct.c_double)

_libliquidSVM.liquid_svm_clean.argtypes = [ct.c_int]
