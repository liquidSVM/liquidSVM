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
'''Drop-in replacements to use liquidSVM in legacy sklearn code.

Where you would else use sklearn.svm.SVC or SVR you can just
use ours. First load some data:
>>> import liquidSVM.liquidData as ld
>>> banana = ld.LiquidData('banana-bc')
Now in sklearn you would do something like
>>> import sklearn.svm as sk
>>> model = sk.SVC(verbose=1)
>>> model.fit(banana.train.data, banana.train.target)
>>> ( model.predict(banana.test.data) != banana.test.target).mean()
You can just replace it with:
>>> import liquidSVM.learn as ll
>>> model = ll.SVC(display=1)
>>> model.fit(banana.train.data, banana.train.target)
>>> ( model.predict(banana.test.data) != banana.test.target).mean()

Be careful to let liquidSVM do its internal cross validation.
Furthermore at the moment arguments do not get translated
(e.g. verbose -> display).


@author: Philipp Thomann
'''

import numpy as np
from .model import *

# import sklearn

__all__ = ['SVR','SVC']

class SVR(lsSVM):
    '''
    This class can be used as a drop-in replacement for sklearn.svm.SVR.
    '''
    def __init__(self, **kwargs):
        # do not call super().__init__ just yet...
        self.kwargs = kwargs
    
    def fit(self, X, y):
        super().__init__(X, y, **self.kwargs)
        return self

class SVC(mcSVM):
    '''
    This class can be used as a drop-in replacement for sklearn.svm.SVC.
    '''
    def __init__(self, **kwargs):
        # do not call super().__init__ just yet...
        self.kwargs = kwargs
    
    def fit(self, X, y):
        super().__init__(X, y, **self.kwargs)
        return self


