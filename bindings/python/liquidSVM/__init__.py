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
Because of a fully integrated hyper-parameter selection, very carefully
implemented solvers, multi-threading and GPU support, and several built-in
data decomposition strategies it provides unprecedented speed for small
training sizes as well as for data sets of tens of millions of samples.

To install use

> pip install --user --upgrade liquidSVM

Then you can use it like:

>>> import liquidSVM as svm
>>> from liquidSVM import iris, iris_labs
>>> model = svm.mc(iris, iris_labs, display=1,threads=2)
>>> result, err = model.test(iris, iris_labs)
>>> result = model.predict(iris)

For more information see the README and the demo notebook.

Copyright: Ingo Steinwart and Philipp Thomann
'''

from __future__ import print_function
from . import model, data
# pylint: disable=wildcard-import
from .model import * # noqa
# pylint: disable=wildcard-import
from .data import * # noqa
import liquidSVM.doc

__all__ = model.__all__ + data.__all__ + ["doc"]


# Aliases for more access to functions,
# however only the xxxSVM are in __all__
# so that they are not loaded when using
#   from liquidSVM import *

ls = lsSVM
LeastSquares = lsSVM
mc = mcSVM
MultiClass = mcSVM
qt = qtSVM
QuantileRegression = qtSVM
ex = exSVM
ExpectileRegression = exSVM
npl = nplSVM
NeymanPearsonLearning = nplSVM
roc = rocSVM
ROC = rocSVM


if __name__ == '__main__':
    print("Hello to liquidSVM (python)")

    model.mcSVM('banana-mc', display=1, mcType="OvA_ls")
