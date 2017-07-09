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
'''This module helps reading and managing train/test splits from various
places.
'''
import os
import numpy as np
import pkg_resources
import warnings

__all__ = ["iris", "iris_labs", "LiquidData"]

# pylint: disable=line-too-long, invalid-name, no-member
iris = np.array([5.1, 4.9, 4.7, 4.6, 5, 5.4, 4.6, 5, 4.4, 4.9, 5.4, 4.8, 4.8, 4.3, 5.8, 5.7, 5.4, 5.1, 5.7, 5.1, 5.4, 5.1, 4.6, 5.1, 4.8, 5, 5, 5.2, 5.2, 4.7, 4.8, 5.4, 5.2, 5.5, 4.9, 5, 5.5, 4.9, 4.4, 5.1, 5, 4.5, 4.4, 5, 5.1, 4.8, 5.1, 4.6, 5.3, 5, 7, 6.4, 6.9, 5.5, 6.5, 5.7, 6.3, 4.9, 6.6, 5.2, 5, 5.9, 6, 6.1, 5.6, 6.7, 5.6, 5.8, 6.2, 5.6, 5.9, 6.1, 6.3, 6.1, 6.4, 6.6, 6.8, 6.7, 6, 5.7, 5.5, 5.5, 5.8, 6, 5.4, 6, 6.7, 6.3, 5.6, 5.5, 5.5, 6.1, 5.8, 5, 5.6, 5.7, 5.7, 6.2, 5.1, 5.7, 6.3, 5.8, 7.1, 6.3, 6.5, 7.6, 4.9, 7.3, 6.7, 7.2, 6.5, 6.4, 6.8, 5.7, 5.8, 6.4, 6.5, 7.7, 7.7, 6, 6.9, 5.6, 7.7, 6.3, 6.7, 7.2, 6.2, 6.1, 6.4, 7.2, 7.4, 7.9, 6.4, 6.3, 6.1, 7.7, 6.3, 6.4, 6, 6.9, 6.7, 6.9, 5.8, 6.8, 6.7, 6.7, 6.3, 6.5, 6.2, 5.9, 3.5, 3, 3.2, 3.1, 3.6, 3.9, 3.4, 3.4, 2.9, 3.1, 3.7, 3.4, 3, 3, 4, 4.4, 3.9, 3.5, 3.8, 3.8, 3.4, 3.7, 3.6, 3.3, 3.4, 3, 3.4, 3.5, 3.4, 3.2, 3.1, 3.4, 4.1, 4.2, 3.1, 3.2, 3.5, 3.6, 3, 3.4, 3.5, 2.3, 3.2, 3.5, 3.8, 3, 3.8, 3.2, 3.7, 3.3, 3.2, 3.2, 3.1, 2.3, 2.8, 2.8, 3.3, 2.4, 2.9, 2.7, 2, 3, 2.2, 2.9, 2.9, 3.1, 3, 2.7, 2.2, 2.5, 3.2, 2.8, 2.5, 2.8, 2.9, 3, 2.8, 3, 2.9, 2.6, 2.4, 2.4, 2.7, 2.7, 3, 3.4, 3.1, 2.3, 3, 2.5, 2.6, 3, 2.6, 2.3, 2.7, 3, 2.9, 2.9, 2.5, 2.8, 3.3, 2.7, 3, 2.9, 3, 3, 2.5, 2.9, 2.5, 3.6, 3.2, 2.7, 3, 2.5, 2.8, 3.2, 3, 3.8, 2.6, 2.2, 3.2, 2.8, 2.8, 2.7, 3.3, 3.2, 2.8, 3, 2.8, 3, 2.8, 3.8, 2.8, 2.8, 2.6, 3, 3.4, 3.1, 3, 3.1, 3.1, 3.1, 2.7, 3.2, 3.3, 3, 2.5, 3, 3.4, 3, 1.4, 1.4, 1.3, 1.5, 1.4, 1.7, 1.4, 1.5, 1.4, 1.5, 1.5, 1.6, 1.4, 1.1, 1.2, 1.5, 1.3, 1.4, 1.7, 1.5, 1.7, 1.5, 1, 1.7, 1.9, 1.6, 1.6, 1.5, 1.4, 1.6, 1.6, 1.5, 1.5, 1.4, 1.5, 1.2, 1.3, 1.4, 1.3, 1.5, 1.3, 1.3, 1.3, 1.6, 1.9, 1.4, 1.6, 1.4, 1.5, 1.4, 4.7, 4.5, 4.9, 4, 4.6, 4.5, 4.7, 3.3, 4.6, 3.9, 3.5, 4.2, 4, 4.7, 3.6, 4.4, 4.5, 4.1, 4.5, 3.9, 4.8, 4, 4.9, 4.7, 4.3, 4.4, 4.8, 5, 4.5, 3.5, 3.8, 3.7, 3.9, 5.1, 4.5, 4.5, 4.7, 4.4, 4.1, 4, 4.4, 4.6, 4, 3.3, 4.2, 4.2, 4.2, 4.3, 3, 4.1, 6, 5.1, 5.9, 5.6, 5.8, 6.6, 4.5, 6.3, 5.8, 6.1, 5.1, 5.3, 5.5, 5, 5.1, 5.3, 5.5, 6.7, 6.9, 5, 5.7, 4.9, 6.7, 4.9, 5.7, 6, 4.8, 4.9, 5.6, 5.8, 6.1, 6.4, 5.6, 5.1, 5.6, 6.1, 5.6, 5.5, 4.8, 5.4, 5.6, 5.1, 5.1, 5.9, 5.7, 5.2, 5, 5.2, 5.4, 5.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.4, 0.3, 0.2, 0.2, 0.1, 0.2, 0.2, 0.1, 0.1, 0.2, 0.4, 0.4, 0.3, 0.3, 0.3, 0.2, 0.4, 0.2, 0.5, 0.2, 0.2, 0.4, 0.2, 0.2, 0.2, 0.2, 0.4, 0.1, 0.2, 0.2, 0.2, 0.2, 0.1, 0.2, 0.2, 0.3, 0.3, 0.2, 0.6, 0.4, 0.3, 0.2, 0.2, 0.2, 0.2, 1.4, 1.5, 1.5, 1.3, 1.5, 1.3, 1.6, 1, 1.3, 1.4, 1, 1.5, 1, 1.4, 1.3, 1.4, 1.5, 1, 1.5, 1.1, 1.8, 1.3, 1.5, 1.2, 1.3, 1.4, 1.4, 1.7, 1.5, 1, 1.1, 1, 1.2, 1.6, 1.5, 1.6, 1.5, 1.3, 1.3, 1.3, 1.2, 1.4, 1.2, 1, 1.3, 1.2, 1.3, 1.3, 1.1, 1.3, 2.5, 1.9, 2.1, 1.8, 2.2, 2.1, 1.7, 1.8, 1.8, 2.5, 2, 1.9, 2.1, 2, 2.4, 2.3, 1.8, 2.2, 2.3, 1.5, 2.3, 2, 2, 1.8, 2.1, 1.8, 1.8, 1.8, 2.1, 1.6, 1.9, 2, 2.2, 1.5, 1.4, 2.3, 2.4, 1.8, 1.8, 2.1, 2.4, 2.3, 1.9, 2.3, 2.5, 2.3, 1.9, 2, 2.3, 1.8]).reshape(150, 4)
# pylint: disable=line-too-long, invalid-name
iris_labs = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3])


class LiquidData(object): # pylint: disable=too-few-public-methods
    """This class helps to organize train/test splits and targets. It has a
    train and a test attribute, of which each has a data, target, and DESCR
    attribute as known from sklearn. This looks at several locations to find
    a name.train.csv and name.test.csv. If it does then it loads or downloads
    it, parses it, and returns an liquidData-object. The files also can be
    gzipped having names name.train.csv.gz and name.test.csv.gz.

    Included in the package are 'banana-bc', 'banana-mc', 'covtype.1000', and
    'reg-1d'.

    Parameters
    ----------
    name : str
        The base name of a train/test splitted data set to load.
    targetCol : int
        The index of the target column where the labels are
    header : bool
        do the data files have headers
    loc : str or list of str
        The location where the data was found
    prob : None or float
        probability of sample being put into test set
    trainSize : None or int
        size of the train set. If stratified, this will only be approximately fulfilled.
    testSize : None or int
        size of the test set. If stratified, this will only be approximately fulfilled.
    stratified : None or bool or int
        whether sampling should be done separately in every bin defined by
        the unique values of the target column.
        Also can be index or name of the column in \code{data} that should be used to define bins.
    delimiter : str (", " is default)
        passed to `numpy.genfromtext`
    **kwargs : dict
        passed to `numpy.genfromtext`

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
    def __init__(self, name, targetCol=0, header=False,loc=[
                    ".","~/liquidData",
                    pkg_resources.resource_filename(__name__, 'data/'),
                    "http://www.isa.uni-stuttgart.de/liquidData"
                 ], prob=None, testSize=None, trainSize=None, stratified=None,
                 delimiter=", ", **kwargs):
        train_data = None
        test_data = None
        the_loc = None
        if isinstance(loc, str):
            loc = [ loc ]
        for l in loc:
            # print('Trying to load from ' + loc)
            for ext in ['.csv', '.csv.gz']:
                try:
                    # print(loc+name+'.train'+ext)
                    train_file_name = l + "/" + name + '.train' + ext
                    test_file_name = l + "/" + name + '.train' + ext
                    train_data = np.genfromtxt(train_file_name, delimiter=delimiter, **kwargs)
                    test_data = np.genfromtxt(test_file_name, delimiter=delimiter, **kwargs)
                    the_loc = l
                    break
                except: # pylint: disable=bare-except
                    pass
            if the_loc is not None:
                break
        if train_data is None or test_data is None:
            raise IOError("Data files for name %s not found." % name)

        self.name = name
        self.train = self.__bunch(train_data, targetCol, name + " (train)",
                                  prob=prob, size=trainSize, stratified=stratified)
        self.test = self.__bunch(test_data, targetCol, name + " (test)",
                                  prob=prob, size=testSize, stratified=stratified)
        self.loc = the_loc

    @staticmethod
    def _isIntegerArray(x):
        if x.dtype.kind == 'i':
            return True
        return bool(np.equal(np.mod(x, 1), 0).all())

    @staticmethod
    def _sampleBunch(bunch, prob, size, stratified):
        I = LiquidData._sampleIt(bunch.data, bunch.target, prob, size, stratified)
        ret = bunch.copy()
        ret.data = bunch.data[I,:]
        if bunch.target is not None:
            ret.target = bunch.target[I]
        return ret

    @staticmethod
    def _sampleIt(data, target, prob, size, stratified):

        n = data.shape[0]
        if target is not None and target.shape != (n,):
            raise ValueError('target has shape %s but data has %d samples.' % (target.shape, n))

        if prob is None and size is None:
            return range(n)

        if stratified is None:
            if target is None:
                stratified = False
            else:
                stratified = LiquidData._isIntegerArray(target)
        if stratified == True:
            if target is None:
                raise ValueError('if stratified=True also target has to be specified.')
            stratified = target

        if size is not None:
            prob = size / float(n)
        else:
            size = max(round(n * prob), 1)
        size = int(size)

        if size >= n:
            # first we recommend to use Inf
            if size != np.inf and size > n and prob > 1:
                warnings.warn("Trying to sample more data than available. This is interpreted to shuffle all available"
                              " data. If this is what you want use size=Inf or prob=1")
            # now we just coule do
            #   return(1:n)
            # but we still want to have it shuffled:
            return np.random.permutation(n)

        if stratified is None or (isinstance(stratified, bool) and stratified == False):
            return np.random.choice(n, size, replace=False)
        else:
            ## split indices into groups
            import collections
            groups = collections.defaultdict(list)
            for i in range(n):
                groups[stratified[i]].append(i)
            ## do the stratified sampling
            samples = [
                np.random.choice(g, size=max(int(prob * len(g)), 1), replace=False) for g in groups.values()
                ]
            samples =np.concatenate(samples)
            ## finally shuffle everything around
            return np.random.permutation(samples)

    def sample(self, prob=None, trainSize=None, testSize=None, stratified=None):
        """Creates a new LiquidData that samples from the current one.

        Parameters
        ----------
        prob : None or float
            probability of sample being put into test set
        trainSize : None or int
            size of the train set. If stratified, this will only be approximately fulfilled.
        testSize : None or int
            size of the test set. If stratified, this will only be approximately fulfilled.
        stratified : None or bool or int
            whether sampling should be done separately in every bin defined by
            the unique values of the target column.
            Also can be index of the column in ``data`` that should be used to define bins.
        Examples
        --------
        ## example for sample.liquidData
        banana = LiquidData('banana-mc')
        banana.sample(prob=0.1)
        # this is equivalent to
        LiquidData('banana-mc', prob=0.1)
        """
        ## first we create a new environment
        ret = self.copy()

        if prob is None:
            if trainSize is None and testSize is None:
                raise ValueError("one of prob, trainSize, testSize hast to be specified!")
            if testSize is None:
                testSize = ret.test.data.shape[0] * trainSize / ret.train.data.shape[0]
            if trainSize is None:
                trainSize = ret.train.data.shape[0] * testSize / ret.test.data.shape[0]

        ret.name = ret.name + " (sample)"
        ret.train = LiquidData._sampleBunch(ret.train, prob, trainSize, stratified)
        ret.test = LiquidData._sampleBunch(ret.test, prob, testSize, stratified)
        return ret

    def __repr__(self):
        return "LiquidData(%s) <dim: %d train: %d, test: %d>" % (
                                self.name, self.train.data.shape[1],
                                self.train.data.shape[0], self.test.data.shape[0])

    def __str__(self):
        ret = []
        x = self
        def cat(*args, **kwargs):
            sep = kwargs.get('sep', " ")
            ret.append(sep.join([ str(i) for i in args ]))
        cat('LiquidData "', x.name, '"', sep = '')
        cat(" with", x.train.data.shape[0], "train samples and", x.train.data.shape[0], "test samples")
        cat("\n")
        cat("  having", x.train.data.shape[1], "columns")
        if x.train.target is not None:
            cat(' and a target with ')#, x.train.target, '"', sep='')
            try:
                col = x.train.target
                if col.dtype.kind == 'i':
                    lev, a = np.unique(col, return_counts=True)
                    cat(len(a), 'unique levels: ')
                    b = [ "".join([str(l), ' (', str(c), ' samples)']) for l,c in zip(lev, a) ]
                    cat(", ".join(b[:min(3, len(b))]))
                    if len(b) > 3:
                        cat(", ...")
                else:
                    cat('mean %.3f and range [%.3f,%.3f]' % (col.mean(), np.min(col), np.max(col)))
            except Exception as e:
                pass
        cat('\n')
        return "".join(ret)

    @staticmethod
    def __bunch(data, targetCol, descr, prob, size, stratified):
        # ret = []
        # ret.data = data[:,1:]
        # ret.target = data[:,0]
        # ret.DESCR = descr
        if targetCol is None:
            bunch = Bunch(data=data, target=None, DESCR=descr)
        else:
            if targetCol < 0 or targetCol > data.shape[1]:
                raise ValueError('targetCol is %d but has to be between 0 and %d.' % (targetCol, data.shape[1]))
            target = data[:, targetCol]
            data = np.delete(data, targetCol, axis=1)
            if LiquidData._isIntegerArray(target):
                target = target.astype(np.int)
            bunch = Bunch(data=data, target=target, DESCR=descr)
        return LiquidData._sampleBunch(bunch, prob=prob, size=size, stratified=stratified)

    @staticmethod
    def from_data(self, train_x, train_y, test_x, test_y, prob=1, trainSize=None, testSize=None, stratified=None):
        """Creates a LiquidData from given np.array objects

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
        self.train = Bunch(data=train_x, target=train_y, DESCR='')
        self.test = Bunch(data=test_x, target=test_y, DESCR='')

    def copy(self):
        import copy
        return copy.copy(self)
    # def copy(self):
    #     newone = LiquidData.__new__(LiquidData)
    #     newone.__dict__.update(self.__dict__)
    #     return newone


class Bunch(dict):
    """This emulates the Bunch of sklearn"""
    def __init__(self, *args, **kwds):
        super(Bunch, self).__init__(*args, **kwds)
        self.__dict__ = self
    def copy(self):
        import copy
        return copy.copy(self)
    # def copy(self):
    #     newone = Bunch.__new__(Bunch)
    #     newone.__dict__.update(self.__dict__)
    #     return newone
