'''
Created on 11.03.2017

@author: Philipp Thomann
'''
import warnings

import numpy as np
import unittest
from liquidSVM import LiquidData, lsSVM, mcSVM
import liquidSVM as svm


class Test(unittest.TestCase):

    def testliquidSVM_ls(self):
        d = LiquidData('reg-1d', trainSize=400)
        m = lsSVM(d.train)
        self.assertIn('selected', str(m))
        self.assertIn('lsSVM', repr(m))
        pred = m.predict(d.test.data)
        myerr = ((pred - d.test.target)**2).mean()
        self.assertLess(myerr, 0.01)
        result, err = m.test(d.test)
        self.assertAlmostEqual(myerr, err[0, 0])

    def testliquidSVM_bc(self):
        d = LiquidData('covtype.1000', trainSize=400)
        m = mcSVM(d.train)
        result, err = m.test(d.test)
        myerr = (result[:, 0] != d.test.target).mean()
        self.assertLess(myerr, 0.28)
        self.assertAlmostEqual(myerr, err[0, 0])

    def testliquidSVM_mc(self):
        d = LiquidData('banana-mc', trainSize=400)
        lev = 4
        tasks = lev * (lev - 1) / 2
        m = mcSVM(d.train)
        result, err = m.test(d.test)
        myerr = (result[:, 0] != d.test.target).mean()

        self.assertLess(myerr, 0.25)
        self.assertAlmostEqual(myerr, err[0, 0])

        self.assertEqual(result.shape, (len(d.test.target), tasks + 1))
        self.assertEqual(err.shape, (tasks + 1, 3))

    def testliquidSVM_coverage_quick(self):
        # using string for data:
        m = svm.SVM('reg-1d')

        # other scenarios
        reg = LiquidData("reg-1d",trainSize=100)
        q = svm.qt(reg.train)
        e = svm.ex(reg.train)

        cl = LiquidData("banana-bc", trainSize=100)
        npl = svm.npl(cl.train)
        roc = svm.roc(cl.train)

        # solution
        m = lsSVM(reg.train)
        m.solution(1,1,1)

        # boolean args
        m = svm.SVM(reg.train, store_solutions_internally=False)

        # weights, groups, ids
        n = reg.train.target.shape[0]
        ones = np.ones(n)
        m = svm.SVM(reg.train, sampleWeights=ones, groupIds=ones, ids=np.arange(n))

    def testLiquidData(self):
        cov = LiquidData('covtype.1000')
        n = cov.train.data.shape[0]
        self.assertIn(str(cov.train.data.shape[0]), str(cov))
        self.assertIn(str(cov.train.data.shape[1]), str(cov))

        banana = LiquidData('banana-mc')
        banana1 = banana.sample(trainSize=400)
        banana2 = banana.sample(trainSize=400, stratified=True)
        banana3 = banana.sample(trainSize=400, stratified=False)

        def assertIt(self, d, size, table={1:120,2:120,3:80,4:80},
                     stratified=True):
            self.assertEqual(d.data.shape[0], size)
            self.assertEqual(d.target.shape[0], size)
            table2 = dict(zip(*np.unique(d.target, return_counts=True)))
            if stratified:
                self.assertDictEqual(table2, table)
            else:
                self.assertFalse(table == table2)

        assertIt(self, banana1.train, 400)
        assertIt(self, banana2.train, 400)
        assertIt(self, banana3.train, 400, stratified=False)

        assertIt(self, banana1.test, 400)
        assertIt(self, banana2.test, 400)
        assertIt(self, banana3.test, 400, stratified=False)

        with self.assertRaises(IOError):
            LiquidData('file_does_not_exist',loc='folder_does_not_exist')

        reg = LiquidData('reg-1d')
        self.assertIn(str(reg.train.data.shape[0]), str(reg))
        self.assertIn(str(reg.train.data.shape[1]), str(reg))

        def assertItTT(self, d, tr, ts):
            self.assertEqual(d.train.target.shape[0], tr)
            self.assertEqual(d.test.target.shape[0], ts)
        assertItTT(self, reg.sample(trainSize=100), 100, 100)
        assertItTT(self, reg.sample(testSize=100), 100, 100)
        assertItTT(self, reg.sample(trainSize=100, testSize=200), 100, 200)
        assertItTT(self, reg.sample(prob=0.1), 200, 200)

        # unfortunately self.assertWarns() is not in Python 2.7...
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            reg2 = reg.sample(prob=2)
            self.assertTrue(any(item.category == UserWarning for item in w))


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testliquidSVM']
    unittest.main()
