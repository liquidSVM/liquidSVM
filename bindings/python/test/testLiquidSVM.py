'''
Created on 11.03.2017

@author: Philipp Thomann
'''
import unittest
from liquidSVM import *


class Test(unittest.TestCase):

    def testliquidSVM_ls(self):
        d = LiquidData('reg-1d')
        m = lsSVM(d.train)
        pred = m.predict(d.test.data)
        myerr = ((pred-d.test.target)**2).mean()
        self.assertLess(myerr,0.01)
        result, err = m.test(d.test)
        self.assertAlmostEqual(myerr, err[0,0])

    def testliquidSVM_bc(self):
        d = LiquidData('covtype.1000')
        m = mcSVM(d.train)
        result, err = m.test(d.test)
        myerr = (result[:,0] != d.test.target).mean()
        self.assertLess(myerr, 0.26)
        self.assertAlmostEqual(myerr, err[0, 0])

    def testliquidSVM_mc(self):
        d = LiquidData('banana-mc')
        lev = 4
        tasks = lev * (lev-1) / 2
        m = mcSVM(d.train)
        result, err = m.test(d.test)
        myerr = (result[:,0] != d.test.target).mean()

        self.assertLess(myerr, 0.23)
        self.assertAlmostEqual(myerr, err[0, 0])

        self.assertEqual(result.shape, (len(d.test.target), tasks+1) )
        self.assertEqual(err.shape, (tasks+1, 3) )

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testliquidSVM']
    unittest.main()
