#!/usr/bin/env python

# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-array/blob/master/LICENSE

import unittest

import numpy

from awkward import *
from awkward.type import *

class Test(unittest.TestCase):
    def runTest(self):
        pass

    def test_issue49(self):
        a = JaggedArray([2], [5], [1, 2, 3, 4, 5])
        m = JaggedArray([0], [3], [False, False, True])
        assert a[m].tolist() == [[5]]

    def test_issue144(self):
        x = fromiter([[], [0.5], [0.6], []])
        isloose = fromiter([[], [False], [True], []])
        assert x[isloose].tolist() == [[], [], [0.6], []]
        assert x.sum().tolist() == [0.0, 0.5, 0.6, 0.0]
