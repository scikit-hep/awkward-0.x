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
