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

    def test_issue163(self):
        a = fromiter([[1, 3], [4, 5]])
        b = a[a.counts > 10]
        assert b[:,:1].tolist() == []

    def test_issue_190(self):
        a = JaggedArray.fromiter([[], []])
        assert a.pad(1).tolist() == [[None], [None]]
        assert a.pad(2).tolist() == [[None, None], [None, None]]
        assert a.pad(3).tolist() == [[None, None, None], [None, None, None]]

    def test_issue_208(self):
        a = awkward.MaskedArray([True, False, False, True, False, True], awkward.fromiter([[1, 2, 3], [4, 5], [6], [7, 8], [10, 11, 12], [999]]))
        assert a.flatten().tolist() == [None, 4, 5, 6, None, 10, 11, 12, None]
        assert (a + 100).flatten().tolist() == [None, 104, 105, 106, None, 110, 111, 112, None]
        a = awkward.MaskedArray([True, False, False, True, False, True], [1.1, 2.2, 3.3, 4.4, 5.5, 6.6])
        assert a.flatten().tolist() == [None, 2.2, 3.3, None, 5.5, None]
        assert (a + 100).flatten().tolist() == [None, 102.2, 103.3, None, 105.5, None]
