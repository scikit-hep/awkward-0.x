#!/usr/bin/env python

# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-array/blob/master/LICENSE

import unittest

import numpy

from awkward import *

class Test(unittest.TestCase):
    def runTest(self):
        pass

    def test_union_nbytes(self):
        assert isinstance(UnionArray([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [[0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], [0, 100, 200, 300, 400, 500, 600, 700, 800, 900]]).nbytes, int)

    def test_union_get(self):
        a = UnionArray([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [[0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], [0, 100, 200, 300, 400, 500, 600, 700, 800, 900]])
        assert a.tolist() == [0.0, 100, 2.2, 300, 4.4, 500, 6.6, 700, 8.8, 900]
        assert [a[i] for i in range(len(a))] == [0.0, 100, 2.2, 300, 4.4, 500, 6.6, 700, 8.8, 900]
        assert a[:].tolist() == [0.0, 100, 2.2, 300, 4.4, 500, 6.6, 700, 8.8, 900]
        assert a[1:].tolist() == [100, 2.2, 300, 4.4, 500, 6.6, 700, 8.8, 900]
        assert a[2:].tolist() == [2.2, 300, 4.4, 500, 6.6, 700, 8.8, 900]
        assert a[:-1].tolist() == [0.0, 100, 2.2, 300, 4.4, 500, 6.6, 700, 8.8]
        assert a[1:-2].tolist() == [100, 2.2, 300, 4.4, 500, 6.6, 700]
        assert a[2:-2].tolist() == [2.2, 300, 4.4, 500, 6.6, 700]
        assert [a[2:-2][i] for i in range(6)] == [2.2, 300, 4.4, 500, 6.6, 700]

        assert a[[1, 2, 3, 8, 8, 1]].tolist() == [100, 2.2, 300, 8.8, 8.8, 100]
        assert [a[[1, 2, 3, 8, 8, 1]][i] for i in range(6)] == [100, 2.2, 300, 8.8, 8.8, 100]

        assert a[[False, True, True, True, False, True, False, False, False, False]].tolist() == [100, 2.2, 300, 500]
        assert [a[[False, True, True, True, False, True, False, False, False, False]][i] for i in range(4)] == [100, 2.2, 300, 500]

    def test_union_ufunc(self):
        a = UnionArray.fromtags([0, 1, 1, 0, 0], [[100, 200, 300], [1.1, 2.2]])
        b = UnionArray.fromtags([1, 1, 0, 1, 0], [[10.1, 20.2], [123, 456, 789]])
        assert (a + a).tolist() == [200, 2.2, 4.4, 400, 600]
        assert (a + b).tolist() == [223, 457.1, 12.3, 989, 320.2]
