#!/usr/bin/env python

# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-array/blob/master/LICENSE

import unittest

import pytest
import numpy

awkward_numba = pytest.importorskip("awkward.numba")

class Test(unittest.TestCase):
    def runTest(self):
        pass

    def test_awkward_numba_init(self):
        a = awkward_numba.JaggedArray([0, 3, 3, 5], [3, 3, 5, 10], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        assert isinstance(a, awkward_numba.array.jagged.JaggedArrayNumba)
        assert a.JaggedArray is awkward_numba.array.jagged.JaggedArrayNumba

        assert isinstance(a[2:], awkward_numba.array.jagged.JaggedArrayNumba)
        assert isinstance(a[[False, True, True, False]], awkward_numba.array.jagged.JaggedArrayNumba)
        assert isinstance(a[[1, 2]], awkward_numba.array.jagged.JaggedArrayNumba)
        assert isinstance(a + 100, awkward_numba.array.jagged.JaggedArrayNumba)
        assert isinstance(numpy.sin(a), awkward_numba.array.jagged.JaggedArrayNumba)
        assert isinstance(a.cross(a), awkward_numba.array.jagged.JaggedArrayNumba)
        assert isinstance(a.pairs(), awkward_numba.array.jagged.JaggedArrayNumba)
        assert isinstance(a.distincts(), awkward_numba.array.jagged.JaggedArrayNumba)
        assert isinstance(a.cross(a, nested=True), awkward_numba.array.jagged.JaggedArrayNumba)
        assert isinstance(a.pairs(nested=True), awkward_numba.array.jagged.JaggedArrayNumba)
        assert isinstance(a.distincts(nested=True), awkward_numba.array.jagged.JaggedArrayNumba)
        assert isinstance(a.concatenate([a]), awkward_numba.array.jagged.JaggedArrayNumba)

        b = awkward_numba.JaggedArray([1, 4, 4, 6], [4, 4, 6, 11], [999, 0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        assert isinstance(a + b, awkward_numba.array.jagged.JaggedArrayNumba)

    def test_awkward_numba_argmin(self):
        a = awkward_numba.JaggedArray([0, 3, 3, 5], [3, 3, 5, 10], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        assert isinstance(a.argmin(), awkward_numba.array.jagged.JaggedArrayNumba)
        assert a.argmin().tolist() == [[0], [], [0], [0]]

        a = awkward_numba.JaggedArray([[0, 3], [3, 5]], [[3, 3], [5, 10]], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        assert isinstance(a.argmin(), awkward_numba.array.jagged.JaggedArrayNumba)
        assert a.argmin().tolist() == [[[0], []], [[0], [0]]]

    def test_awkward_numba_argmax(self):
        a = awkward_numba.JaggedArray([0, 3, 3, 5], [3, 3, 5, 10], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        assert isinstance(a.argmin(), awkward_numba.array.jagged.JaggedArrayNumba)
        assert a.argmax().tolist() == [[2], [], [1], [4]]

        a = awkward_numba.JaggedArray([[0, 3], [3, 5]], [[3, 3], [5, 10]], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        assert isinstance(a.argmin(), awkward_numba.array.jagged.JaggedArrayNumba)
        assert a.argmax().tolist() == [[[2], []], [[1], [4]]]
