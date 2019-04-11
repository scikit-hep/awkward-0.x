#!/usr/bin/env python

# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-array/blob/master/LICENSE

import struct
import unittest

import numpy

from awkward import *
import awkward.type

class Test(unittest.TestCase):
    def runTest(self):
        pass

    def test_virtual_nbytes(self):
        assert isinstance(VirtualArray(lambda: [1, 2, 3]).nbytes, int)
        assert VirtualArray(lambda: [1, 2, 3], nbytes=12345).nbytes == 12345

    def test_virtual_nocache(self):
        a = VirtualArray(lambda: [1, 2, 3])
        assert not a.ismaterialized
        assert numpy.array_equal(a[:], numpy.array([1, 2, 3]))
        assert a.ismaterialized

        a = VirtualArray(lambda: range(10))
        assert not a.ismaterialized
        assert numpy.array_equal(a[::2], numpy.array([0, 2, 4, 6, 8]))
        assert a.ismaterialized

        a = VirtualArray(lambda: range(10))
        assert not a.ismaterialized
        assert numpy.array_equal(a[[5, 3, 6, 0, 6]], numpy.array([5, 3, 6, 0, 6]))
        assert a.ismaterialized

        a = VirtualArray(lambda: range(10))
        assert not a.ismaterialized
        assert numpy.array_equal(a[[True, False, True, False, True, False, True, False, True, False]], numpy.array([0, 2, 4, 6, 8]))
        assert a.ismaterialized

    def test_virtual_transientcache(self):
        cache = {}
        a = VirtualArray(lambda: [1, 2, 3], cache=cache)
        assert not a.ismaterialized
        a[:]
        assert a.ismaterialized
        assert list(cache) == [a.TransientKey(id(a))]
        assert list(cache) == [a.key]
        assert numpy.array_equal(cache[a.key], numpy.array([1, 2, 3]))
        del a

    def test_virtual_persistentcache(self):
        cache = {}
        a = VirtualArray(lambda: [1, 2, 3], cache=cache, persistentkey="find-me-again")
        assert not a.ismaterialized
        a[:]
        assert a.ismaterialized
        assert list(cache) == ["find-me-again"]
        assert list(cache) == [a.key]
        assert numpy.array_equal(cache[a.key], numpy.array([1, 2, 3]))
        del a

    def test_virtual_dontmaterialize(self):
        a = VirtualArray(lambda: [1, 2, 3], type=awkward.type.fromnumpy(3, int))
        assert not a.ismaterialized
        assert a.dtype == numpy.dtype(int)
        assert a.shape == (3,)
        assert len(a) == 3
        assert a._array == None
        assert not a.ismaterialized
        assert numpy.array_equal(a[:], numpy.array([1, 2, 3]))
        assert a.ismaterialized
