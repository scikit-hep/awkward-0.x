#!/usr/bin/env python

# Copyright (c) 2019, IRIS-HEP
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import sys

import unittest

import numpy
import pytest

import awkward
numba = pytest.importorskip("numba")
awkward_numba = pytest.importorskip("awkward.numba")

from awkward import *

class Test(unittest.TestCase):
    def runTest(self):
        pass

    def test_numba_unbox(self):
        a = JaggedArray.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
        a2 = JaggedArray.fromcounts([2, 0, 1], a)
        @numba.njit
        def test(x):
            return 3.14
        test(a)
        test(a2)

    def test_numba_box(self):
        a = JaggedArray.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
        a2 = JaggedArray.fromcounts([2, 0, 1], a)
        @numba.njit
        def test(x):
            return x
        assert test(a).tolist() == a.tolist()
        assert test(a2).tolist() == a2.tolist()

    def test_numba_init(self):
        @numba.njit
        def test(starts, stops, content):
            return JaggedArray(starts, stops, content)
        starts = numpy.array([0, 3, 3])
        stops = numpy.array([3, 3, 5])
        content = numpy.array([1.1, 2.2, 3.3, 4.4, 5.5])
        z = test(starts, stops, content)
        assert z.tolist() == [[1.1, 2.2, 3.3], [], [4.4, 5.5]]
        assert z.starts is starts
        assert z.stops is stops
        assert z.content is content
        z = test(starts, stops, content)
        assert z.tolist() == [[1.1, 2.2, 3.3], [], [4.4, 5.5]]
        assert z.starts is starts
        assert z.stops is stops
        assert z.content is content
        a = JaggedArray.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
        starts2 = numpy.array([0, 2, 2])
        stops2 = numpy.array([2, 2, 3])
        assert test(starts2, stops2, a).tolist() == [[[1.1, 2.2, 3.3], []], [], [[4.4, 5.5]]]

    def test_numba_new(self):
        @numba.njit
        def test(x, starts, stops, content):
            return awkward_numba.array.jagged._JaggedArray_new(x, starts, stops, content, False)
        starts = numpy.array([0, 3, 3])
        stops = numpy.array([3, 3, 5])
        content = numpy.array([1.1, 2.2, 3.3, 4.4, 5.5])
        a = awkward_numba.JaggedArray.fromiter([[999.9], [3.14], [2.2, 2.2, 2.2]])
        a2 = awkward_numba.JaggedArray.fromcounts([2, 0, 1], a)
        z = test(a, starts, stops, content)
        assert z.tolist() == [[1.1, 2.2, 3.3], [], [4.4, 5.5]]
        assert isinstance(z, awkward_numba.JaggedArray)
        assert type(z) is not awkward.JaggedArray
        starts2 = numpy.array([0, 2, 2])
        stops2 = numpy.array([2, 2, 3])
        z2 = test(a2, starts2, stops2, z)
        assert z2.tolist() == [[[1.1, 2.2, 3.3], []], [], [[4.4, 5.5]]]
        assert isinstance(z2, awkward_numba.JaggedArray)
        assert type(z2) is not awkward.JaggedArray

    def test_numba_compact(self):
        @numba.njit
        def test(x):
            return x.compact()
        starts = numpy.array([0, 3, 4])
        stops = numpy.array([3, 3, 6])
        content = numpy.array([1.1, 2.2, 3.3, 999, 4.4, 5.5])
        a = JaggedArray(starts, stops, content)
        a2 = JaggedArray([1, 0, 0], [3, 0, 1], a)
        z = test(a)
        assert z.tolist() == [[1.1, 2.2, 3.3], [], [4.4, 5.5]]
        assert z.content.tolist() == [1.1, 2.2, 3.3, 4.4, 5.5]
        assert z.iscompact
        z2 = test(a2)
        assert z2.tolist() == [[[], [4.4, 5.5]], [], [[1.1, 2.2, 3.3]]]
        assert z2.content.tolist() == [[], [4.4, 5.5], [1.1, 2.2, 3.3]]
        assert z2.iscompact

    def test_numba_flatten(self):
        @numba.njit
        def test(x):
            return x.flatten()
        starts = numpy.array([0, 3, 4])
        stops = numpy.array([3, 3, 6])
        content = numpy.array([1.1, 2.2, 3.3, 999, 4.4, 5.5])
        a = JaggedArray(starts, stops, content)
        z = test(a)
        assert z.tolist() == [1.1, 2.2, 3.3, 4.4, 5.5]

    def test_numba_len(self):
        a = JaggedArray.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
        a2 = JaggedArray.fromcounts([2, 1], a)
        @numba.njit
        def test1(x):
            return len(x)
        assert test1(a) == 3
        assert test1(a2) == 2

    def test_numba_getitem_integer(self):
        a = JaggedArray.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
        a2 = JaggedArray.fromcounts([2, 0, 1], a)
        @numba.njit
        def test1(x, i, j):
            return x[i][j]
        assert test1(a, 0, 0) == 1.1
        assert test1(a, 0, 1) == 2.2
        assert test1(a, 0, 2) == 3.3
        assert test1(a, 2, 0) == 4.4
        assert test1(a, 2, 1) == 5.5
        @numba.njit
        def test2(x, i):
            return x[i]
        assert test2(a, 0).tolist() == [1.1, 2.2, 3.3]
        assert test2(a, 1).tolist() == []
        assert test2(a, 2).tolist() == [4.4, 5.5]
        assert test2(a2, 0).tolist() == [[1.1, 2.2, 3.3], []]
        assert test2(a2, 1).tolist() == []
        assert test2(a2, 2).tolist() == [[4.4, 5.5]]
        assert test2(a2, 0).content.tolist() == a.content.tolist()

    def test_numba_getitem_slice(self):
        a = JaggedArray.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
        a2 = JaggedArray.fromcounts([2, 0, 1], a)   # [[[1.1, 2.2, 3.3], []], [], [[4.4, 5.5]]]
        @numba.njit
        def test1(x, i, j):
            return x[i:j]
        assert test1(a, 0, 2).tolist() == [[1.1, 2.2, 3.3], []]
        assert test1(a, 1, 3).tolist() == [[], [4.4, 5.5]]
        assert test1(a2, 0, 2).tolist() == [[[1.1, 2.2, 3.3], []], []]
        assert test1(a2, 1, 3).tolist() == [[], [[4.4, 5.5]]]

    def test_numba_getitem_intarray(self):
        a = JaggedArray.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
        starts = numpy.array([0, 3, 4])
        stops = numpy.array([3, 3, 6])
        content = numpy.array([1.1, 2.2, 3.3, 999, 4.4, 5.5])
        a2 = JaggedArray(starts, stops, content)
        index = numpy.array([2, 2, 0, 1])
        @numba.njit
        def test1(x, i):
            return x[i]
        z = test1(a, index)
        assert z.tolist() == [[4.4, 5.5], [4.4, 5.5], [1.1, 2.2, 3.3], []]
        assert z.content.tolist() == [1.1, 2.2, 3.3, 4.4, 5.5]
        z2 = test1(a2, index)
        assert z2.tolist() == [[4.4, 5.5], [4.4, 5.5], [1.1, 2.2, 3.3], []]
        assert z2.content.tolist() == [1.1, 2.2, 3.3, 999, 4.4, 5.5]
        @numba.njit
        def test2(x, i):
            return x[i].compact()
        z = test2(a, index)
        assert z.tolist() == [[4.4, 5.5], [4.4, 5.5], [1.1, 2.2, 3.3], []]
        assert z.content.tolist() == [4.4, 5.5, 4.4, 5.5, 1.1, 2.2, 3.3]
        z2 = test2(a2, index)
        assert z2.tolist() == [[4.4, 5.5], [4.4, 5.5], [1.1, 2.2, 3.3], []]
        assert z2.content.tolist() == [4.4, 5.5, 4.4, 5.5, 1.1, 2.2, 3.3]
        a3 = JaggedArray.fromcounts([2, 0, 1], a)
        assert test1(a3, index).tolist() == [[[4.4, 5.5]], [[4.4, 5.5]], [[1.1, 2.2, 3.3], []], []]

    def test_numba_getitem_boolarray(self):
        a = JaggedArray.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
        starts = numpy.array([0, 3, 4])
        stops = numpy.array([3, 3, 6])
        content = numpy.array([1.1, 2.2, 3.3, 999, 4.4, 5.5])
        a2 = JaggedArray(starts, stops, content)
        index = numpy.array([False, True, True])
        @numba.njit
        def test1(x, i):
            return x[i]
        z = test1(a, index)
        assert z.tolist() == [[], [4.4, 5.5]]
        assert z.content.tolist() == [1.1, 2.2, 3.3, 4.4, 5.5]
        z2 = test1(a2, index)
        assert z2.tolist() == [[], [4.4, 5.5]]
        assert z2.content.tolist() == [1.1, 2.2, 3.3, 999, 4.4, 5.5]
        @numba.njit
        def test2(x, i):
            return x[i].compact()
        z = test2(a, index)
        assert z.tolist() == [[], [4.4, 5.5]]
        assert z.content.tolist() == [4.4, 5.5]
        z2 = test2(a2, index)
        assert z2.tolist() == [[], [4.4, 5.5]]
        assert z2.content.tolist() == [4.4, 5.5]
        a3 = JaggedArray.fromcounts([2, 0, 1], a)
        assert test1(a3, index).tolist() == [[], [[4.4, 5.5]]]

    def test_numba_getitem_tuple_integer(self):
        a = JaggedArray.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
        @numba.njit
        def test1(x, i):
            return x[i,]
        assert test1(a, 2).tolist() == [4.4, 5.5]

        a2 = JaggedArray.fromcounts([2, 0, 1], a)  # [[[1.1, 2.2, 3.3], []], [], [[4.4, 5.5]]]
        @numba.njit
        def test2(x, i, j):
            return x[i, j]
        assert test2(a2, 0, 0).tolist() == [1.1, 2.2, 3.3]

    def test_numba_getitem_tuple_slice(self):
        a = JaggedArray.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
        a2 = JaggedArray.fromcounts([2, 0, 1], a)   # [[[1.1, 2.2, 3.3], []], [], [[4.4, 5.5]]]
        @numba.njit
        def test1(x, i, j):
            return x[i:j,]
        assert test1(a, 0, 2).tolist() == [[1.1, 2.2, 3.3], []]
        assert test1(a, 1, 3).tolist() == [[], [4.4, 5.5]]
        assert test1(a2, 0, 2).tolist() == [[[1.1, 2.2, 3.3], []], []]
        assert test1(a2, 1, 3).tolist() == [[], [[4.4, 5.5]]]

        a3 = JaggedArray.fromcounts([2, 1], a)   # [[[1.1, 2.2, 3.3], []], [[4.4, 5.5]]]
        assert a3[0:2, 0].tolist() == [[1.1, 2.2, 3.3], [4.4, 5.5]]
        @numba.njit
        def test2(x, i, j, k):
            return x[i:j, k]
        assert test2(a3, 0, 2, 0).tolist() == [[1.1, 2.2, 3.3], [4.4, 5.5]]

    def test_numba_getitem_tuple_slice_integer(self):
        a = JaggedArray.fromiter([[1.1, 2.2, 3.3], [4.4, 5.5], [6.6, 7.7], [8.8, 9.9]])
        a2 = JaggedArray.fromcounts([2, 2], a)
        @numba.njit
        def test3(x, i, j, k):
            return x[i:j, k]
        assert test3(a, 0, 2, 1).tolist() == [2.2, 5.5]
        assert test3(a2, 0, 2, 1).tolist() == [[4.4, 5.5], [8.8, 9.9]]

    def test_numba_getitem_tuple_slice_slice(self):
        a = JaggedArray.fromiter([[1.1, 2.2, 3.3], [4.4, 5.5], [6.6, 7.7], [8.8, 9.9]])
        a2 = JaggedArray.fromcounts([2, 2], a)
        @numba.njit
        def test3(x, i, j, k, l):
            return x[i:j, k:l]
        assert test3(a, 0, 2, -2, None).tolist() == [[2.2, 3.3], [4.4, 5.5]]
        assert test3(a2, 0, 2, 1, 2).tolist() == [[[4.4, 5.5]], [[8.8, 9.9]]]

        @numba.njit
        def test4(x, i, j, k, l):
            return x[i:j, k:l:1]
        assert test4(a, 0, 2, -2, None).tolist() == [[2.2, 3.3], [4.4, 5.5]]
        assert test4(a2, 0, 2, 1, 2).tolist() == [[[4.4, 5.5]], [[8.8, 9.9]]]

    def test_numba_getitem_tuple_boolarray(self):
        a = JaggedArray.fromiter([[1.1, 2.2, 3.3], [4.4, 5.5], [6.6, 7.7], [8.8, 9.9]])
        a2 = JaggedArray.fromcounts([2, 2], a)   # [[[1.1, 2.2, 3.3], [4.4, 5.5]], [[6.6, 7.7], [8.8, 9.9]]]
        @numba.njit
        def test1(x, i, j):
            return x[i, j]
        assert test1(a, numpy.array([True, False, True, False]), 1).tolist() == [2.2, 7.7]
        assert test1(a2, numpy.array([False, True]), 1).tolist() == [[8.8, 9.9]]

    def test_numba_getitem_tuple_intarray(self):
        a = JaggedArray.fromiter([[1.1, 2.2, 3.3], [4.4, 5.5], [6.6, 7.7], [8.8, 9.9]])
        a2 = JaggedArray.fromcounts([2, 2], a)   # [[[1.1, 2.2, 3.3], [4.4, 5.5]], [[6.6, 7.7], [8.8, 9.9]]]
        @numba.njit
        def test1(x, i, j):
            return x[i, j]
        assert test1(a, numpy.array([2, 0]), 1).tolist() == [7.7, 2.2]
        assert test1(a2, numpy.array([1, 0, 0]), 1).tolist() == [[8.8, 9.9], [4.4, 5.5], [4.4, 5.5]]

    def test_numba_getitem_tuple_slice_boolarray(self):
        a = numpy.arange(36).reshape(4, 3, 3)
        a2 = awkward.fromiter(a)
        @numba.njit
        def test1(x, i):
            return x[1:3, i]
        assert test1(a, numpy.array([True, False, True])).tolist() == [[[9, 10, 11], [15, 16, 17]], [[18, 19, 20], [24, 25, 26]]]
        assert test1(a2, numpy.array([True, False, True])).tolist() == [[[9, 10, 11], [15, 16, 17]], [[18, 19, 20], [24, 25, 26]]]
        @numba.njit
        def test2(x, i, j):
            return x[1:3, i, j]
        assert test2.py_func(a, numpy.array([True, False, True]), numpy.array([True, True, False])).tolist() == [[9, 16], [18, 25]]
        assert test2(a2, numpy.array([True, False, True]), numpy.array([True, True, False])).tolist() == [[9, 16], [18, 25]]
        a = numpy.arange(27).reshape(3, 3, 3)
        a2 = awkward.fromiter(a)
        @numba.njit
        def test3(x, i, j):
            return x[i, j]
        assert test3.py_func(a, numpy.array([True, False, True]), numpy.array([True, True, False])).tolist() == [[0, 1, 2], [21, 22, 23]]
        assert test3(a2, numpy.array([True, False, True]), numpy.array([True, True, False])).tolist() == [[0, 1, 2], [21, 22, 23]]
        @numba.njit
        def test4(x, i, j):
            return x[i, :, j]
        assert test4.py_func(a, numpy.array([True, False, True]), numpy.array([True, True, False])).tolist() == [[0, 3, 6], [19, 22, 25]]
        assert test4(a2, numpy.array([True, False, True]), numpy.array([True, True, False])).tolist() == [[0, 3, 6], [19, 22, 25]]

    def test_numba_getitem_tuple_slice_intarray(self):
        a = numpy.arange(36).reshape(4, 3, 3)
        a2 = awkward.fromiter(a)
        @numba.njit
        def test1(x, i):
            return x[1:3, i]
        assert test1(a, numpy.array([1, 0, 2])).tolist() == [[[12, 13, 14], [9, 10, 11], [15, 16, 17]], [[21, 22, 23], [18, 19, 20], [24, 25, 26]]]
        assert test1(a2, numpy.array([1, 0, 2])).tolist() == [[[12, 13, 14], [9, 10, 11], [15, 16, 17]], [[21, 22, 23], [18, 19, 20], [24, 25, 26]]]

    def test_numba_getitem_jagged_boolarray(self):
        a = JaggedArray.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
        a2 = JaggedArray.fromcounts([2, 0, 1], a)
        @numba.njit
        def test1(x, i):
            return x[i]
        assert test1(a, awkward.fromiter([[True, False, True], [], [False, True]])).tolist() == [[1.1, 3.3], [], [5.5]]
        assert test1(a2, awkward.fromiter([[True, False], [], [True]])).tolist() == [[[1.1, 2.2, 3.3]], [], [[4.4, 5.5]]]
        assert test1(a2, awkward.fromiter([[[True, False, True], []], [], [[False, True]]])).tolist() == [[[1.1, 3.3], []], [], [[5.5]]]

    def test_numba_getitem_jagged_intarray(self):
        a = JaggedArray.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
        a2 = JaggedArray.fromcounts([2, 0, 1], a)
        @numba.njit
        def test1(x, i):
            return x[i]
        assert test1(a, awkward.fromiter([[2, 0, 0], [], [1]])).tolist() == [[3.3, 1.1, 1.1], [], [5.5]]
        assert test1(a2, awkward.fromiter([[1, 0], [], [0]])).tolist() == [[[], [1.1, 2.2, 3.3]], [], [[4.4, 5.5]]]
        assert test1(a2, awkward.fromiter([[[2, 0, 0], []], [], [[1]]])).tolist() == [[[3.3, 1.1, 1.1], []], [], [[5.5]]]

    def test_numba_getiter(self):
        a = JaggedArray.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
        a2 = JaggedArray.fromcounts([2, 0, 1], a)
        @numba.njit
        def test1(x):
            out = 0.0
            for xi in x:
                for xij in xi:
                    out += xij
            return out
        assert test1(a) == 16.5
        @numba.njit
        def test2(x):
            out = 0.0
            for xi in x:
                for xij in xi:
                    for xijk in xij:
                        out += xijk
            return out
        assert test2(a2) == 16.5

    def test_numba_reducers(self):
        a = awkward.numba.fromiter([[0, numpy.nan, 3.3], [], [4, 5]])
        a2 = awkward.numba.JaggedArray.fromcounts([2, 0, 1], a)
        assert numba.njit()(lambda x: x.any())(a).tolist() == [True, False, True]
        assert numba.njit()(lambda x: x.any())(a2).tolist() == [[True, False], [], [True]]
        assert numba.njit()(lambda x: x.all())(a).tolist() == [False, True, True]
        assert numba.njit()(lambda x: x.all())(a2).tolist() == [[False, True], [], [True]]
        assert numba.njit()(lambda x: x.count())(a).tolist() == [2, 0, 2]
        assert numba.njit()(lambda x: x.count())(a2).tolist() == [[2, 0], [], [2]]
        assert numba.njit()(lambda x: x.count_nonzero())(a).tolist() == [1, 0, 2]
        assert numba.njit()(lambda x: x.count_nonzero())(a2).tolist() == [[1, 0], [], [2]]
        assert numba.njit()(lambda x: x.sum())(a).tolist() == [3.3, 0.0, 9.0]
        assert numba.njit()(lambda x: x.sum())(a2).tolist() == [[3.3, 0.0], [], [9.0]]
        assert numba.njit()(lambda x: x.prod())(a).tolist() == [0.0, 1.0, 20.0]
        assert numba.njit()(lambda x: x.prod())(a2).tolist() == [[0.0, 1.0], [], [20.0]]
        assert numba.njit()(lambda x: x.min())(a).tolist() == [0.0, numpy.inf, 4.0]
        assert numba.njit()(lambda x: x.min())(a2).tolist() == [[0.0, numpy.inf], [], [4.0]]
        assert numba.njit()(lambda x: x.max())(a).tolist() == [3.3, -numpy.inf, 5.0]
        assert numba.njit()(lambda x: x.max())(a2).tolist() == [[3.3, -numpy.inf], [], [5.0]]

        a = awkward.numba.fromiter([[1, 2, 3], [], [4, 5]])
        a2 = awkward.numba.JaggedArray.fromcounts([2, 0, 1], a)
        assert numba.njit()(lambda x: x.any())(a).tolist() == [True, False, True]
        assert numba.njit()(lambda x: x.any())(a2).tolist() == [[True, False], [], [True]]
        assert numba.njit()(lambda x: x.all())(a).tolist() == [True, True, True]
        assert numba.njit()(lambda x: x.all())(a2).tolist() == [[True, True], [], [True]]
        assert numba.njit()(lambda x: x.count())(a).tolist() == [3, 0, 2]
        assert numba.njit()(lambda x: x.count())(a2).tolist() == [[3, 0], [], [2]]
        assert numba.njit()(lambda x: x.count_nonzero())(a).tolist() == [3, 0, 2]
        assert numba.njit()(lambda x: x.count_nonzero())(a2).tolist() == [[3, 0], [], [2]]
        assert numba.njit()(lambda x: x.sum())(a).tolist() == [6, 0, 9]
        assert numba.njit()(lambda x: x.sum())(a2).tolist() == [[6, 0], [], [9]]
        assert numba.njit()(lambda x: x.prod())(a).tolist() == [6, 1, 20]
        assert numba.njit()(lambda x: x.prod())(a2).tolist() == [[6, 1], [], [20]]
        assert numba.njit()(lambda x: x.min())(a).tolist() == [1, numpy.iinfo(numpy.int64).max, 4]
        assert numba.njit()(lambda x: x.min())(a2).tolist() == [[1, numpy.iinfo(numpy.int64).max], [], [4]]
        assert numba.njit()(lambda x: x.max())(a).tolist() == [3, numpy.iinfo(numpy.int64).min, 5]
        assert numba.njit()(lambda x: x.max())(a2).tolist() == [[3, numpy.iinfo(numpy.int64).min], [], [5]]

        a = awkward.numba.fromiter([[True, False, True], [], [False, True]])
        a2 = awkward.numba.JaggedArray.fromcounts([2, 0, 1], a)
        assert numba.njit()(lambda x: x.any())(a).tolist() == [True, False, True]
        assert numba.njit()(lambda x: x.any())(a2).tolist() == [[True, False], [], [True]]
        assert numba.njit()(lambda x: x.all())(a).tolist() == [False, True, False]
        assert numba.njit()(lambda x: x.all())(a2).tolist() == [[False, True], [], [False]]
        assert numba.njit()(lambda x: x.count())(a).tolist() == [3, 0, 2]
        assert numba.njit()(lambda x: x.count())(a2).tolist() == [[3, 0], [], [2]]
        assert numba.njit()(lambda x: x.count_nonzero())(a).tolist() == [2, 0, 1]
        assert numba.njit()(lambda x: x.count_nonzero())(a2).tolist() == [[2, 0], [], [1]]
        assert numba.njit()(lambda x: x.sum())(a).tolist() == [True, False, True]
        assert numba.njit()(lambda x: x.sum())(a2).tolist() == [[True, False], [], [True]]
        assert numba.njit()(lambda x: x.prod())(a).tolist() == [False, True, False]
        assert numba.njit()(lambda x: x.prod())(a2).tolist() == [[False, True], [], [False]]
        assert numba.njit()(lambda x: x.min())(a).tolist() == [False, True, False]
        assert numba.njit()(lambda x: x.min())(a2).tolist() == [[False, True], [], [False]]
        assert numba.njit()(lambda x: x.max())(a).tolist() == [True, False, True]
        assert numba.njit()(lambda x: x.max())(a2).tolist() == [[True, False], [], [True]]
