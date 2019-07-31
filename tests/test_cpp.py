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
awkward_cpp = pytest.importorskip("awkward.cpp")
from awkward.cpp import *

class Test(unittest.TestCase):
    def runTest(self):
        pass

    def test_cpp_unbox(self):
        a = JaggedArray.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
        a2 = JaggedArray.fromcounts([2, 0, 1], a)
        def test(x):
            return 3.14
        test(a)
        test(a2)

    def test_cpp_box(self):
        a = JaggedArray.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
        a2 = JaggedArray.fromcounts([2, 0, 1], a)
        def test(x):
            return x
        assert test(a).tolist() == a.tolist()
        assert test(a2).tolist() == a2.tolist()

    def test_cpp_len(self):
        a = JaggedArray.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
        a2 = JaggedArray.fromcounts([2, 1], a)
        def test1(x):
            return len(x)
        assert test1(a) == 3
        assert test1(a2) == 2

    def test_cpp_getitem_integer(self):
        a = JaggedArray.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
        a2 = JaggedArray.fromcounts([2, 0, 1], a)
        def test1(x, i, j):
            return x[i][j]
        assert test1(a, 0, 0) == 1.1
        assert test1(a, 0, 1) == 2.2
        assert test1(a, 0, 2) == 3.3
        assert test1(a, 2, 0) == 4.4
        assert test1(a, 2, 1) == 5.5
        def test2(x, i):
            return x[i]
        assert test2(a, 0).tolist() == [1.1, 2.2, 3.3]
        assert test2(a, 1).tolist() == []
        assert test2(a, 2).tolist() == [4.4, 5.5]
        assert test2(a2, 0).tolist() == [[1.1, 2.2, 3.3], []]
        assert test2(a2, 1).tolist() == []
        assert test2(a2, 2).tolist() == [[4.4, 5.5]]
        assert test2(a2, 0).content.tolist() == a.content.tolist()

    def test_cpp_getitem_slice(self):
        a = JaggedArray.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
        a2 = JaggedArray.fromcounts([2, 0, 1], a)   # [[[1.1, 2.2, 3.3], []], [], [[4.4, 5.5]]]
        def test1(x, i, j):
            return x[i:j]
        assert test1(a, 0, 2).tolist() == [[1.1, 2.2, 3.3], []]
        assert test1(a, 1, 3).tolist() == [[], [4.4, 5.5]]
        assert test1(a2, 0, 2).tolist() == [[[1.1, 2.2, 3.3], []], []]
        assert test1(a2, 1, 3).tolist() == [[], [[4.4, 5.5]]]

    def test_cpp_getitem_intarray(self):
        a = JaggedArray.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
        starts = numpy.array([0, 3, 4])
        stops = numpy.array([3, 3, 6])
        content = numpy.array([1.1, 2.2, 3.3, 999, 4.4, 5.5])
        a2 = JaggedArray(starts, stops, content)
        index = numpy.array([2, 2, 0, 1])
        def test1(x, i):
            return x[i]
        z = test1(a, index)
        assert z.tolist() == [[4.4, 5.5], [4.4, 5.5], [1.1, 2.2, 3.3], []]
        assert z.content.tolist() == [1.1, 2.2, 3.3, 4.4, 5.5]
        z2 = test1(a2, index)
        assert z2.tolist() == [[4.4, 5.5], [4.4, 5.5], [1.1, 2.2, 3.3], []]
        assert z2.content.tolist() == [1.1, 2.2, 3.3, 999, 4.4, 5.5]
        #def test2(x, i):
        #    return x[i].compact()
        #z = test2(a, index)
        #assert z.tolist() == [[4.4, 5.5], [4.4, 5.5], [1.1, 2.2, 3.3], []]
        #ssert z.content.tolist() == [4.4, 5.5, 4.4, 5.5, 1.1, 2.2, 3.3]
        #z2 = test2(a2, index)
        #assert z2.tolist() == [[4.4, 5.5], [4.4, 5.5], [1.1, 2.2, 3.3], []]
        #assert z2.content.tolist() == [4.4, 5.5, 4.4, 5.5, 1.1, 2.2, 3.3]
        #a3 = JaggedArray.fromcounts([2, 0, 1], a)
        #assert test1(a3, index).tolist() == [[[4.4, 5.5]], [[4.4, 5.5]], [[1.1, 2.2, 3.3], []], []]

    def test_cpp_getitem_boolarray(self):
        a = JaggedArray.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
        starts = numpy.array([0, 3, 4])
        stops = numpy.array([3, 3, 6])
        content = numpy.array([1.1, 2.2, 3.3, 999, 4.4, 5.5])
        a2 = JaggedArray(starts, stops, content)
        index = numpy.array([False, True, True])
        def test1(x, i):
            return x[i]
        z = test1(a, index)
        assert z.tolist() == [[], [4.4, 5.5]]
        assert z.content.tolist() == [1.1, 2.2, 3.3, 4.4, 5.5]
        z2 = test1(a2, index)
        assert z2.tolist() == [[], [4.4, 5.5]]
        assert z2.content.tolist() == [1.1, 2.2, 3.3, 999, 4.4, 5.5]
        #def test2(x, i):
        #    return x[i].compact()
        #z = test2(a, index)
        #assert z.tolist() == [[], [4.4, 5.5]]
        #assert z.content.tolist() == [4.4, 5.5]
        #z2 = test2(a2, index)
        #assert z2.tolist() == [[], [4.4, 5.5]]
        #assert z2.content.tolist() == [4.4, 5.5]
        #a3 = JaggedArray.fromcounts([2, 0, 1], a)
        #assert test1(a3, index).tolist() == [[], [[4.4, 5.5]]]

    def test_cpp_getitem_tuple_integer(self):
        a = JaggedArray.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
        def test1(x, i):
            return x[i,]
        assert test1(a, 2).tolist() == [4.4, 5.5]

        a2 = JaggedArray.fromcounts([2, 0, 1], a)  # [[[1.1, 2.2, 3.3], []], [], [[4.4, 5.5]]]
        def test2(x, i, j):
            return x[i, j]
        assert test2(a2, 0, 0).tolist() == [1.1, 2.2, 3.3]

    def test_cpp_getitem_tuple_slice(self):
        a = JaggedArray.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
        a2 = JaggedArray.fromcounts([2, 0, 1], a)   # [[[1.1, 2.2, 3.3], []], [], [[4.4, 5.5]]]
        def test1(x, i, j):
            return x[i:j,]
        assert test1(a, 0, 2).tolist() == [[1.1, 2.2, 3.3], []]
        assert test1(a, 1, 3).tolist() == [[], [4.4, 5.5]]
        assert test1(a2, 0, 2).tolist() == [[[1.1, 2.2, 3.3], []], []]
        assert test1(a2, 1, 3).tolist() == [[], [[4.4, 5.5]]]

        a3 = JaggedArray.fromcounts([2, 1], a)   # [[[1.1, 2.2, 3.3], []], [[4.4, 5.5]]]
        assert a3[0:2, 0].tolist() == [[1.1, 2.2, 3.3], [4.4, 5.5]]
        def test2(x, i, j, k):
            return x[i:j, k]
        assert test2(a3, 0, 2, 0).tolist() == [[1.1, 2.2, 3.3], [4.4, 5.5]]

    def test_cpp_getitem_tuple_slice_integer(self):
        a = JaggedArray.fromiter([[1.1, 2.2, 3.3], [4.4, 5.5], [6.6, 7.7], [8.8, 9.9]])
        a2 = JaggedArray.fromcounts([2, 2], a)
        def test3(x, i, j, k):
            return x[i:j, k]
        assert test3(a, 0, 2, 1).tolist() == [2.2, 5.5]
        assert test3(a2, 0, 2, 1).tolist() == [[4.4, 5.5], [8.8, 9.9]]

    def test_cpp_getitem_tuple_slice_slice(self):
        a = JaggedArray.fromiter([[1.1, 2.2, 3.3], [4.4, 5.5], [6.6, 7.7], [8.8, 9.9]])
        a2 = JaggedArray.fromcounts([2, 2], a)
        def test3(x, i, j, k, l):
            return x[i:j, k:l]
        assert test3(a, 0, 2, -2, None).tolist() == [[2.2, 3.3], [4.4, 5.5]]
        assert test3(a2, 0, 2, 1, 2).tolist() == [[[4.4, 5.5]], [[8.8, 9.9]]]

        def test4(x, i, j, k, l):
            return x[i:j, k:l:1]
        assert test4(a, 0, 2, -2, None).tolist() == [[2.2, 3.3], [4.4, 5.5]]
        assert test4(a2, 0, 2, 1, 2).tolist() == [[[4.4, 5.5]], [[8.8, 9.9]]]

    def test_cpp_getitem_tuple_boolarray(self):
        a = JaggedArray.fromiter([[1.1, 2.2, 3.3], [4.4, 5.5], [6.6, 7.7], [8.8, 9.9]])
        a2 = JaggedArray.fromcounts([2, 2], a)   # [[[1.1, 2.2, 3.3], [4.4, 5.5]], [[6.6, 7.7], [8.8, 9.9]]]
        def test1(x, i, j):
            return x[i, j]
        assert test1(a, numpy.array([True, False, True, False]), 1).tolist() == [2.2, 7.7]
        assert test1(a2, numpy.array([False, True]), 1).tolist() == [[8.8, 9.9]]

    def test_cpp_getitem_tuple_intarray(self):
        a = JaggedArray.fromiter([[1.1, 2.2, 3.3], [4.4, 5.5], [6.6, 7.7], [8.8, 9.9]])
        a2 = JaggedArray.fromcounts([2, 2], a)   # [[[1.1, 2.2, 3.3], [4.4, 5.5]], [[6.6, 7.7], [8.8, 9.9]]]
        def test1(x, i, j):
            return x[i, j]
        assert test1(a, numpy.array([2, 0]), 1).tolist() == [7.7, 2.2]
        assert test1(a2, numpy.array([1, 0, 0]), 1).tolist() == [[8.8, 9.9], [4.4, 5.5], [4.4, 5.5]]

    def test_cpp_getitem_jagged_boolarray(self):
        a = JaggedArray.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
        a2 = JaggedArray.fromcounts([2, 0, 1], a)
        def test1(x, i):
            return x[i]
        assert test1(a, JaggedArray.fromiter([[True, False, True], [], [False, True]])).tolist() == [[1.1, 3.3], [], [5.5]]
        assert test1(a2, JaggedArray.fromiter([[True, False], [], [True]])).tolist() == [[[1.1, 2.2, 3.3]], [], [[4.4, 5.5]]]
        assert test1(a2, JaggedArray.fromiter([[[True, False, True], []], [], [[False, True]]])).tolist() == [[[1.1, 3.3], []], [], [[5.5]]]

    def test_cpp_getitem_jagged_intarray(self):
        a = JaggedArray.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
        a2 = JaggedArray.fromcounts([2, 0, 1], a)
        def test1(x, i):
            return x[i]
        assert test1(a, JaggedArray.fromiter([[2, 0, 0], [], [1]])).tolist() == [[3.3, 1.1, 1.1], [], [5.5]]
        assert test1(a2, JaggedArray.fromiter([[1, 0], [], [0]])).tolist() == [[[], [1.1, 2.2, 3.3]], [], [[4.4, 5.5]]]
        assert test1(a2, JaggedArray.fromiter([[[2, 0, 0], []], [], [[1]]])).tolist() == [[[3.3, 1.1, 1.1], []], [], [[5.5]]]

    def test_cpp_offsets2parents(self):
        offsets = numpy.array([0, 2, 4, 4, 7], dtype=numpy.int64)
        parents = JaggedArray.offsets2parents(offsets)
        assert parents.tolist() == [0, 0, 1, 1, 3, 3, 3]

    def test_cpp_counts2offsets(self):
        counts = numpy.array([4, 0, 3, 4, 1], dtype=numpy.int64)
        offsets = JaggedArray.counts2offsets(counts)
        assert offsets.tolist() == [0, 4, 4, 7, 11, 12]

    def test_cpp_startsstops2parents(self):
        starts = numpy.array([0, 4, 5, 9], dtype=numpy.int64)
        stops = numpy.array([1, 6, 7, 10], dtype=numpy.int64)
        parents = JaggedArray.startsstops2parents(starts, stops)
        assert parents.tolist() == [0, -1, -1, -1, 1, 2, 2, -1, -1, 3]

    def test_cpp_parents2startsstops(self):
        parents = numpy.array([-1, 0, 0, -1, 2, 2, 2, 3], dtype=numpy.int64)
        startsstops = JaggedArray.parents2startsstops(parents)
        starts = startsstops[0]
        stops = startsstops[1]
        assert starts.tolist() == [1, 0, 4, 7] and stops.tolist() == [3, 0, 7, 8]

    def test_cpp_uniques2offsetsparents(self):
        uniques = numpy.array([0, 3, 4, 6, 8, 8, 9], dtype=numpy.int64)
        offsetsparents = JaggedArray.uniques2offsetsparents(uniques)
        offsets = offsetsparents[0]
        parents = offsetsparents[1]
        assert offsets.tolist() == [0, 1, 2, 3, 4, 6, 7] and parents.tolist() == [0, 1, 2, 3, 4, 4, 5]

    def test_cpp_init(self):
        # This depends on all setters working,
        # as well as str and getitem methods
        a = numpy.array([1, 4, 1, 0])
        b = numpy.array([2, 9, 5, 10])
        c = numpy.arange(10, dtype=numpy.int16)
        test = awkward.cpp.JaggedArray(a, b, c)
        assert str(test) == "[[1] [4 5 6 7 8] [1 2 3 4] [0 1 2 3 4 5 6 7 8 9]]"

    def test_cpp_deepcopy(self):
        a = JaggedArray.fromiter([[1, 5, 2], [], [-6.1]])
        b = a.deepcopy()
        assert a.tolist() == b.tolist()
        assert a.__repr__() != b.__repr__()
