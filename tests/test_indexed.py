#!/usr/bin/env python

# Copyright (c) 2018, DIANA-HEP
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

import unittest

import numpy

from awkward import *

class Test(unittest.TestCase):
    def runTest(self):
        pass

    def test_indexed_get(self):
        a = IndexedArray([3, 2, 4, 2, 2, 4, 0], [0.0, 1.1, 2.2, 3.3, 4.4])
        assert [x for x in a] == [3.3, 2.2, 4.4, 2.2, 2.2, 4.4, 0.0]
        assert [a[i] for i in range(len(a))] == [3.3, 2.2, 4.4, 2.2, 2.2, 4.4, 0.0]
        assert [a[i : i + 1].tolist() for i in range(len(a))] == [[3.3], [2.2], [4.4], [2.2], [2.2], [4.4], [0.0]]
        assert [a[i : i + 2].tolist() for i in range(len(a) - 1)] == [[3.3, 2.2], [2.2, 4.4], [4.4, 2.2], [2.2, 2.2], [2.2, 4.4], [4.4, 0.0]]
        assert a[:].tolist() == [3.3, 2.2, 4.4, 2.2, 2.2, 4.4, 0.0]
        assert a[[6, 5, 4, 3, 2, 1, 0]].tolist() == [0.0, 4.4, 2.2, 2.2, 4.4, 2.2, 3.3]
        assert a[[True, False, True, False, True, False, True]].tolist() == [3.3, 4.4, 2.2, 0.0]
        assert a[[-1, -2, -3, -4, -5, -6, -7]].tolist() == [0.0, 4.4, 2.2, 2.2, 4.4, 2.2, 3.3]

    def test_indexed_get2d_index(self):
        a = IndexedArray([[3, 2], [4, 2], [2, 0]], [0.0, 1.1, 2.2, 3.3, 4.4])
        assert [x.tolist() for x in a] == [[3.3, 2.2], [4.4, 2.2], [2.2, 0.0]]
        assert [a[i].tolist() for i in range(len(a))] == [[3.3, 2.2], [4.4, 2.2], [2.2, 0.0]]
        assert [a[i : i + 1].tolist() for i in range(len(a))] == [[[3.3, 2.2]], [[4.4, 2.2]], [[2.2, 0.0]]]
        assert [a[i : i + 2].tolist() for i in range(len(a) - 1)] == [[[3.3, 2.2], [4.4, 2.2]], [[4.4, 2.2], [2.2, 0.0]]]
        assert a[:].tolist() == [[3.3, 2.2], [4.4, 2.2], [2.2, 0.0]]
        assert a[[1, 1, 0]].tolist() == [[4.4, 2.2], [4.4, 2.2], [3.3, 2.2]]
        assert a[[True, False, True]].tolist() == [[3.3, 2.2], [2.2, 0.0]]
        assert a[[-2, -2, -3]].tolist() == [[4.4, 2.2], [4.4, 2.2], [3.3, 2.2]]

    def test_indexed_get2d_content(self):
        a = IndexedArray([3, 2, 4, 2, 2, 4, 0], [[0.0, 0.0], [1.1, 1.1], [2.2, 2.2], [3.3, 3.3], [4.4, 4.4]])
        assert [x.tolist() for x in a] == [[3.3, 3.3], [2.2, 2.2], [4.4, 4.4], [2.2, 2.2], [2.2, 2.2], [4.4, 4.4], [0.0, 0.0]]
        assert [a[i].tolist() for i in range(len(a))] == [[3.3, 3.3], [2.2, 2.2], [4.4, 4.4], [2.2, 2.2], [2.2, 2.2], [4.4, 4.4], [0.0, 0.0]]
        assert a[:].tolist() == [[3.3, 3.3], [2.2, 2.2], [4.4, 4.4], [2.2, 2.2], [2.2, 2.2], [4.4, 4.4], [0.0, 0.0]]
        assert a[[6, 5, 4, 3, 2, 1, 0]].tolist() == [[0.0, 0.0], [4.4, 4.4], [2.2, 2.2], [2.2, 2.2], [4.4, 4.4], [2.2, 2.2], [3.3, 3.3]]
        assert a[[True, False, True, False, True, False, True]].tolist() == [[3.3, 3.3], [4.4, 4.4], [2.2, 2.2], [0.0, 0.0]]

    def test_indexed_getstruct(self):
        a = IndexedArray([3, 2, 4, 2, 2, 4, 0], numpy.array([(0.0, 0.0), (1.1, 1.1), (2.2, 2.2), (3.3, 3.3), (4.4, 4.4)], dtype=[("a", float), ("b", float)]))
        assert [x.tolist() for x in a] == [(3.3, 3.3), (2.2, 2.2), (4.4, 4.4), (2.2, 2.2), (2.2, 2.2), (4.4, 4.4), (0.0, 0.0)]
        assert [a[i].tolist() for i in range(len(a))] == [(3.3, 3.3), (2.2, 2.2), (4.4, 4.4), (2.2, 2.2), (2.2, 2.2), (4.4, 4.4), (0.0, 0.0)]
        assert a[:].tolist() == [(3.3, 3.3), (2.2, 2.2), (4.4, 4.4), (2.2, 2.2), (2.2, 2.2), (4.4, 4.4), (0.0, 0.0)]
        assert a[[6, 5, 4, 3, 2, 1, 0]].tolist() == [(0.0, 0.0), (4.4, 4.4), (2.2, 2.2), (2.2, 2.2), (4.4, 4.4), (2.2, 2.2), (3.3, 3.3)]
        assert a[[True, False, True, False, True, False, True]].tolist() == [(3.3, 3.3), (4.4, 4.4), (2.2, 2.2), (0.0, 0.0)]

    def test_indexed_getempty(self):
        a = IndexedArray([], [0.0, 1.1, 2.2, 3.3, 4.4])
        assert a[:].tolist() == []

    def test_indexed_indexed(self):
        a = IndexedArray([6, 5, 4, 3, 2, 1, 0], IndexedArray([3, 2, 4, 2, 2, 4, 0], [0.0, 1.1, 2.2, 3.3, 4.4]))
        assert [x for x in a] == [0.0, 4.4, 2.2, 2.2, 4.4, 2.2, 3.3]
        assert [a[i] for i in range(len(a))] == [0.0, 4.4, 2.2, 2.2, 4.4, 2.2, 3.3]
        assert a[:].tolist() == [0.0, 4.4, 2.2, 2.2, 4.4, 2.2, 3.3]

        a = IndexedArray([6, 5, 4, 3, 6], IndexedArray([3, 2, 4, 2, 2, 4, 0], [0.0, 1.1, 2.2, 3.3, 4.4]))
        assert [x for x in a] == [0.0, 4.4, 2.2, 2.2, 0.0]
        assert [a[i] for i in range(len(a))] == [0.0, 4.4, 2.2, 2.2, 0.0]
        assert a[:].tolist() == [0.0, 4.4, 2.2, 2.2, 0.0]

    def test_indexed_ufunc(self):
        a = IndexedArray([3, 2, 4, 2, 2, 4, 0], [0.0, 1.1, 2.2, 3.3, 4.4])
        assert (a + 100).tolist() == [103.3, 102.2, 104.4, 102.2, 102.2, 104.4, 100.0]

    def test_indexed_table(self):
        a = IndexedArray([3, 2, 4, 0], Table(a=[0.0, 1.1, 2.2, 3.3, 4.4], b=[0, 1, 2, 3, 4]))
        a["c"] = numpy.array(["a", "b", "c", "d"])
        assert a["a"].tolist() == [3.3, 2.2, 4.4, 0.0]
        assert a["b"].tolist() == [3, 2, 4, 0]
        assert a["c"].tolist() == ["a", "b", "c", "d"]

    def test_sparse_get(self):
        a = SparseArray(10, [1, 3, 5, 7, 9], [100, 101, 102, 103, 104])

        assert a.tolist() == [0, 100, 0, 101, 0, 102, 0, 103, 0, 104]
        assert [a[i].tolist() for i in range(len(a))] == [0, 100, 0, 101, 0, 102, 0, 103, 0, 104]
        assert [a[i : i + 1].tolist() for i in range(len(a))] == [[0], [100], [0], [101], [0], [102], [0], [103], [0], [104]]
        assert [a[i : i + 2].tolist() for i in range(len(a) - 1)] == [[0, 100], [100, 0], [0, 101], [101, 0], [0, 102], [102, 0], [0, 103], [103, 0], [0, 104]]

        assert a[:].tolist() == [0, 100, 0, 101, 0, 102, 0, 103, 0, 104]
        assert a[1:].tolist() == [100, 0, 101, 0, 102, 0, 103, 0, 104]
        assert a[2:].tolist() == [0, 101, 0, 102, 0, 103, 0, 104]
        assert a[2:-1].tolist() == [0, 101, 0, 102, 0, 103, 0]
        assert a[2:-2].tolist() == [0, 101, 0, 102, 0, 103]
        assert a[:-2].tolist() == [0, 100, 0, 101, 0, 102, 0, 103]
        assert a[::2].tolist() == [0, 0, 0, 0, 0]
        assert a[1::2].tolist() == [100, 101, 102, 103, 104]
        assert a[2::2].tolist() == [0, 0, 0, 0]
        assert a[3::2].tolist() == [101, 102, 103, 104]
        assert a[::-1].tolist() == [104, 0, 103, 0, 102, 0, 101, 0, 100, 0]
        assert a[-2::-1].tolist() == [0, 103, 0, 102, 0, 101, 0, 100, 0]
        assert a[-3::-1].tolist() == [103, 0, 102, 0, 101, 0, 100, 0]
        assert a[-3:0:-1].tolist() == [103, 0, 102, 0, 101, 0, 100]
        assert a[-3:1:-1].tolist() == [103, 0, 102, 0, 101, 0]
        assert a[::-2].tolist() == [104, 103, 102, 101, 100]
        assert a[-1::-2].tolist() == [104, 103, 102, 101, 100]
        assert a[-2::-2].tolist() == [0, 0, 0, 0, 0]
        assert a[-3::-2].tolist() == [103, 102, 101, 100]
        assert a[[1, 3, 5, 7, 9]].tolist() == [100, 101, 102, 103, 104]
        assert a[[1, 3, 5, 7, 8, 9]].tolist() == [100, 101, 102, 103, 0, 104]
        assert a[[1, 3, 5, 9, 7]].tolist() == [100, 101, 102, 104, 103]
        assert a[[1, 3, 5, 9, 8, 7]].tolist() == [100, 101, 102, 104, 0, 103]
        assert a[[False, True, False, True, False, True, False, True, False, True]].tolist() == [100, 101, 102, 103, 104]
        assert a[[True, True, False, True, False, True, False, True, False, True]].tolist() == [0, 100, 101, 102, 103, 104]
        assert a[[True, True, True, True, False, True, False, True, False, True]].tolist() == [0, 100, 0, 101, 102, 103, 104]

        assert [a[1:][i].tolist() for i in range(9)] == [100, 0, 101, 0, 102, 0, 103, 0, 104]
        assert [a[[1, 3, 5, 9, 8, 7]][i].tolist() for i in range(6)] == [100, 101, 102, 104, 0, 103]
        assert [a[[True, True, True, True, False, True, False, True, False, True]][i].tolist() for i in range(7)] == [0, 100, 0, 101, 102, 103, 104]

        assert a.dense.tolist() == [0, 100, 0, 101, 0, 102, 0, 103, 0, 104]
        assert [a.dense[i].tolist() for i in range(len(a))] == [0, 100, 0, 101, 0, 102, 0, 103, 0, 104]
        assert [a.dense[i : i + 1].tolist() for i in range(len(a))] == [[0], [100], [0], [101], [0], [102], [0], [103], [0], [104]]
        assert [a.dense[i : i + 2].tolist() for i in range(len(a) - 1)] == [[0, 100], [100, 0], [0, 101], [101, 0], [0, 102], [102, 0], [0, 103], [103, 0], [0, 104]]

        assert a[:].dense.tolist() == [0, 100, 0, 101, 0, 102, 0, 103, 0, 104]
        assert a[1:].dense.tolist() == [100, 0, 101, 0, 102, 0, 103, 0, 104]
        assert a[2:].dense.tolist() == [0, 101, 0, 102, 0, 103, 0, 104]
        assert a[2:-1].dense.tolist() == [0, 101, 0, 102, 0, 103, 0]
        assert a[2:-2].dense.tolist() == [0, 101, 0, 102, 0, 103]
        assert a[:-2].dense.tolist() == [0, 100, 0, 101, 0, 102, 0, 103]
        assert a[::2].dense.tolist() == [0, 0, 0, 0, 0]
        assert a[1::2].dense.tolist() == [100, 101, 102, 103, 104]
        assert a[2::2].dense.tolist() == [0, 0, 0, 0]
        assert a[3::2].dense.tolist() == [101, 102, 103, 104]
        assert a[::-1].dense.tolist() == [104, 0, 103, 0, 102, 0, 101, 0, 100, 0]
        assert a[-2::-1].dense.tolist() == [0, 103, 0, 102, 0, 101, 0, 100, 0]
        assert a[-3::-1].dense.tolist() == [103, 0, 102, 0, 101, 0, 100, 0]
        assert a[-3:0:-1].dense.tolist() == [103, 0, 102, 0, 101, 0, 100]
        assert a[-3:1:-1].dense.tolist() == [103, 0, 102, 0, 101, 0]
        assert a[::-2].dense.tolist() == [104, 103, 102, 101, 100]
        assert a[-1::-2].dense.tolist() == [104, 103, 102, 101, 100]
        assert a[-2::-2].dense.tolist() == [0, 0, 0, 0, 0]
        assert a[-3::-2].dense.tolist() == [103, 102, 101, 100]

        assert [a[1:].dense[i].tolist() for i in range(9)] == [100, 0, 101, 0, 102, 0, 103, 0, 104]

    def test_sparse_get2d_content(self):
        a = SparseArray(10, [1, 3, 5, 7, 9], [[100], [101], [102], [103], [104]])

        assert a.tolist() == [[0], [100], [0], [101], [0], [102], [0], [103], [0], [104]]
        assert [a[i].tolist() for i in range(len(a))] == [[0], [100], [0], [101], [0], [102], [0], [103], [0], [104]]
        assert [a[i : i + 1].tolist() for i in range(len(a))] == [[[0]], [[100]], [[0]], [[101]], [[0]], [[102]], [[0]], [[103]], [[0]], [[104]]]
        assert [a[i : i + 2].tolist() for i in range(len(a) - 1)] == [[[0], [100]], [[100], [0]], [[0], [101]], [[101], [0]], [[0], [102]], [[102], [0]], [[0], [103]], [[103], [0]], [[0], [104]]]

        assert a[:].tolist() == [[0], [100], [0], [101], [0], [102], [0], [103], [0], [104]]
        assert a[1:].tolist() == [[100], [0], [101], [0], [102], [0], [103], [0], [104]]
        assert a[2:].tolist() == [[0], [101], [0], [102], [0], [103], [0], [104]]
        assert a[2:-1].tolist() == [[0], [101], [0], [102], [0], [103], [0]]
        assert a[2:-2].tolist() == [[0], [101], [0], [102], [0], [103]]
        assert a[:-2].tolist() == [[0], [100], [0], [101], [0], [102], [0], [103]]
        assert a[::2].tolist() == [[0], [0], [0], [0], [0]]
        assert a[1::2].tolist() == [[100], [101], [102], [103], [104]]
        assert a[2::2].tolist() == [[0], [0], [0], [0]]
        assert a[3::2].tolist() == [[101], [102], [103], [104]]
        assert a[::-1].tolist() == [[104], [0], [103], [0], [102], [0], [101], [0], [100], [0]]
        assert a[-2::-1].tolist() == [[0], [103], [0], [102], [0], [101], [0], [100], [0]]
        assert a[-3::-1].tolist() == [[103], [0], [102], [0], [101], [0], [100], [0]]
        assert a[-3:0:-1].tolist() == [[103], [0], [102], [0], [101], [0], [100]]
        assert a[-3:1:-1].tolist() == [[103], [0], [102], [0], [101], [0]]
        assert a[::-2].tolist() == [[104], [103], [102], [101], [100]]
        assert a[-1::-2].tolist() == [[104], [103], [102], [101], [100]]
        assert a[-2::-2].tolist() == [[0], [0], [0], [0], [0]]
        assert a[-3::-2].tolist() == [[103], [102], [101], [100]]
        assert a[[1, 3, 5, 7, 9]].tolist() == [[100], [101], [102], [103], [104]]
        assert a[[1, 3, 5, 7, 8, 9]].tolist() == [[100], [101], [102], [103], [0], [104]]
        assert a[[1, 3, 5, 9, 7]].tolist() == [[100], [101], [102], [104], [103]]
        assert a[[1, 3, 5, 9, 8, 7]].tolist() == [[100], [101], [102], [104], [0], [103]]
        assert a[[False, True, False, True, False, True, False, True, False, True]].tolist() == [[100], [101], [102], [103], [104]]
        assert a[[True, True, False, True, False, True, False, True, False, True]].tolist() == [[0], [100], [101], [102], [103], [104]]
        assert a[[True, True, True, True, False, True, False, True, False, True]].tolist() == [[0], [100], [0], [101], [102], [103], [104]]

        assert [a[1:][i].tolist() for i in range(9)] == [[100], [0], [101], [0], [102], [0], [103], [0], [104]]
        assert [a[[1, 3, 5, 9, 8, 7]][i].tolist() for i in range(6)] == [[100], [101], [102], [104], [0], [103]]
        assert [a[[True, True, True, True, False, True, False, True, False, True]][i].tolist() for i in range(7)] == [[0], [100], [0], [101], [102], [103], [104]]

        assert a.dense.tolist() == [[0], [100], [0], [101], [0], [102], [0], [103], [0], [104]]
        assert [a.dense[i].tolist() for i in range(len(a))] == [[0], [100], [0], [101], [0], [102], [0], [103], [0], [104]]
        assert [a.dense[i : i + 1].tolist() for i in range(len(a))] == [[[0]], [[100]], [[0]], [[101]], [[0]], [[102]], [[0]], [[103]], [[0]], [[104]]]
        assert [a.dense[i : i + 2].tolist() for i in range(len(a) - 1)] == [[[0], [100]], [[100], [0]], [[0], [101]], [[101], [0]], [[0], [102]], [[102], [0]], [[0], [103]], [[103], [0]], [[0], [104]]]

        assert a[:].dense.tolist() == [[0], [100], [0], [101], [0], [102], [0], [103], [0], [104]]
        assert a[1:].dense.tolist() == [[100], [0], [101], [0], [102], [0], [103], [0], [104]]
        assert a[2:].dense.tolist() == [[0], [101], [0], [102], [0], [103], [0], [104]]
        assert a[2:-1].dense.tolist() == [[0], [101], [0], [102], [0], [103], [0]]
        assert a[2:-2].dense.tolist() == [[0], [101], [0], [102], [0], [103]]
        assert a[:-2].dense.tolist() == [[0], [100], [0], [101], [0], [102], [0], [103]]
        assert a[::2].dense.tolist() == [[0], [0], [0], [0], [0]]
        assert a[1::2].dense.tolist() == [[100], [101], [102], [103], [104]]
        assert a[2::2].dense.tolist() == [[0], [0], [0], [0]]
        assert a[3::2].dense.tolist() == [[101], [102], [103], [104]]
        assert a[::-1].dense.tolist() == [[104], [0], [103], [0], [102], [0], [101], [0], [100], [0]]
        assert a[-2::-1].dense.tolist() == [[0], [103], [0], [102], [0], [101], [0], [100], [0]]
        assert a[-3::-1].dense.tolist() == [[103], [0], [102], [0], [101], [0], [100], [0]]
        assert a[-3:0:-1].dense.tolist() == [[103], [0], [102], [0], [101], [0], [100]]
        assert a[-3:1:-1].dense.tolist() == [[103], [0], [102], [0], [101], [0]]
        assert a[::-2].dense.tolist() == [[104], [103], [102], [101], [100]]
        assert a[-1::-2].dense.tolist() == [[104], [103], [102], [101], [100]]
        assert a[-2::-2].dense.tolist() == [[0], [0], [0], [0], [0]]
        assert a[-3::-2].dense.tolist() == [[103], [102], [101], [100]]

        assert [a[1:].dense[i].tolist() for i in range(9)] == [[100], [0], [101], [0], [102], [0], [103], [0], [104]]

    def test_indexed_ufunc(self):
        a = SparseArray(10, [1, 3, 5, 7, 9], [100, 101, 102, 103, 104])
        assert (a + 100).tolist() == [100, 200, 100, 201, 100, 202, 100, 203, 100, 204]

    def test_crossref(self):
        a = IndexedArray([0], UnionArray.fromtags([1, 0, 1, 0, 1, 0, 0, 1], [numpy.array([1.1, 2.2, 3.3, 4.4]), JaggedArray([1, 3, 5, 8], [3, 5, 8, 8], [])]))
        a.content.contents[1].content = a.content
        assert a.tolist() == [[1.1, [2.2, [3.3, 4.4, []]]]]
