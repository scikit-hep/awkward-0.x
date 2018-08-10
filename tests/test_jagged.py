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
from awkward.type import *

class TestJagged(unittest.TestCase):
    def runTest(self):
        pass

    def test_jagged_init(self):
        a = JaggedArray([0, 3, 3, 5], [3, 3, 5, 10], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        self.assertEqual(a.tolist(), [[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5, 6.6, 7.7, 8.8, 9.9]])

        a = JaggedArray([[0, 3], [3, 5]], [[3, 3], [5, 10]], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        self.assertEqual(a.tolist(), [[[0.0, 1.1, 2.2], []], [[3.3, 4.4], [5.5, 6.6, 7.7, 8.8, 9.9]]])

        a = JaggedArray([0, 3, 3, 5], [3, 3, 5, 10], [[0.0], [1.1], [2.2], [3.3], [4.4], [5.5], [6.6], [7.7], [8.8], [9.9]])
        self.assertEqual(a.tolist(), [[[0.0], [1.1], [2.2]], [], [[3.3], [4.4]], [[5.5], [6.6], [7.7], [8.8], [9.9]]])

        self.assertEqual(JaggedArray.fromiter([[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5, 6.6, 7.7, 8.8, 9.9]]).tolist(), [[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5, 6.6, 7.7, 8.8, 9.9]])
        self.assertEqual(JaggedArray.fromoffsets([0, 3, 3, 5, 10], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]).tolist(), [[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5, 6.6, 7.7, 8.8, 9.9]])
        self.assertEqual(JaggedArray.fromcounts([3, 0, 2, 5], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]).tolist(), [[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5, 6.6, 7.7, 8.8, 9.9]])
        self.assertEqual(JaggedArray.fromparents([0, 0, 0, 2, 2, 3, 3, 3, 3, 3], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]).tolist(), [[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5, 6.6, 7.7, 8.8, 9.9]])
        self.assertEqual(JaggedArray.fromuniques([9, 9, 9, 8, 8, 7, 7, 7, 7, 7], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]).tolist(), [[0.0, 1.1, 2.2], [3.3, 4.4], [5.5, 6.6, 7.7, 8.8, 9.9]])
        self.assertEqual(JaggedArray.fromuniques([9, 9, 9, 8, 8, 7, 7, 7, 7, 7], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])._parents.tolist(), [0, 0, 0, 1, 1, 2, 2, 2, 2, 2])

        a = JaggedArray([], [], [0.0, 1.1, 2.2, 3.3, 4.4])
        self.assertEqual(a[:].tolist(), [])

    def test_jagged_type(self):
        a = JaggedArray([0, 3, 3, 5], [3, 3, 5, 10], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        self.assertEqual(a.type, ArrayType(4, numpy.inf, float))

        a = JaggedArray([[0, 3], [3, 5]], [[3, 3], [5, 10]], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        self.assertEqual(a.type, ArrayType(2, 2, numpy.inf, float))

        a = JaggedArray([0, 3, 3, 5], [3, 3, 5, 10], [[0.0], [1.1], [2.2], [3.3], [4.4], [5.5], [6.6], [7.7], [8.8], [9.9]])
        self.assertEqual(a.type, ArrayType(4, numpy.inf, 1, float))

    def test_jagged_str(self):
        pass

    def test_jagged_tuple(self):
        a = JaggedArray([0, 3, 3, 5], [3, 3, 5, 10], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        self.assertEqual(a[2][1], 4.4)
        self.assertEqual(a[2, 1], 4.4)
        self.assertEqual(a[2:, 1].tolist(), [4.4, 6.6])
        self.assertEqual(a[2:, -2].tolist(), [3.3, 8.8])

        a = JaggedArray([[0, 3], [3, 5]], [[3, 3], [5, 10]], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        self.assertEqual(a[1][1].tolist(), [5.5, 6.6, 7.7, 8.8, 9.9])
        self.assertEqual(a[1][1][1], 6.6)
        self.assertEqual(a[1, 1].tolist(), [5.5, 6.6, 7.7, 8.8, 9.9])
        self.assertEqual(a[1, 1, 1], 6.6)
        self.assertEqual(a[:, 1].tolist(), [[], [5.5, 6.6, 7.7, 8.8, 9.9]])
        self.assertEqual(a[:, 1][1].tolist(), [5.5, 6.6, 7.7, 8.8, 9.9])
        self.assertEqual(a[:, 0].tolist(), [[0.0, 1.1, 2.2], [3.3, 4.4]])
        self.assertEqual(a[:, 0, 1].tolist(), [1.1, 4.4])
        self.assertEqual(a[:, 0, 1, 1].tolist(), 4.4)

        a = JaggedArray([0, 3, 3, 5], [3, 3, 5, 10], [[0.0], [1.1], [2.2], [3.3], [4.4], [5.5], [6.6], [7.7], [8.8], [9.9]])
        self.assertEqual(a[2][1].tolist(), [4.4])
        self.assertEqual(a[2][1][0], 4.4)
        self.assertEqual(a[2, 1].tolist(), [4.4])
        self.assertEqual(a[2, 1, 0], 4.4)
        self.assertEqual(a[2:, 1].tolist(), [[4.4], [6.6]])
        self.assertEqual(a[2:, 1][1].tolist(), [6.6])
        self.assertEqual(a[2:, 1, 1].tolist(), [6.6])
        self.assertEqual(a[2:, 1, 1][0], 6.6)
        self.assertEqual(a[2:, 1, 1, 0], 6.6)
        self.assertEqual(a[2:, -2].tolist(), [[3.3], [8.8]])

    def test_jagged_slice(self):
        a = JaggedArray([0, 3, 3, 5], [3, 3, 5, 10], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        self.assertEqual(a[1:-1].tolist(), [[], [3.3, 4.4]])

        a = JaggedArray([[0, 3], [3, 5]], [[3, 3], [5, 10]], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        self.assertEqual(a[:].tolist(), [[[0.0, 1.1, 2.2], []], [[3.3, 4.4], [5.5, 6.6, 7.7, 8.8, 9.9]]])
        self.assertEqual(a[1:].tolist(), [[[3.3, 4.4], [5.5, 6.6, 7.7, 8.8, 9.9]]])

        a = JaggedArray([0, 3, 3, 5], [3, 3, 5, 10], [[0.0], [1.1], [2.2], [3.3], [4.4], [5.5], [6.6], [7.7], [8.8], [9.9]])
        self.assertEqual(a[1:-1].tolist(), [[], [[3.3], [4.4]]])

    def test_jagged_mask(self):
        a = JaggedArray([0, 3, 3, 5], [3, 3, 5, 10], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        self.assertEqual(a[[False, True, True, False]].tolist(), [[], [3.3, 4.4]])

        a = JaggedArray([[0, 3], [3, 5]], [[3, 3], [5, 10]], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        self.assertEqual(a[[True, True]].tolist(), [[[0.0, 1.1, 2.2], []], [[3.3, 4.4], [5.5, 6.6, 7.7, 8.8, 9.9]]])
        self.assertEqual(a[[False, True]].tolist(), [[[3.3, 4.4], [5.5, 6.6, 7.7, 8.8, 9.9]]])

        a = JaggedArray([0, 3, 3, 5], [3, 3, 5, 10], [[0.0], [1.1], [2.2], [3.3], [4.4], [5.5], [6.6], [7.7], [8.8], [9.9]])
        self.assertEqual(a[[False, True, True, False]].tolist(), [[], [[3.3], [4.4]]])

    def test_jagged_fancy(self):
        a = JaggedArray([0, 3, 3, 5], [3, 3, 5, 10], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        self.assertEqual(a[[1, 2]].tolist(), [[], [3.3, 4.4]])

        a = JaggedArray([[0, 3], [3, 5]], [[3, 3], [5, 10]], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        self.assertEqual(a[[0, 1]].tolist(), [[[0.0, 1.1, 2.2], []], [[3.3, 4.4], [5.5, 6.6, 7.7, 8.8, 9.9]]])
        self.assertEqual(a[[1]].tolist(), [[[3.3, 4.4], [5.5, 6.6, 7.7, 8.8, 9.9]]])

        a = JaggedArray([0, 3, 3, 5], [3, 3, 5, 10], [[0.0], [1.1], [2.2], [3.3], [4.4], [5.5], [6.6], [7.7], [8.8], [9.9]])
        self.assertEqual(a[[1, 2]].tolist(), [[], [[3.3], [4.4]]])

    def test_jagged_jagged(self):
        a = JaggedArray.fromoffsets([0, 3, 3, 5], JaggedArray.fromoffsets([0, 3, 3, 8, 10, 10], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
        self.assertEqual([a[i].tolist() for i in range(len(a))], [[[0.0, 1.1, 2.2], [], [3.3, 4.4, 5.5, 6.6, 7.7]], [], [[8.8, 9.9], []]])
        self.assertEqual([x.tolist() for x in a], [[[0.0, 1.1, 2.2], [], [3.3, 4.4, 5.5, 6.6, 7.7]], [], [[8.8, 9.9], []]])
        self.assertEqual([x.tolist() for x in a[:]], [[[0.0, 1.1, 2.2], [], [3.3, 4.4, 5.5, 6.6, 7.7]], [], [[8.8, 9.9], []]])
        self.assertEqual([x.tolist() for x in a[:-1]], [[[0.0, 1.1, 2.2], [], [3.3, 4.4, 5.5, 6.6, 7.7]], []])
        self.assertEqual([x.tolist() for x in a[[2, 1, 0]]], [[[8.8, 9.9], []], [], [[0.0, 1.1, 2.2], [], [3.3, 4.4, 5.5, 6.6, 7.7]]])
        self.assertEqual([x.tolist() for x in a[[True, True, False]]], [[[0.0, 1.1, 2.2], [], [3.3, 4.4, 5.5, 6.6, 7.7]], []])
        self.assertEqual(a[::2, 0].tolist(), [[0.0, 1.1, 2.2], [8.8, 9.9]])
        self.assertEqual(a[::2, 1].tolist(), [[], []])
        self.assertEqual(a[::2, 0, 1].tolist(), [1.1, 9.9])

    def test_jagged_ufunc(self):
        a = JaggedArray([0, 3, 3, 5], [3, 3, 5, 10], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        self.assertEqual((100 + a).tolist(), [[100.0, 101.1, 102.2], [], [103.3, 104.4], [105.5, 106.6, 107.7, 108.8, 109.9]])
        self.assertEqual((numpy.array([100, 200, 300, 400]) + a).tolist(), [[100.0, 101.1, 102.2], [], [303.3, 304.4], [405.5, 406.6, 407.7, 408.8, 409.9]])

    def test_jagged_cross(self):
        pass

    def test_jagged_sum(self):
        a = JaggedArray([0, 3, 3, 5], [3, 3, 5, 10], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        self.assertEqual(a.sum().tolist(), [3.3000000000000003, 0.0, 7.699999999999999, 38.5])

        a = JaggedArray([[0, 3], [3, 5]], [[3, 3], [5, 10]], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        self.assertEqual(a.sum().tolist(), [[3.3000000000000003, 0.0], [7.699999999999999, 38.5]])

        a = JaggedArray([0, 3, 3, 5], [3, 3, 5, 10], [[0.0], [1.1], [2.2], [3.3], [4.4], [5.5], [6.6], [7.7], [8.8], [9.9]])
        self.assertEqual(a.sum().tolist(), [[3.3000000000000003], [0.0], [7.699999999999999], [38.5]])

        a = JaggedArray([0, 3, 3, 5], [3, 3, 5, 10], [[0.0, 0.0], [1.1, 1.1], [2.2, 2.2], [3.3, 3.3], [4.4, 4.4], [5.5, 5.5], [6.6, 6.6], [7.7, 7.7], [8.8, 8.8], [9.9, 9.9]])
        self.assertEqual(a.sum().tolist(), [[3.3000000000000003, 3.3000000000000003], [0.0, 0.0], [7.699999999999999, 7.699999999999999], [38.5, 38.5]])

    def test_jagged_prod(self):
        a = JaggedArray([0, 3, 3, 5], [3, 3, 5, 10], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        self.assertEqual(a.prod().tolist(), [0.0, 1.0, 14.52, 24350.911200000002])

        a = JaggedArray([[0, 3], [3, 5]], [[3, 3], [5, 10]], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        self.assertEqual(a.prod().tolist(), [[0.0, 1.0], [14.52, 24350.911200000002]])

        a = JaggedArray([0, 3, 3, 5], [3, 3, 5, 10], [[0.0], [1.1], [2.2], [3.3], [4.4], [5.5], [6.6], [7.7], [8.8], [9.9]])
        self.assertEqual(a.prod().tolist(), [[0.0], [1.0], [14.52], [24350.911200000002]])

        a = JaggedArray([0, 3, 3, 5], [3, 3, 5, 10], [[0.0, 0.0], [1.1, 1.1], [2.2, 2.2], [3.3, 3.3], [4.4, 4.4], [5.5, 5.5], [6.6, 6.6], [7.7, 7.7], [8.8, 8.8], [9.9, 9.9]])
        self.assertEqual(a.prod().tolist(), [[0.0, 0.0], [1.0, 1.0], [14.52, 14.52], [24350.911200000002, 24350.911200000002]])

    def test_jagged_argmin(self):
        a = JaggedArray([0, 3, 3, 5], [3, 3, 5, 10], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        self.assertEqual(a.argmin().tolist(), [[0], [], [0], [0]])

        a = JaggedArray([[0, 3], [3, 5]], [[3, 3], [5, 10]], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        self.assertEqual(a.argmin().tolist(), [[[0], []], [[0], [0]]])

        a = JaggedArray([0, 3, 3, 5], [3, 3, 5, 10], [[0.0], [1.1], [2.2], [3.3], [4.4], [5.5], [6.6], [7.7], [8.8], [9.9]])
        self.assertEqual(a.argmin().tolist(), [[[0]], [], [[0]], [[0]]])

        a = JaggedArray([0, 3, 3, 5], [3, 3, 5, 10], [[0.0, 0.0], [1.1, 1.1], [2.2, 2.2], [3.3, 3.3], [4.4, 4.4], [5.5, 5.5], [6.6, 6.6], [7.7, 7.7], [8.8, 8.8], [9.9, 9.9]])
        self.assertEqual(a.argmin().tolist(), [[[0, 0]], [], [[0, 0]], [[0, 0]]])

    def test_jagged_argmax(self):
        a = JaggedArray([0, 3, 3, 5], [3, 3, 5, 10], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        self.assertEqual(a.argmax().tolist(), [[2], [], [1], [4]])

        a = JaggedArray([[0, 3], [3, 5]], [[3, 3], [5, 10]], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        self.assertEqual(a.argmax().tolist(), [[[2], []], [[1], [4]]])

        a = JaggedArray([0, 3, 3, 5], [3, 3, 5, 10], [[0.0], [1.1], [2.2], [3.3], [4.4], [5.5], [6.6], [7.7], [8.8], [9.9]])
        self.assertEqual(a.argmax().tolist(), [[[2]], [], [[1]], [[4]]])

        a = JaggedArray([0, 3, 3, 5], [3, 3, 5, 10], [[0.0, 0.0], [1.1, 1.1], [2.2, 2.2], [3.3, 3.3], [4.4, 4.4], [5.5, 5.5], [6.6, 6.6], [7.7, 7.7], [8.8, 8.8], [9.9, 9.9]])
        self.assertEqual(a.argmax().tolist(), [[[2, 2]], [], [[1, 1]], [[4, 4]]])

    def test_jagged_min(self):
        a = JaggedArray([0, 3, 3, 5], [3, 3, 5, 10], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        self.assertEqual(a.min().tolist(), [0.0, numpy.inf, 3.3, 5.5])

        a = JaggedArray([[0, 3], [3, 5]], [[3, 3], [5, 10]], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        self.assertEqual(a.min().tolist(), [[0.0, numpy.inf], [3.3, 5.5]])

        a = JaggedArray([0, 3, 3, 5], [3, 3, 5, 10], [[0.0], [1.1], [2.2], [3.3], [4.4], [5.5], [6.6], [7.7], [8.8], [9.9]])
        self.assertEqual(a.min().tolist(), [[0.0], [numpy.inf], [3.3], [5.5]])

        a = JaggedArray([0, 3, 3, 5], [3, 3, 5, 10], [[0.0, 0.0], [1.1, 1.1], [2.2, 2.2], [3.3, 3.3], [4.4, 4.4], [5.5, 5.5], [6.6, 6.6], [7.7, 7.7], [8.8, 8.8], [9.9, 9.9]])
        self.assertEqual(a.min().tolist(), [[0.0, 0.0], [numpy.inf, numpy.inf], [3.3, 3.3], [5.5, 5.5]])

    def test_jagged_max(self):
        a = JaggedArray([0, 3, 3, 5], [3, 3, 5, 10], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        self.assertEqual(a.max().tolist(), [2.2, -numpy.inf, 4.4, 9.9])

        a = JaggedArray([[0, 3], [3, 5]], [[3, 3], [5, 10]], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        self.assertEqual(a.max().tolist(), [[2.2, -numpy.inf], [4.4, 9.9]])

        a = JaggedArray([0, 3, 3, 5], [3, 3, 5, 10], [[0.0], [1.1], [2.2], [3.3], [4.4], [5.5], [6.6], [7.7], [8.8], [9.9]])
        self.assertEqual(a.max().tolist(), [[2.2], [-numpy.inf], [4.4], [9.9]])

        a = JaggedArray([0, 3, 3, 5], [3, 3, 5, 10], [[0.0, 0.0], [1.1, 1.1], [2.2, 2.2], [3.3, 3.3], [4.4, 4.4], [5.5, 5.5], [6.6, 6.6], [7.7, 7.7], [8.8, 8.8], [9.9, 9.9]])
        self.assertEqual(a.max().tolist(), [[2.2, 2.2], [-numpy.inf, -numpy.inf], [4.4, 4.4], [9.9, 9.9]])

    def test_jagged_get(self):
        a = JaggedArray.fromoffsets([0, 3, 3, 8, 10, 10], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        self.assertEqual([a[i].tolist() for i in range(len(a))], [[0.0, 1.1, 2.2], [], [3.3, 4.4, 5.5, 6.6, 7.7], [8.8, 9.9], []])
        self.assertEqual([x.tolist() for x in a], [[0.0, 1.1, 2.2], [], [3.3, 4.4, 5.5, 6.6, 7.7], [8.8, 9.9], []])
        self.assertEqual([x.tolist() for x in a[:]], [[0.0, 1.1, 2.2], [], [3.3, 4.4, 5.5, 6.6, 7.7], [8.8, 9.9], []])
        self.assertEqual([a[i : i + 1].tolist() for i in range(len(a))], [[[0.0, 1.1, 2.2]], [[]], [[3.3, 4.4, 5.5, 6.6, 7.7]], [[8.8, 9.9]], [[]]])
        self.assertEqual([a[i : i + 2].tolist() for i in range(len(a) - 1)], [[[0.0, 1.1, 2.2], []], [[], [3.3, 4.4, 5.5, 6.6, 7.7]], [[3.3, 4.4, 5.5, 6.6, 7.7], [8.8, 9.9]], [[8.8, 9.9], []]])
        self.assertEqual([x.tolist() for x in a[[2, 1, 0, -2]]], [[3.3, 4.4, 5.5, 6.6, 7.7], [], [0.0, 1.1, 2.2], [8.8, 9.9]])
        self.assertEqual([x.tolist() for x in a[[True, False, True, False, True]]], [[0.0, 1.1, 2.2], [3.3, 4.4, 5.5, 6.6, 7.7], []])

    def test_jagged_get_startsstops(self):
        a = JaggedArray([5, 2, 99, 1], [8, 7, 99, 3], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        self.assertEqual([x.tolist() for x in a], [[5.5, 6.6, 7.7], [2.2, 3.3, 4.4, 5.5, 6.6], [], [1.1, 2.2]])
        self.assertEqual([x.tolist() for x in a[:]], [[5.5, 6.6, 7.7], [2.2, 3.3, 4.4, 5.5, 6.6], [], [1.1, 2.2]])

    def test_jagged_get2d(self):
        a = JaggedArray.fromoffsets([0, 3, 3, 8, 10, 10], [[0.0, 0.0], [1.1, 1.1], [2.2, 2.2], [3.3, 3.3], [4.4, 4.4], [5.5, 5.5], [6.6, 6.6], [7.7, 7.7], [8.8, 8.8], [9.9, 9.9]])
        self.assertEqual([a[i].tolist() for i in range(len(a))], [[[0.0, 0.0], [1.1, 1.1], [2.2, 2.2]], [], [[3.3, 3.3], [4.4, 4.4], [5.5, 5.5], [6.6, 6.6], [7.7, 7.7]], [[8.8, 8.8], [9.9, 9.9]], []])
        self.assertEqual([x.tolist() for x in a], [[[0.0, 0.0], [1.1, 1.1], [2.2, 2.2]], [], [[3.3, 3.3], [4.4, 4.4], [5.5, 5.5], [6.6, 6.6], [7.7, 7.7]], [[8.8, 8.8], [9.9, 9.9]], []])
        self.assertEqual([x.tolist() for x in a[:]], [[[0.0, 0.0], [1.1, 1.1], [2.2, 2.2]], [], [[3.3, 3.3], [4.4, 4.4], [5.5, 5.5], [6.6, 6.6], [7.7, 7.7]], [[8.8, 8.8], [9.9, 9.9]], []])
        self.assertEqual([a[i : i + 1].tolist() for i in range(len(a))], [[[[0.0, 0.0], [1.1, 1.1], [2.2, 2.2]]], [[]], [[[3.3, 3.3], [4.4, 4.4], [5.5, 5.5], [6.6, 6.6], [7.7, 7.7]]], [[[8.8, 8.8], [9.9, 9.9]]], [[]]])
        self.assertEqual([a[i : i + 2].tolist() for i in range(len(a) - 1)], [[[[0.0, 0.0], [1.1, 1.1], [2.2, 2.2]], []], [[], [[3.3, 3.3], [4.4, 4.4], [5.5, 5.5], [6.6, 6.6], [7.7, 7.7]]], [[[3.3, 3.3], [4.4, 4.4], [5.5, 5.5], [6.6, 6.6], [7.7, 7.7]], [[8.8, 8.8], [9.9, 9.9]]], [[[8.8, 8.8], [9.9, 9.9]], []]])
        self.assertEqual([x.tolist() for x in a[[2, 1, 0, -2]]], [[[3.3, 3.3], [4.4, 4.4], [5.5, 5.5], [6.6, 6.6], [7.7, 7.7]], [], [[0.0, 0.0], [1.1, 1.1], [2.2, 2.2]], [[8.8, 8.8], [9.9, 9.9]]])
        self.assertEqual([x.tolist() for x in a[[True, False, True, False, True]]], [[[0.0, 0.0], [1.1, 1.1], [2.2, 2.2]], [[3.3, 3.3], [4.4, 4.4], [5.5, 5.5], [6.6, 6.6], [7.7, 7.7]], []])

    def test_jagged_getstruct(self):
        a = JaggedArray.fromoffsets([0, 3, 3, 8, 10, 10], numpy.array([(0.0, 0.0), (1.1, 1.1), (2.2, 2.2), (3.3, 3.3), (4.4, 4.4), (5.5, 5.5), (6.6, 6.6), (7.7, 7.7), (8.8, 8.8), (9.9, 9.9)], dtype=[("a", float), ("b", float)]))
        self.assertEqual([a[i].tolist() for i in range(len(a))], [[(0.0, 0.0), (1.1, 1.1), (2.2, 2.2)], [], [(3.3, 3.3), (4.4, 4.4), (5.5, 5.5), (6.6, 6.6), (7.7, 7.7)], [(8.8, 8.8), (9.9, 9.9)], []])
        self.assertEqual([x.tolist() for x in a], [[(0.0, 0.0), (1.1, 1.1), (2.2, 2.2)], [], [(3.3, 3.3), (4.4, 4.4), (5.5, 5.5), (6.6, 6.6), (7.7, 7.7)], [(8.8, 8.8), (9.9, 9.9)], []])
        self.assertEqual([x.tolist() for x in a[:]], [[(0.0, 0.0), (1.1, 1.1), (2.2, 2.2)], [], [(3.3, 3.3), (4.4, 4.4), (5.5, 5.5), (6.6, 6.6), (7.7, 7.7)], [(8.8, 8.8), (9.9, 9.9)], []])
        self.assertEqual([a[i : i + 1].tolist() for i in range(len(a))], [[[(0.0, 0.0), (1.1, 1.1), (2.2, 2.2)]], [[]], [[(3.3, 3.3), (4.4, 4.4), (5.5, 5.5), (6.6, 6.6), (7.7, 7.7)]], [[(8.8, 8.8), (9.9, 9.9)]], [[]]])
        self.assertEqual([a[i : i + 2].tolist() for i in range(len(a) - 1)], [[[(0.0, 0.0), (1.1, 1.1), (2.2, 2.2)], []], [[], [(3.3, 3.3), (4.4, 4.4), (5.5, 5.5), (6.6, 6.6), (7.7, 7.7)]], [[(3.3, 3.3), (4.4, 4.4), (5.5, 5.5), (6.6, 6.6), (7.7, 7.7)], [(8.8, 8.8), (9.9, 9.9)]], [[(8.8, 8.8), (9.9, 9.9)], []]])
        self.assertEqual([x.tolist() for x in a[[2, 1, 0, -2]]], [[(3.3, 3.3), (4.4, 4.4), (5.5, 5.5), (6.6, 6.6), (7.7, 7.7)], [], [(0.0, 0.0), (1.1, 1.1), (2.2, 2.2)], [(8.8, 8.8), (9.9, 9.9)]])
        self.assertEqual([x.tolist() for x in a[[True, False, True, False, True]]], [[(0.0, 0.0), (1.1, 1.1), (2.2, 2.2)], [(3.3, 3.3), (4.4, 4.4), (5.5, 5.5), (6.6, 6.6), (7.7, 7.7)], []])

    ################### old tests

    # def test_bytejagged_offsets(self):
    #     a = ByteJaggedArray.fromoffsets([5, 17, 17, 25], b"\xff\x00\x00\x00\x00\x01\x00\x00\x00\x02\x00\x00\x00\x03\x00\x00\x00\x04\x00\x00\x00\x05\x00\x00\x00\xff\xff", numpy.int32)
    #     self.assertEqual([x.tolist() for x in a], [[1, 2, 3], [], [4, 5]])

    #     a = ByteJaggedArray([5, 17, 19], [17, 17, 27], b"\xff\x00\x00\x00\x00\x01\x00\x00\x00\x02\x00\x00\x00\x03\x00\x00\x00\xff\xff\x04\x00\x00\x00\x05\x00\x00\x00\xff", numpy.int32)
    #     self.assertEqual([x.tolist() for x in a], [[1, 2, 3], [], [4, 5]])

    # def test_bytejagged_iterable(self):
    #     a = ByteJaggedArray.fromiter([[1, 2, 3], [], [4, 5]])
    #     self.assertEqual([x.tolist() for x in a], [[1, 2, 3], [], [4, 5]])        
    #     if a.dtype.itemsize == 8:
    #         self.assertEqual(a.offsets.tolist(), [0, 24, 24, 40])
    #         self.assertEqual(a.content.tobytes(), b"\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x03\x00\x00\x00\x00\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x05\x00\x00\x00\x00\x00\x00\x00")
    #     elif a.dtype.itemsize == 4:
    #         self.assertEqual(a.offsets.tolist(), [0, 12, 12, 20])
    #         self.assertEqual(a.content.tobytes(), b"\x01\x00\x00\x00\x02\x00\x00\x00\x03\x00\x00\x00\x04\x00\x00\x00\x05\x00\x00\x00")
    #     else:
    #         raise AssertionError(a.dtype.itemsize)

    # def test_bytejagged_get(self):
    #     a = ByteJaggedArray([5, 17, 19], [17, 17, 27], b"\xff\x00\x00\x00\x00\x01\x00\x00\x00\x02\x00\x00\x00\x03\x00\x00\x00\xff\xff\x04\x00\x00\x00\x05\x00\x00\x00\xff", numpy.int32)
    #     self.assertEqual([a[i].tolist() for i in range(len(a))], [[1, 2, 3], [], [4, 5]])
    #     self.assertEqual([x.tolist() for x in a], [[1, 2, 3], [], [4, 5]])
    #     self.assertEqual([x.tolist() for x in a[:]], [[1, 2, 3], [], [4, 5]])
    #     self.assertEqual([a[i : i + 1].tolist() for i in range(len(a))], [[[1, 2, 3]], [[]], [[4, 5]]])
    #     self.assertEqual([a[i : i + 2].tolist() for i in range(len(a) - 1)], [[[1, 2, 3], []], [[], [4, 5]]])
    #     self.assertEqual([x.tolist() for x in a[[2, 0, 1, 2]]], [[4, 5], [1, 2, 3], [], [4, 5]])
    #     self.assertEqual([x.tolist() for x in a[[2, 0]]], [[4, 5], [1, 2, 3]])
    #     self.assertEqual([x.tolist() for x in a[[True, True, False]]], [[1, 2, 3], []])

    # def test_jagged_argproduct(self):
    #     starts1 = [0,1,4,4]
    #     stops1 = [1,4,4,8]

    #     starts2 = [0,1,1,4]
    #     stops2 = [1,1,4,5]

    #     arr1 = JaggedArray(starts1, stops1,content=[0,1,2,3,4,5,6,7])
    #     arr2 = JaggedArray(starts2, stops2,content=['z', 'a','b','c','d'])
