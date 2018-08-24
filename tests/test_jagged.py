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

class Test(unittest.TestCase):
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

    def test_jagged_ufunc_object(self):
        class Z(object):
            def __init__(self, z):
                try:
                    self.z = list(z)
                except TypeError:
                    self.z = z
            def __eq__(self, other):
                return isinstance(other, Z) and self.z == other.z
            def __ne__(self, other):
                return not self.__eq__(other)
            def __repr__(self):
                return "Z({0})".format(self.z)

        a = JaggedArray([0, 3, 3, 5], [3, 3, 5, 10], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        self.assertEqual((awkward.ObjectArray([100, 200, 300, 400], Z) + a).tolist(), [Z([100., 101.1, 102.2]), Z([]), Z([303.3, 304.4]), Z([405.5, 406.6, 407.7, 408.8, 409.9])])

        a = JaggedArray([0, 3, 3, 5], [3, 3, 5, 10], awkward.ObjectArray([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], Z))
        self.assertEqual(a, [[Z(0.0), Z(1.1), Z(2.2)], [], [Z(3.3), Z(4.4)], [Z(5.5), Z(6.6), Z(7.7), Z(8.8), Z(9.9)]])
        self.assertEqual(a + awkward.ObjectArray([100, 200, 300, 400], Z), [[Z(100.0), Z(101.1), Z(102.2)], [], [Z(303.3), Z(304.4)], [Z(405.5), Z(406.6), Z(407.7), Z(408.8), Z(409.9)]])

    def test_jagged_ufunc_table(self):
        a = JaggedArray([0, 3, 3, 5], [3, 3, 5, 10], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        self.assertEqual((awkward.Table(x=[100, 200, 300, 400], y=[1000, 2000, 3000, 4000]) + a).tolist(), [{"x": [100.0, 101.1, 102.2], "y": [1000.0, 1001.1, 1002.2]}, {"x": [], "y": []}, {"x": [303.3, 304.4], "y": [3003.3, 3004.4]}, {"x": [405.5, 406.6, 407.7, 408.8, 409.9], "y": [4005.5, 4006.6, 4007.7, 4008.8, 4009.9]}])

        a = JaggedArray([0, 3, 3, 5], [3, 3, 5, 10], awkward.Table(x=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], y=[0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
        self.assertEqual((a + 1000).tolist(), [[{"x": 1000, "y": 1000.0}, {"x": 1001, "y": 1001.1}, {"x": 1002, "y": 1002.2}], [], [{"x": 1003, "y": 1003.3}, {"x": 1004, "y": 1004.4}], [{"x": 1005, "y": 1005.5}, {"x": 1006, "y": 1006.6}, {"x": 1007, "y": 1007.7}, {"x": 1008, "y": 1008.8}, {"x": 1009, "y": 1009.9}]])
        self.assertEqual((a + numpy.array([100, 200, 300, 400])).tolist(), [[{"x": 100, "y": 100.0}, {"x": 101, "y": 101.1}, {"x": 102, "y": 102.2}], [], [{"x": 303, "y": 303.3}, {"x": 304, "y": 304.4}], [{"x": 405, "y": 405.5}, {"x": 406, "y": 406.6}, {"x": 407, "y": 407.7}, {"x": 408, "y": 408.8}, {"x": 409, "y": 409.9}]])
        self.assertEqual((a + awkward.Table(x=[100, 200, 300, 400], y=[1000, 2000, 3000, 4000])).tolist(), [[{"x": 100, "y": 1000.0}, {"x": 101, "y": 1001.1}, {"x": 102, "y": 1002.2}], [], [{"x": 303, "y": 3003.3}, {"x": 304, "y": 3004.4}], [{"x": 405, "y": 4005.5}, {"x": 406, "y": 4006.6}, {"x": 407, "y": 4007.7}, {"x": 408, "y": 4008.8}, {"x": 409, "y": 4009.9}]])

    def test_jagged_cross(self):
        pass

    def test_jagged_sum(self):
        a = JaggedArray([0, 3, 3, 5], [3, 3, 5, 10], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        self.assertEqual(a.sum().tolist(), [3.3000000000000003, 0.0, 7.7, 38.5])

        a = JaggedArray([[0, 3], [3, 5]], [[3, 3], [5, 10]], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        self.assertEqual(a.sum().tolist(), [[3.3000000000000003, 0.0], [7.699999999999999, 38.5]])

        a = JaggedArray([0, 3, 3, 5], [3, 3, 5, 10], [[0.0], [1.1], [2.2], [3.3], [4.4], [5.5], [6.6], [7.7], [8.8], [9.9]])
        self.assertEqual(a.sum().tolist(), [[3.3000000000000003], [0.0], [7.7], [38.5]])

        a = JaggedArray([0, 3, 3, 5], [3, 3, 5, 10], [[0.0, 0.0], [1.1, 1.1], [2.2, 2.2], [3.3, 3.3], [4.4, 4.4], [5.5, 5.5], [6.6, 6.6], [7.7, 7.7], [8.8, 8.8], [9.9, 9.9]])
        self.assertEqual(a.sum().tolist(), [[3.3000000000000003, 3.3000000000000003], [0.0, 0.0], [7.7, 7.7], [38.5, 38.5]])

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

    def test_jagged_argmax(self):
        a = JaggedArray([0, 3, 3, 5], [3, 3, 5, 10], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        self.assertEqual(a.argmax().tolist(), [[2], [], [1], [4]])

        a = JaggedArray([[0, 3], [3, 5]], [[3, 3], [5, 10]], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        self.assertEqual(a.argmax().tolist(), [[[2], []], [[1], [4]]])

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

    def test_bytejagged_offsets(self):
        a = ByteJaggedArray.fromoffsets([5, 17, 17, 25], b"\xff\x00\x00\x00\x00\x01\x00\x00\x00\x02\x00\x00\x00\x03\x00\x00\x00\x04\x00\x00\x00\x05\x00\x00\x00\xff\xff", numpy.int32)
        self.assertEqual([x.tolist() for x in a], [[1, 2, 3], [], [4, 5]])

        a = ByteJaggedArray([5, 17, 19], [17, 17, 27], b"\xff\x00\x00\x00\x00\x01\x00\x00\x00\x02\x00\x00\x00\x03\x00\x00\x00\xff\xff\x04\x00\x00\x00\x05\x00\x00\x00\xff", numpy.int32)
        self.assertEqual([x.tolist() for x in a], [[1, 2, 3], [], [4, 5]])

    def test_bytejagged_iterable(self):
        a = ByteJaggedArray.fromiter([[1, 2, 3], [], [4, 5]])
        self.assertEqual([x.tolist() for x in a], [[1, 2, 3], [], [4, 5]])        
        if a.dtype.itemsize == 8:
            self.assertEqual(a.offsets.tolist(), [0, 24, 24, 40])
            self.assertEqual(a.content.tobytes(), b"\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x03\x00\x00\x00\x00\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x05\x00\x00\x00\x00\x00\x00\x00")
        elif a.dtype.itemsize == 4:
            self.assertEqual(a.offsets.tolist(), [0, 12, 12, 20])
            self.assertEqual(a.content.tobytes(), b"\x01\x00\x00\x00\x02\x00\x00\x00\x03\x00\x00\x00\x04\x00\x00\x00\x05\x00\x00\x00")
        else:
            raise AssertionError(a.dtype.itemsize)

    def test_bytejagged_get(self):
        a = ByteJaggedArray([5, 17, 19], [17, 17, 27], b"\xff\x00\x00\x00\x00\x01\x00\x00\x00\x02\x00\x00\x00\x03\x00\x00\x00\xff\xff\x04\x00\x00\x00\x05\x00\x00\x00\xff", numpy.int32)
        self.assertEqual([a[i].tolist() for i in range(len(a))], [[1, 2, 3], [], [4, 5]])
        self.assertEqual([x.tolist() for x in a], [[1, 2, 3], [], [4, 5]])
        self.assertEqual([x.tolist() for x in a[:]], [[1, 2, 3], [], [4, 5]])
        self.assertEqual([a[i : i + 1].tolist() for i in range(len(a))], [[[1, 2, 3]], [[]], [[4, 5]]])
        self.assertEqual([a[i : i + 2].tolist() for i in range(len(a) - 1)], [[[1, 2, 3], []], [[], [4, 5]]])
        self.assertEqual([x.tolist() for x in a[[2, 0, 1, 2]]], [[4, 5], [1, 2, 3], [], [4, 5]])
        self.assertEqual([x.tolist() for x in a[[2, 0]]], [[4, 5], [1, 2, 3]])
        self.assertEqual([x.tolist() for x in a[[True, True, False]]], [[1, 2, 3], []])

    def test_bytejagged_tojagged(self):
        a = awkward.ByteJaggedArray([0*4, 2*4, 3*4], [2*4, 3*4, 4*4], numpy.array([3, 0, 0, 0, 4, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 99, 99, 99, 99], "u1"), "u4")
        self.assertEqual(a.tolist(), [[3, 4], [2], [0]])
        self.assertEqual(a._tojagged(numpy.array([10, 30, 20]), numpy.array([12, 31, 21])).tolist(), [[3, 4], [2], [0]])
        self.assertEqual(a._tojagged(numpy.array([10, 30, 20]), numpy.array([12, 31, 21]))._tojagged().tolist(), [[3, 4], [2], [0]])
        self.assertEqual(a._tojagged(numpy.array([10, 30, 20]), numpy.array([12, 31, 21]))._tojagged(numpy.array([0, 2, 3]), numpy.array([2, 3, 4])).tolist(), [[3, 4], [2], [0]])

        a = awkward.ByteJaggedArray([0*4, 2*4, 4*4], [2*4, 3*4, 5*4], numpy.array([3, 0, 0, 0, 4, 0, 0, 0, 2, 0, 0, 0, 99, 99, 99, 99, 0, 0, 0, 0], "u1"), "u4")
        self.assertEqual(a.tolist(), [[3, 4], [2], [0]])
        self.assertEqual(a._tojagged(numpy.array([10, 30, 20]), numpy.array([12, 31, 21])).tolist(), [[3, 4], [2], [0]])
        self.assertEqual(a._tojagged(numpy.array([10, 30, 20]), numpy.array([12, 31, 21]))._tojagged().tolist(), [[3, 4], [2], [0]])
        self.assertEqual(a._tojagged(numpy.array([10, 30, 20]), numpy.array([12, 31, 21]))._tojagged(numpy.array([0, 2, 3]), numpy.array([2, 3, 4])).tolist(), [[3, 4], [2], [0]])

        a = awkward.ByteJaggedArray([3*4, 2*4, 0*4], [5*4, 3*4, 1*4], numpy.array([0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0], "u1"), "u4")
        self.assertEqual(a.tolist(), [[3, 4], [2], [0]])
        self.assertEqual(a._tojagged(numpy.array([10, 30, 20]), numpy.array([12, 31, 21])).tolist(), [[3, 4], [2], [0]])
        self.assertEqual(a._tojagged(numpy.array([10, 30, 20]), numpy.array([12, 31, 21]))._tojagged().tolist(), [[3, 4], [2], [0]])
        self.assertEqual(a._tojagged(numpy.array([10, 30, 20]), numpy.array([12, 31, 21]))._tojagged(numpy.array([0, 2, 3]), numpy.array([2, 3, 4])).tolist(), [[3, 4], [2], [0]])
