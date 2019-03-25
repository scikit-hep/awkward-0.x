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

import unittest

import numpy

import awkward
from awkward import *
from awkward.type import *

class Test(unittest.TestCase):
    def runTest(self):
        pass

    def test_jagged_nbytes(self):
        assert isinstance(JaggedArray([0, 3, 3, 5], [3, 3, 5, 10], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]).nbytes, int)

    def test_jagged_init(self):
        a = JaggedArray([0, 3, 3, 5], [3, 3, 5, 10], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        assert a.tolist() == [[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5, 6.6, 7.7, 8.8, 9.9]]

        a = JaggedArray([[0, 3], [3, 5]], [[3, 3], [5, 10]], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        assert a.tolist() == [[[0.0, 1.1, 2.2], []], [[3.3, 4.4], [5.5, 6.6, 7.7, 8.8, 9.9]]]

        a = JaggedArray([0, 3, 3, 5], [3, 3, 5, 10], [[0.0], [1.1], [2.2], [3.3], [4.4], [5.5], [6.6], [7.7], [8.8], [9.9]])
        assert a.tolist() == [[[0.0], [1.1], [2.2]], [], [[3.3], [4.4]], [[5.5], [6.6], [7.7], [8.8], [9.9]]]

        assert JaggedArray.fromiter([[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5, 6.6, 7.7, 8.8, 9.9]]).tolist() == [[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5, 6.6, 7.7, 8.8, 9.9]]
        assert JaggedArray.fromoffsets([0, 3, 3, 5, 10], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]).tolist() == [[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5, 6.6, 7.7, 8.8, 9.9]]
        assert JaggedArray.fromcounts([3, 0, 2, 5], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]).tolist() == [[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5, 6.6, 7.7, 8.8, 9.9]]
        assert JaggedArray.fromparents([0, 0, 0, 2, 2, 3, 3, 3, 3, 3], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]).tolist() == [[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5, 6.6, 7.7, 8.8, 9.9]]
        assert JaggedArray.fromuniques([9, 9, 9, 8, 8, 7, 7, 7, 7, 7], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]).tolist() == [[0.0, 1.1, 2.2], [3.3, 4.4], [5.5, 6.6, 7.7, 8.8, 9.9]]
        assert JaggedArray.fromuniques([9, 9, 9, 8, 8, 7, 7, 7, 7, 7], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])._parents.tolist() == [0, 0, 0, 1, 1, 2, 2, 2, 2, 2]

        a = JaggedArray([], [], [0.0, 1.1, 2.2, 3.3, 4.4])
        assert a[:].tolist() == []

    def test_jagged_type(self):
        a = JaggedArray([0, 3, 3, 5], [3, 3, 5, 10], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        assert a.type == ArrayType(4, numpy.inf, float)

        a = JaggedArray([[0, 3], [3, 5]], [[3, 3], [5, 10]], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        assert a.type == ArrayType(2, 2, numpy.inf, float)

        a = JaggedArray([0, 3, 3, 5], [3, 3, 5, 10], [[0.0], [1.1], [2.2], [3.3], [4.4], [5.5], [6.6], [7.7], [8.8], [9.9]])
        assert a.type == ArrayType(4, numpy.inf, 1, float)

    def test_jagged_fromindex(self):
        a = JaggedArray.fromindex([0, 1, 0, 0, 0, 1, 2, 0, 1, 0], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        self.assertEqual(a.tolist(), [[0.0, 1.1], [2.2], [3.3], [4.4, 5.5, 6.6], [7.7, 8.8], [9.9]])
        self.assertEqual(a.starts.tolist(), [0, 2, 3, 4, 7, 9])
        self.assertEqual(a.stops.tolist(), [2, 3, 4, 7, 9, 10])

    def test_jagged_tojagged(self):
        a = JaggedArray.fromiter([[1], [2, 3], []])
        assert a.tojagged(a).tolist() == a.tolist()
        b = numpy.array([3,4,5])
        assert a.tojagged(b+1).tolist() == JaggedArray.fromiter([[4], [5, 5], []]).tolist()
        a = JaggedArray.fromiter([[]])
        assert a.tojagged(a).tolist() == a.tolist()
        a = JaggedArray.fromiter([[], []])
        assert a.tojagged(a).tolist() == a.tolist()

    def test_jagged_str(self):
        pass

    def test_jagged_tuple(self):
        a = JaggedArray([0, 3, 3, 5], [3, 3, 5, 10], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        assert a[2][1] == 4.4
        assert a[2, 1] == 4.4
        assert a[2:, 1].tolist() == [4.4, 6.6]
        assert a[2:, -2].tolist() == [3.3, 8.8]

        a = JaggedArray([[0, 3], [3, 5]], [[3, 3], [5, 10]], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        assert a[1][1].tolist() == [5.5, 6.6, 7.7, 8.8, 9.9]
        assert a[1][1][1] == 6.6
        assert a[1, 1].tolist() == [5.5, 6.6, 7.7, 8.8, 9.9]
        assert a[1, 1, 1] == 6.6
        assert a[:, 1].tolist() == [[], [5.5, 6.6, 7.7, 8.8, 9.9]]
        assert a[:, 1][1].tolist() == [5.5, 6.6, 7.7, 8.8, 9.9]
        assert a[:, 0].tolist() == [[0.0, 1.1, 2.2], [3.3, 4.4]]
        assert a[:, 0, 1].tolist() == [1.1, 4.4]
        assert a[:, 0, 1, 1].tolist() == 4.4

        a = JaggedArray([0, 3, 3, 5], [3, 3, 5, 10], [[0.0], [1.1], [2.2], [3.3], [4.4], [5.5], [6.6], [7.7], [8.8], [9.9]])
        assert a[2][1].tolist() == [4.4]
        assert a[2][1][0] == 4.4
        assert a[2, 1].tolist() == [4.4]
        assert a[2, 1, 0] == 4.4
        assert a[2:, 1].tolist() == [[4.4], [6.6]]
        assert a[2:, 1][1].tolist() == [6.6]
        assert a[2:, 1, 1].tolist() == [6.6]
        assert a[2:, 1, 1][0] == 6.6
        assert a[2:, 1, 1, 0] == 6.6
        assert a[2:, -2].tolist() == [[3.3], [8.8]]

    def test_jagged_slice(self):
        a = JaggedArray([0, 3, 3, 5], [3, 3, 5, 10], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        assert a[1:-1].tolist() == [[], [3.3, 4.4]]

        a = JaggedArray([[0, 3], [3, 5]], [[3, 3], [5, 10]], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        assert a[:].tolist() == [[[0.0, 1.1, 2.2], []], [[3.3, 4.4], [5.5, 6.6, 7.7, 8.8, 9.9]]]
        assert a[1:].tolist() == [[[3.3, 4.4], [5.5, 6.6, 7.7, 8.8, 9.9]]]

        a = JaggedArray([0, 3, 3, 5], [3, 3, 5, 10], [[0.0], [1.1], [2.2], [3.3], [4.4], [5.5], [6.6], [7.7], [8.8], [9.9]])
        assert a[1:-1].tolist() == [[], [[3.3], [4.4]]]

    def test_jagged_mask(self):
        a = JaggedArray([0, 3, 3, 5], [3, 3, 5, 10], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        assert a[[False, True, True, False]].tolist() == [[], [3.3, 4.4]]

        a = JaggedArray([[0, 3], [3, 5]], [[3, 3], [5, 10]], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        assert a[[True, True]].tolist() == [[[0.0, 1.1, 2.2], []], [[3.3, 4.4], [5.5, 6.6, 7.7, 8.8, 9.9]]]
        assert a[[False, True]].tolist() == [[[3.3, 4.4], [5.5, 6.6, 7.7, 8.8, 9.9]]]

        a = JaggedArray([0, 3, 3, 5], [3, 3, 5, 10], [[0.0], [1.1], [2.2], [3.3], [4.4], [5.5], [6.6], [7.7], [8.8], [9.9]])
        assert a[[False, True, True, False]].tolist() == [[], [[3.3], [4.4]]]

    def test_jagged_fancy(self):
        a = JaggedArray([0, 3, 3, 5], [3, 3, 5, 10], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        assert a[[1, 2]].tolist() == [[], [3.3, 4.4]]

        a = JaggedArray([[0, 3], [3, 5]], [[3, 3], [5, 10]], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        assert a[[0, 1]].tolist() == [[[0.0, 1.1, 2.2], []], [[3.3, 4.4], [5.5, 6.6, 7.7, 8.8, 9.9]]]
        assert a[[1]].tolist() == [[[3.3, 4.4], [5.5, 6.6, 7.7, 8.8, 9.9]]]

        a = JaggedArray([0, 3, 3, 5], [3, 3, 5, 10], [[0.0], [1.1], [2.2], [3.3], [4.4], [5.5], [6.6], [7.7], [8.8], [9.9]])
        assert a[[1, 2]].tolist() == [[], [[3.3], [4.4]]]

    def test_jagged_subslice(self):
        a = JaggedArray.fromiter([[], [100, 101, 102], [200, 201, 202, 203], [300, 301, 302, 303, 304], [], [500, 501], [600], []])
        for start in None, 0, 1, 2, 3, 4, 5, -1, -2, -3, -4, -5, -6:
            for stop in None, 0, 1, 2, 3, 4, 5, -1, -2, -3, -4, -5, -6:
                for step in None, 1, 2, 3, 4, 5, -1, -2, -3, -4, -5:
                    assert a[:, start:stop:step].tolist() == [x.tolist()[start:stop:step] for x in a]

    def test_jagged_jagged(self):
        a = JaggedArray.fromoffsets([0, 3, 3, 5], JaggedArray.fromoffsets([0, 3, 3, 8, 10, 10], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
        assert [a[i].tolist() for i in range(len(a))] == [[[0.0, 1.1, 2.2], [], [3.3, 4.4, 5.5, 6.6, 7.7]], [], [[8.8, 9.9], []]]
        assert [x.tolist() for x in a] == [[[0.0, 1.1, 2.2], [], [3.3, 4.4, 5.5, 6.6, 7.7]], [], [[8.8, 9.9], []]]
        assert [x.tolist() for x in a[:]] == [[[0.0, 1.1, 2.2], [], [3.3, 4.4, 5.5, 6.6, 7.7]], [], [[8.8, 9.9], []]]
        assert [x.tolist() for x in a[:-1]] == [[[0.0, 1.1, 2.2], [], [3.3, 4.4, 5.5, 6.6, 7.7]], []]
        assert [x.tolist() for x in a[[2, 1, 0]]] == [[[8.8, 9.9], []], [], [[0.0, 1.1, 2.2], [], [3.3, 4.4, 5.5, 6.6, 7.7]]]
        assert [x.tolist() for x in a[[True, True, False]]] == [[[0.0, 1.1, 2.2], [], [3.3, 4.4, 5.5, 6.6, 7.7]], []]
        assert a[::2, 0].tolist() == [[0.0, 1.1, 2.2], [8.8, 9.9]]
        assert a[::2, 1].tolist() == [[], []]
        assert a[::2, 0, 1].tolist() == [1.1, 9.9]

    def test_jagged_ufunc(self):
        a = JaggedArray([0, 3, 3, 5], [3, 3, 5, 10], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        assert (100 + a).tolist() == [[100.0, 101.1, 102.2], [], [103.3, 104.4], [105.5, 106.6, 107.7, 108.8, 109.9]]
        assert (numpy.array([100, 200, 300, 400]) + a).tolist() == [[100.0, 101.1, 102.2], [], [303.3, 304.4], [405.5, 406.6, 407.7, 408.8, 409.9]]

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
        assert (awkward.ObjectArray([100, 200, 300, 400], Z) + a).tolist() == [Z([100., 101.1, 102.2]), Z([]), Z([303.3, 304.4]), Z([405.5, 406.6, 407.7, 408.8, 409.9])]

        self.assertRaises(ValueError, lambda: "yay" if (a + [100, 200, 300, 400]) == [[100.0, 101.1, 102.2], [], [303.3, 304.4], [405.5, 406.6, 407.7, 408.8, 409.9]] else "boo")
        self.assertRaises(ValueError, lambda: "yay" if (a + [100, 200, 300, 400]).content == [100.0, 101.1, 102.2, 303.3, 304.4, 405.5, 406.6, 407.7, 408.8, 409.9] else "boo")

        a = JaggedArray([0, 3, 3, 5], [3, 3, 5, 10], awkward.ObjectArray([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], Z))
        assert a.tolist() == [[Z(0.0), Z(1.1), Z(2.2)], [], [Z(3.3), Z(4.4)], [Z(5.5), Z(6.6), Z(7.7), Z(8.8), Z(9.9)]]
        assert (a + awkward.ObjectArray([100, 200, 300, 400], Z)).tolist() == [[Z(100.0), Z(101.1), Z(102.2)], [], [Z(303.3), Z(304.4)], [Z(405.5), Z(406.6), Z(407.7), Z(408.8), Z(409.9)]]

    def test_jagged_ufunc_table(self):
        a = JaggedArray([0, 3, 3, 5], [3, 3, 5, 10], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        assert (awkward.Table(x=[100, 200, 300, 400], y=[1000, 2000, 3000, 4000]) + a).tolist() == [{"x": [100.0, 101.1, 102.2], "y": [1000.0, 1001.1, 1002.2]}, {"x": [], "y": []}, {"x": [303.3, 304.4], "y": [3003.3, 3004.4]}, {"x": [405.5, 406.6, 407.7, 408.8, 409.9], "y": [4005.5, 4006.6, 4007.7, 4008.8, 4009.9]}]

        a = JaggedArray([0, 3, 3, 5], [3, 3, 5, 10], awkward.Table(x=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], y=[0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
        assert (a + 1000).tolist() == [[{"x": 1000, "y": 1000.0}, {"x": 1001, "y": 1001.1}, {"x": 1002, "y": 1002.2}], [], [{"x": 1003, "y": 1003.3}, {"x": 1004, "y": 1004.4}], [{"x": 1005, "y": 1005.5}, {"x": 1006, "y": 1006.6}, {"x": 1007, "y": 1007.7}, {"x": 1008, "y": 1008.8}, {"x": 1009, "y": 1009.9}]]
        assert (a + numpy.array([100, 200, 300, 400])).tolist() == [[{"x": 100, "y": 100.0}, {"x": 101, "y": 101.1}, {"x": 102, "y": 102.2}], [], [{"x": 303, "y": 303.3}, {"x": 304, "y": 304.4}], [{"x": 405, "y": 405.5}, {"x": 406, "y": 406.6}, {"x": 407, "y": 407.7}, {"x": 408, "y": 408.8}, {"x": 409, "y": 409.9}]]
        assert (a + awkward.Table(x=[100, 200, 300, 400], y=[1000, 2000, 3000, 4000])).tolist() == [[{"x": 100, "y": 1000.0}, {"x": 101, "y": 1001.1}, {"x": 102, "y": 1002.2}], [], [{"x": 303, "y": 3003.3}, {"x": 304, "y": 3004.4}], [{"x": 405, "y": 4005.5}, {"x": 406, "y": 4006.6}, {"x": 407, "y": 4007.7}, {"x": 408, "y": 4008.8}, {"x": 409, "y": 4009.9}]]

    def test_jagged_regular(self):
        a = JaggedArray([0, 3, 6, 9], [3, 6, 9, 12], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.0, 11.0])
        assert a.regular().tolist() == [[0.0, 1.1, 2.2], [3.3, 4.4, 5.5], [6.6, 7.7, 8.8], [9.9, 10.0, 11.0]]

        a = JaggedArray([0, 3, 6, 9], [3, 6, 9, 12], [[0.0], [1.1], [2.2], [3.3], [4.4], [5.5], [6.6], [7.7], [8.8], [9.9], [10.0], [11.0]])
        assert a.regular().tolist() == [[[0.0], [1.1], [2.2]], [[3.3], [4.4], [5.5]], [[6.6], [7.7], [8.8]], [[9.9], [10.0], [11.0]]]

        a = JaggedArray([[0, 3], [6, 9]], [[3, 6], [9, 12]], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.0, 11.0])
        assert a.regular().tolist() == [[[0.0, 1.1, 2.2], [3.3, 4.4, 5.5]], [[6.6, 7.7, 8.8], [9.9, 10.0, 11.0]]]

        a = JaggedArray([[0, 3], [6, 9]], [[3, 6], [9, 12]], [[0.0], [1.1], [2.2], [3.3], [4.4], [5.5], [6.6], [7.7], [8.8], [9.9], [10.0], [11.0]])
        assert a.regular().tolist() == [[[[0.0], [1.1], [2.2]], [[3.3], [4.4], [5.5]]], [[[6.6], [7.7], [8.8]], [[9.9], [10.0], [11.0]]]]

        a = JaggedArray([0, 3, 7, 10], [3, 6, 10, 13], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 999, 6.6, 7.7, 8.8, 9.9, 10.0, 11.0])
        assert a.regular().tolist() == [[0.0, 1.1, 2.2], [3.3, 4.4, 5.5], [6.6, 7.7, 8.8], [9.9, 10.0, 11.0]]

        a = JaggedArray([0, 3, 7, 10], [3, 6, 10, 13], [[0.0], [1.1], [2.2], [3.3], [4.4], [5.5], [999], [6.6], [7.7], [8.8], [9.9], [10.0], [11.0]])
        assert a.regular().tolist() == [[[0.0], [1.1], [2.2]], [[3.3], [4.4], [5.5]], [[6.6], [7.7], [8.8]], [[9.9], [10.0], [11.0]]]

        a = JaggedArray([[0, 3], [7, 10]], [[3, 6], [10, 13]], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 999, 6.6, 7.7, 8.8, 9.9, 10.0, 11.0])
        assert a.regular().tolist() == [[[0.0, 1.1, 2.2], [3.3, 4.4, 5.5]], [[6.6, 7.7, 8.8], [9.9, 10.0, 11.0]]]

        a = JaggedArray([[0, 3], [7, 10]], [[3, 6], [10, 13]], [[0.0], [1.1], [2.2], [3.3], [4.4], [5.5], [999], [6.6], [7.7], [8.8], [9.9], [10.0], [11.0]])
        assert a.regular().tolist() == [[[[0.0], [1.1], [2.2]], [[3.3], [4.4], [5.5]]], [[[6.6], [7.7], [8.8]], [[9.9], [10.0], [11.0]]]]

    def test_jagged_cross(self):
        for i in range(10):
            for j in range(5):
                a = JaggedArray.fromiter([[], [123], list(range(i)), []])
                b = JaggedArray.fromiter([[], [456], list(range(j)), [999]])
                c = a.cross(b)
                assert len(c) == 4
                assert len(c[0]) == 0
                assert len(c[1]) == 1
                assert len(c[2]) == i * j
                assert len(c[3]) == 0
                assert c[2]["0"].tolist() == numpy.repeat(range(i), j).tolist()
                assert c[2]["1"].tolist() == numpy.tile(range(j), i).tolist()

    def test_jagged_pairs(self):
        for i in range(50):
            a = JaggedArray.fromiter([[], [123], list(range(i)), []])
            c = a.pairs()
            assert len(c) == 4
            assert len(c[0]) == 0
            assert len(c[1]) == 1
            assert len(c[2]) == i * (i + 1) // 2
            assert len(c[3]) == 0
            assert c[2]["0"].tolist() == sum([[x] * (i - x) for x in range(i)], [])
            assert c[2]["1"].tolist() == sum([list(range(x, i)) for x in range(i)], [])
        
    def test_jagged_distincts(self):
        for i in range(50):
            a = JaggedArray.fromiter([[], [123], list(range(i)), []])
            c = a.distincts()
            assert len(c) == 4
            assert len(c[0]) == 0
            assert len(c[1]) == 0
            assert len(c[2]) == i * (i - 1) // 2
            assert len(c[3]) == 0
            left = sum([[x] * (i - x) for x in range(i)], [])
            right = sum([list(range(x, i)) for x in range(i)], [])
            assert c[2]["0"].tolist() == [x for x, y in zip(left, right) if x != y]
            assert c[2]["1"].tolist() == [y for x, y in zip(left, right) if x != y]

    def test_jagged_cross_argnested(self):
        a = awkward.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
        b = awkward.fromiter([[100, 200], [300], [400]])
        c = awkward.fromiter([[999], [999], [999, 888]])

        assert a.cross(b).tolist() == [[(1.1, 100), (1.1, 200), (2.2, 100), (2.2, 200), (3.3, 100), (3.3, 200)], [], [(4.4, 400), (5.5, 400)]]
        assert a.argcross(b).tolist() == [[(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)], [], [(0, 0), (1, 0)]]
        assert a.cross(b, nested=True).tolist() == [[[(1.1, 100), (1.1, 200)], [(2.2, 100), (2.2, 200)], [(3.3, 100), (3.3, 200)]], [], [[(4.4, 400)], [(5.5, 400)]]]
        assert a.argcross(b, nested=True).tolist() == [[[(0, 0), (0, 1)], [(1, 0), (1, 1)], [(2, 0), (2, 1)]], [], [[(0, 0)], [(1, 0)]]]

        assert a.cross(b, nested=True).cross(c, nested=True).tolist()[0] == [[[(ai, bi, ci) for ci in c[0]] for bi in b[0]] for ai in a[0]]
        assert a.cross(b, nested=True).cross(c, nested=True).tolist()[1] == [[[(ai, bi, ci) for ci in c[1]] for bi in b[1]] for ai in a[1]]
        assert a.cross(b, nested=True).cross(c, nested=True).tolist()[2] == [[[(ai, bi, ci) for ci in c[2]] for bi in b[2]] for ai in a[2]]

        assert a.cross(b).cross(c).tolist() == [[(1.1, 100, 999), (1.1, 200, 999), (2.2, 100, 999), (2.2, 200, 999), (3.3, 100, 999), (3.3, 200, 999)], [], [(4.4, 400, 999), (4.4, 400, 888), (5.5, 400, 999), (5.5, 400, 888)]]
        assert a.cross(b, nested=True).cross(c).tolist() == [[[(1.1, 100, 999), (1.1, 200, 999)], [(2.2, 100, 999), (2.2, 200, 999)], [(3.3, 100, 999), (3.3, 200, 999)]], [], [[(4.4, 400, 999), (4.4, 400, 888)], [(5.5, 400, 999), (5.5, 400, 888)]]]
        assert a.cross(b).cross(c, nested=True).tolist() == [[[(1.1, 100, 999)], [(1.1, 200, 999)], [(2.2, 100, 999)], [(2.2, 200, 999)], [(3.3, 100, 999)], [(3.3, 200, 999)]], [], [[(4.4, 400, 999), (4.4, 400, 888)], [(5.5, 400, 999), (5.5, 400, 888)]]]

        a = awkward.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
        b = awkward.fromiter([[100, 200], [300], [400]])
        c = awkward.fromiter([[999], [999], [999, 888, 777]])

        assert a.cross(b, nested=True).cross(c, nested=True).tolist()[0] == [[[(ai, bi, ci) for ci in c[0]] for bi in b[0]] for ai in a[0]]
        assert a.cross(b, nested=True).cross(c, nested=True).tolist()[1] == [[[(ai, bi, ci) for ci in c[1]] for bi in b[1]] for ai in a[1]]
        assert a.cross(b, nested=True).cross(c, nested=True).tolist()[2] == [[[(ai, bi, ci) for ci in c[2]] for bi in b[2]] for ai in a[2]]

        assert a.cross(b).cross(c).tolist() == [[(1.1, 100, 999), (1.1, 200, 999), (2.2, 100, 999), (2.2, 200, 999), (3.3, 100, 999), (3.3, 200, 999)], [], [(4.4, 400, 999), (4.4, 400, 888), (4.4, 400, 777), (5.5, 400, 999), (5.5, 400, 888), (5.5, 400, 777)]]
        assert a.cross(b, nested=True).cross(c).tolist() == [[[(1.1, 100, 999), (1.1, 200, 999)], [(2.2, 100, 999), (2.2, 200, 999)], [(3.3, 100, 999), (3.3, 200, 999)]], [], [[(4.4, 400, 999), (4.4, 400, 888), (4.4, 400, 777)], [(5.5, 400, 999), (5.5, 400, 888), (5.5, 400, 777)]]]
        assert a.cross(b).cross(c, nested=True).tolist() == [[[(1.1, 100, 999)], [(1.1, 200, 999)], [(2.2, 100, 999)], [(2.2, 200, 999)], [(3.3, 100, 999)], [(3.3, 200, 999)]], [], [[(4.4, 400, 999), (4.4, 400, 888), (4.4, 400, 777)], [(5.5, 400, 999), (5.5, 400, 888), (5.5, 400, 777)]]]

    def test_jagged_pairs_argnested(self):
        a = awkward.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
        assert a.pairs().tolist() == [[(1.1, 1.1), (1.1, 2.2), (1.1, 3.3), (2.2, 2.2), (2.2, 3.3), (3.3, 3.3)], [], [(4.4, 4.4), (4.4, 5.5), (5.5, 5.5)]]
        assert a.argpairs().tolist() == [[(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)], [], [(0, 0), (0, 1), (1, 1)]]
        assert a.pairs(nested=True).tolist() == [[[(1.1, 1.1), (1.1, 2.2), (1.1, 3.3)], [(2.2, 2.2), (2.2, 3.3)], [(3.3, 3.3)]], [], [[(4.4, 4.4), (4.4, 5.5)], [(5.5, 5.5)]]]
        assert a.argpairs(nested=True).tolist() == [[[(0, 0), (0, 1), (0, 2)], [(1, 1), (1, 2)], [(2, 2)]], [], [[(0, 0), (0, 1)], [(1, 1)]]]

    def test_jagged_distincts_argnested(self):
        a = awkward.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
        assert a.distincts().tolist() == [[(1.1, 2.2), (1.1, 3.3), (2.2, 3.3)], [], [(4.4, 5.5)]]
        assert a.argdistincts().tolist() == [[(0, 1), (0, 2), (1, 2)], [], [(0, 1)]]
        assert a.distincts(nested=True).tolist() == [[[(1.1, 2.2), (1.1, 3.3)], [(2.2, 3.3)]], [], [[(4.4, 5.5)]]]
        assert a.argdistincts(nested=True).tolist() == [[[(0, 1), (0, 2)], [(1, 2)]], [], [[(0, 1)]]]

    def test_jagged_sum(self):
        a = JaggedArray([0, 3, 3, 5], [3, 3, 5, 10], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        assert a.sum().tolist() == [3.3000000000000003, 0.0, 7.7, 38.5]

        a = JaggedArray([[0, 3], [3, 5]], [[3, 3], [5, 10]], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        assert a.sum(regularaxis=0).tolist() == [[3.3000000000000003, 0.0], [7.7, 38.5]]

        a = JaggedArray([0, 3, 3, 5], [3, 3, 5, 10], [[0.0], [1.1], [2.2], [3.3], [4.4], [5.5], [6.6], [7.7], [8.8], [9.9]])
        assert a.sum(regularaxis=0).tolist() == [[3.3000000000000003], [0.0], [7.7], [38.5]]

        a = JaggedArray([0, 3, 3, 5], [3, 3, 5, 10], [[0.0, 0.0], [1.1, 1.1], [2.2, 2.2], [3.3, 3.3], [4.4, 4.4], [5.5, 5.5], [6.6, 6.6], [7.7, 7.7], [8.8, 8.8], [9.9, 9.9]])
        assert a.sum(regularaxis=0).tolist() == [[3.3000000000000003, 3.3000000000000003], [0.0, 0.0], [7.7, 7.7], [38.5, 38.5]]

    def test_jagged_prod(self):
        a = JaggedArray([0, 3, 3, 5], [3, 3, 5, 10], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        assert a.prod().tolist() == [0.0, 1.0, 14.52, 24350.911200000002]

        a = JaggedArray([[0, 3], [3, 5]], [[3, 3], [5, 10]], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        assert a.prod().tolist() == [[0.0, 1.0], [14.52, 24350.911200000002]]

        a = JaggedArray([0, 3, 3, 5], [3, 3, 5, 10], [[0.0], [1.1], [2.2], [3.3], [4.4], [5.5], [6.6], [7.7], [8.8], [9.9]])
        assert a.prod(regularaxis=0).tolist() == [[0.0], [1.0], [14.52], [24350.911200000002]]

        a = JaggedArray([0, 3, 3, 5], [3, 3, 5, 10], [[0.0, 0.0], [1.1, 1.1], [2.2, 2.2], [3.3, 3.3], [4.4, 4.4], [5.5, 5.5], [6.6, 6.6], [7.7, 7.7], [8.8, 8.8], [9.9, 9.9]])
        assert a.prod(regularaxis=0).tolist() == [[0.0, 0.0], [1.0, 1.0], [14.52, 14.52], [24350.911200000002, 24350.911200000002]]

    def test_jagged_argmin(self):
        a = JaggedArray([0, 3, 3, 5], [3, 3, 5, 10], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        assert a.argmin().tolist() == [[0], [], [0], [0]]

        a = JaggedArray([[0, 3], [3, 5]], [[3, 3], [5, 10]], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        assert a.argmin().tolist() == [[[0], []], [[0], [0]]]

    def test_jagged_argmax(self):
        a = JaggedArray([0, 3, 3, 5], [3, 3, 5, 10], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        assert a.argmax().tolist() == [[2], [], [1], [4]]

        a = JaggedArray([[0, 3], [3, 5]], [[3, 3], [5, 10]], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        assert a.argmax().tolist() == [[[2], []], [[1], [4]]]

    def test_jagged_min(self):
        a = JaggedArray([0, 3, 3, 5], [3, 3, 5, 10], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        assert a.min().tolist() == [0.0, numpy.inf, 3.3, 5.5]

        a = JaggedArray([[0, 3], [3, 5]], [[3, 3], [5, 10]], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        assert a.min().tolist() == [[0.0, numpy.inf], [3.3, 5.5]]

        a = JaggedArray([0, 3, 3, 5], [3, 3, 5, 10], [[0.0], [1.1], [2.2], [3.3], [4.4], [5.5], [6.6], [7.7], [8.8], [9.9]])
        assert a.min(regularaxis=0).tolist() == [[0.0], [numpy.inf], [3.3], [5.5]]

        a = JaggedArray([0, 3, 3, 5], [3, 3, 5, 10], [[0.0, 0.0], [1.1, 1.1], [2.2, 2.2], [3.3, 3.3], [4.4, 4.4], [5.5, 5.5], [6.6, 6.6], [7.7, 7.7], [8.8, 8.8], [9.9, 9.9]])
        assert a.min(regularaxis=0).tolist() == [[0.0, 0.0], [numpy.inf, numpy.inf], [3.3, 3.3], [5.5, 5.5]]

    def test_jagged_max(self):
        a = JaggedArray([0, 3, 3, 5], [3, 3, 5, 10], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        assert a.max().tolist() == [2.2, -numpy.inf, 4.4, 9.9]

        a = JaggedArray([[0, 3], [3, 5]], [[3, 3], [5, 10]], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        assert a.max().tolist() == [[2.2, -numpy.inf], [4.4, 9.9]]

        a = JaggedArray([0, 3, 3, 5], [3, 3, 5, 10], [[0.0], [1.1], [2.2], [3.3], [4.4], [5.5], [6.6], [7.7], [8.8], [9.9]])
        assert a.max(regularaxis=0).tolist() == [[2.2], [-numpy.inf], [4.4], [9.9]]

        a = JaggedArray([0, 3, 3, 5], [3, 3, 5, 10], [[0.0, 0.0], [1.1, 1.1], [2.2, 2.2], [3.3, 3.3], [4.4, 4.4], [5.5, 5.5], [6.6, 6.6], [7.7, 7.7], [8.8, 8.8], [9.9, 9.9]])
        assert a.max(regularaxis=0).tolist() == [[2.2, 2.2], [-numpy.inf, -numpy.inf], [4.4, 4.4], [9.9, 9.9]]

    def test_jagged_concatenate(self):
        lst = [[1, 2, 3], [], [4, 5], [6, 7], [8], [], [9], [10, 11], [12]]
        a_orig = JaggedArray.fromiter(lst)
        a1 = JaggedArray.fromiter(lst[:3])
        a2 = JaggedArray.fromiter(lst[3:6])
        a3 = JaggedArray.fromiter(lst[6:])

        a_instance_concat = a1.concatenate([a2, a3])
        assert a_instance_concat.tolist() == a_orig.tolist()

        a_class_concat = JaggedArray.concatenate([a1, a2, a3])
        assert a_class_concat.tolist() == a_orig.tolist()

    def test_jagged_concatenate_axis1(self):
        a1 = JaggedArray([0, 0, 3, 3, 5], [0, 3, 3, 5, 10], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        b1 = JaggedArray([0, 0, 5, 7, 7], [0, 5, 7, 7, 10], [6.5, 7.6, 8.7, 9.8, 10.9, 4.3, 5.4, 1., 2.1, 3.2])
        c1 = a1.concatenate([b1], axis=1)
        assert c1.tolist() == [[],[0.,1.1,2.2,6.5,7.6,8.7,9.8,10.9],[4.3,5.4],[3.3,4.4],[5.5,6.6,7.7,8.8,9.9,1.,2.1,3.2]]

        # Check that concatenating boolean arrays does not accidently promote them to integers
        a2 = JaggedArray([0], [3], [False, False, False])
        b2 = JaggedArray([0], [3], [True, True, True])
        c2 = a2.concatenate([b2], axis=1)
        assert c2.content.dtype == numpy.dtype(bool)

        # Test some masked arrays
        a3 = a1[[True, True, True, False, True]]
        b3 = b1[[True, True, True, True, False]]
        c3 = a3.concatenate([b3], axis=1)
        assert c3.tolist() == [[],[0.,1.1,2.2,6.5,7.6,8.7,9.8,10.9],[4.3,5.4],[5.5,6.6,7.7,8.8,9.9]]


    def test_jagged_get(self):
        a = JaggedArray.fromoffsets([0, 3, 3, 8, 10, 10], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        assert [a[i].tolist() for i in range(len(a))] == [[0.0, 1.1, 2.2], [], [3.3, 4.4, 5.5, 6.6, 7.7], [8.8, 9.9], []]
        assert [x.tolist() for x in a] == [[0.0, 1.1, 2.2], [], [3.3, 4.4, 5.5, 6.6, 7.7], [8.8, 9.9], []]
        assert [x.tolist() for x in a[:]] == [[0.0, 1.1, 2.2], [], [3.3, 4.4, 5.5, 6.6, 7.7], [8.8, 9.9], []]
        assert [a[i : i + 1].tolist() for i in range(len(a))] == [[[0.0, 1.1, 2.2]], [[]], [[3.3, 4.4, 5.5, 6.6, 7.7]], [[8.8, 9.9]], [[]]]
        assert [a[i : i + 2].tolist() for i in range(len(a) - 1)] == [[[0.0, 1.1, 2.2], []], [[], [3.3, 4.4, 5.5, 6.6, 7.7]], [[3.3, 4.4, 5.5, 6.6, 7.7], [8.8, 9.9]], [[8.8, 9.9], []]]
        assert [x.tolist() for x in a[[2, 1, 0, -2]]] == [[3.3, 4.4, 5.5, 6.6, 7.7], [], [0.0, 1.1, 2.2], [8.8, 9.9]]
        assert [x.tolist() for x in a[[True, False, True, False, True]]] == [[0.0, 1.1, 2.2], [3.3, 4.4, 5.5, 6.6, 7.7], []]

    def test_jagged_get_startsstops(self):
        a = JaggedArray([5, 2, 99, 1], [8, 7, 99, 3], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        assert [x.tolist() for x in a] == [[5.5, 6.6, 7.7], [2.2, 3.3, 4.4, 5.5, 6.6], [], [1.1, 2.2]]
        assert [x.tolist() for x in a[:]] == [[5.5, 6.6, 7.7], [2.2, 3.3, 4.4, 5.5, 6.6], [], [1.1, 2.2]]

    def test_jagged_get2d(self):
        a = JaggedArray.fromoffsets([0, 3, 3, 8, 10, 10], [[0.0, 0.0], [1.1, 1.1], [2.2, 2.2], [3.3, 3.3], [4.4, 4.4], [5.5, 5.5], [6.6, 6.6], [7.7, 7.7], [8.8, 8.8], [9.9, 9.9]])
        assert [a[i].tolist() for i in range(len(a))] == [[[0.0, 0.0], [1.1, 1.1], [2.2, 2.2]], [], [[3.3, 3.3], [4.4, 4.4], [5.5, 5.5], [6.6, 6.6], [7.7, 7.7]], [[8.8, 8.8], [9.9, 9.9]], []]
        assert [x.tolist() for x in a] == [[[0.0, 0.0], [1.1, 1.1], [2.2, 2.2]], [], [[3.3, 3.3], [4.4, 4.4], [5.5, 5.5], [6.6, 6.6], [7.7, 7.7]], [[8.8, 8.8], [9.9, 9.9]], []]
        assert [x.tolist() for x in a[:]] == [[[0.0, 0.0], [1.1, 1.1], [2.2, 2.2]], [], [[3.3, 3.3], [4.4, 4.4], [5.5, 5.5], [6.6, 6.6], [7.7, 7.7]], [[8.8, 8.8], [9.9, 9.9]], []]
        assert [a[i : i + 1].tolist() for i in range(len(a))] == [[[[0.0, 0.0], [1.1, 1.1], [2.2, 2.2]]], [[]], [[[3.3, 3.3], [4.4, 4.4], [5.5, 5.5], [6.6, 6.6], [7.7, 7.7]]], [[[8.8, 8.8], [9.9, 9.9]]], [[]]]
        assert [a[i : i + 2].tolist() for i in range(len(a) - 1)] == [[[[0.0, 0.0], [1.1, 1.1], [2.2, 2.2]], []], [[], [[3.3, 3.3], [4.4, 4.4], [5.5, 5.5], [6.6, 6.6], [7.7, 7.7]]], [[[3.3, 3.3], [4.4, 4.4], [5.5, 5.5], [6.6, 6.6], [7.7, 7.7]], [[8.8, 8.8], [9.9, 9.9]]], [[[8.8, 8.8], [9.9, 9.9]], []]]
        assert [x.tolist() for x in a[[2, 1, 0, -2]]] == [[[3.3, 3.3], [4.4, 4.4], [5.5, 5.5], [6.6, 6.6], [7.7, 7.7]], [], [[0.0, 0.0], [1.1, 1.1], [2.2, 2.2]], [[8.8, 8.8], [9.9, 9.9]]]
        assert [x.tolist() for x in a[[True, False, True, False, True]]] == [[[0.0, 0.0], [1.1, 1.1], [2.2, 2.2]], [[3.3, 3.3], [4.4, 4.4], [5.5, 5.5], [6.6, 6.6], [7.7, 7.7]], []]

    def test_jagged_getstruct(self):
        a = JaggedArray.fromoffsets([0, 3, 3, 8, 10, 10], numpy.array([(0.0, 0.0), (1.1, 1.1), (2.2, 2.2), (3.3, 3.3), (4.4, 4.4), (5.5, 5.5), (6.6, 6.6), (7.7, 7.7), (8.8, 8.8), (9.9, 9.9)], dtype=[("a", float), ("b", float)]))
        assert [a[i].tolist() for i in range(len(a))] == [[(0.0, 0.0), (1.1, 1.1), (2.2, 2.2)], [], [(3.3, 3.3), (4.4, 4.4), (5.5, 5.5), (6.6, 6.6), (7.7, 7.7)], [(8.8, 8.8), (9.9, 9.9)], []]
        assert [x.tolist() for x in a] == [[(0.0, 0.0), (1.1, 1.1), (2.2, 2.2)], [], [(3.3, 3.3), (4.4, 4.4), (5.5, 5.5), (6.6, 6.6), (7.7, 7.7)], [(8.8, 8.8), (9.9, 9.9)], []]
        assert [x.tolist() for x in a[:]] == [[(0.0, 0.0), (1.1, 1.1), (2.2, 2.2)], [], [(3.3, 3.3), (4.4, 4.4), (5.5, 5.5), (6.6, 6.6), (7.7, 7.7)], [(8.8, 8.8), (9.9, 9.9)], []]
        assert [a[i : i + 1].tolist() for i in range(len(a))] == [[[(0.0, 0.0), (1.1, 1.1), (2.2, 2.2)]], [[]], [[(3.3, 3.3), (4.4, 4.4), (5.5, 5.5), (6.6, 6.6), (7.7, 7.7)]], [[(8.8, 8.8), (9.9, 9.9)]], [[]]]
        assert [a[i : i + 2].tolist() for i in range(len(a) - 1)] == [[[(0.0, 0.0), (1.1, 1.1), (2.2, 2.2)], []], [[], [(3.3, 3.3), (4.4, 4.4), (5.5, 5.5), (6.6, 6.6), (7.7, 7.7)]], [[(3.3, 3.3), (4.4, 4.4), (5.5, 5.5), (6.6, 6.6), (7.7, 7.7)], [(8.8, 8.8), (9.9, 9.9)]], [[(8.8, 8.8), (9.9, 9.9)], []]]
        assert [x.tolist() for x in a[[2, 1, 0, -2]]] == [[(3.3, 3.3), (4.4, 4.4), (5.5, 5.5), (6.6, 6.6), (7.7, 7.7)], [], [(0.0, 0.0), (1.1, 1.1), (2.2, 2.2)], [(8.8, 8.8), (9.9, 9.9)]]
        assert [x.tolist() for x in a[[True, False, True, False, True]]] == [[(0.0, 0.0), (1.1, 1.1), (2.2, 2.2)], [(3.3, 3.3), (4.4, 4.4), (5.5, 5.5), (6.6, 6.6), (7.7, 7.7)], []]

    def test_jagged_zip(self):
        a = awkward.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
        b = awkward.JaggedArray([1, 5, 5], [4, 5, 7], [999, 10, 20, 30, 999, 40, 50, 999])
        c = numpy.array([100, 200, 300])
        d = 1000
        assert awkward.JaggedArray.zip(one=a, two=b).tolist() == [[{"one": 1.1, "two": 10}, {"one": 2.2, "two": 20}, {"one": 3.3, "two": 30}], [], [{"one": 4.4, "two": 40}, {"one": 5.5, "two": 50}]]
        assert awkward.JaggedArray.zip(one=b, two=a).tolist() == [[{"one": 10, "two": 1.1}, {"one": 20, "two": 2.2}, {"one": 30, "two": 3.3}], [], [{"one": 40, "two": 4.4}, {"one": 50, "two": 5.5}]]
        assert awkward.JaggedArray.zip(one=b, two=c).tolist() == [[{"one": 10, "two": 100}, {"one": 20, "two": 100}, {"one": 30, "two": 100}], [], [{"one": 40, "two": 300}, {"one": 50, "two": 300}]]
        assert awkward.JaggedArray.zip(one=b, two=d).tolist() == [[{"one": 10, "two": 1000}, {"one": 20, "two": 1000}, {"one": 30, "two": 1000}], [], [{"one": 40, "two": 1000}, {"one": 50, "two": 1000}]]
        assert a.zip(b).tolist() == [[(1.1, 10), (2.2, 20), (3.3, 30)], [], [(4.4, 40), (5.5, 50)]]
        assert b.zip(a).tolist() == [[(10, 1.1), (20, 2.2), (30, 3.3)], [], [(40, 4.4), (50, 5.5)]]
        assert b.zip(c).tolist() == [[(10, 100), (20, 100), (30, 100)], [], [(40, 300), (50, 300)]]
        assert b.zip(d).tolist() == [[(10, 1000), (20, 1000), (30, 1000)], [], [(40, 1000), (50, 1000)]]

    def test_jagged_pad(self):
        a = awkward.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
        assert a.pad(4, clip=True).tolist() == [[1.1, 2.2, 3.3, None], [None, None, None, None], [4.4, 5.5, None, None]]
        assert a.pad(4, numpy.ma.masked, clip=True).regular().tolist() == [[1.1, 2.2, 3.3, None], [None, None, None, None], [4.4, 5.5, None, None]]

        assert a.pad(4).tolist() == [[1.1, 2.2, 3.3, None], [None, None, None, None], [4.4, 5.5, None, None]]
        assert a.pad(4, numpy.ma.masked).regular().tolist() == [[1.1, 2.2, 3.3, None], [None, None, None, None], [4.4, 5.5, None, None]]

        a = awkward.fromiter([[1.1, 2.2, 3.3, 4.4, 5.5], [], [6.6, 7.7, 8.8], [9.9]])
        assert a.pad(3).tolist() == [[1.1, 2.2, 3.3, 4.4, 5.5], [None, None, None], [6.6, 7.7, 8.8], [9.9, None, None]]
        assert a.pad(3, clip=True).tolist() == [[1.1, 2.2, 3.3], [None, None, None], [6.6, 7.7, 8.8], [9.9, None, None]]

    def test_jagged_fillna(self):
        a = awkward.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
        assert a.pad(4).fillna(999).tolist() == [[1.1, 2.2, 3.3, 999], [999, 999, 999, 999], [4.4, 5.5, 999, 999]]
        assert a.pad(4, numpy.ma.masked).fillna(999).regular().tolist() == [[1.1, 2.2, 3.3, 999], [999, 999, 999, 999], [4.4, 5.5, 999, 999]]
