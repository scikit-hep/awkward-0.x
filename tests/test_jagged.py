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

class TestJagged(unittest.TestCase):
    def runTest(self):
        pass

    def test_jagged_offsets(self):
        offsets = numpy.array([0, 3, 3, 8, 10, 10])
        a = JaggedArray.fromoffsets(offsets, [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        self.assertTrue(offsets is a.offsets)

        a = JaggedArray([5, 2, 99, 1], [8, 7, 99, 3], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        self.assertRaises(ValueError, lambda: a.offsets)

    def test_jagged_iterable(self):
        a = JaggedArray.fromiterable([[0.0, 1.1, 2.2], [], [3.3, 4.4, 5.5, 6.6, 7.7], [8.8, 9.9], []])
        self.assertEqual([x.tolist() for x in a], [[0.0, 1.1, 2.2], [], [3.3, 4.4, 5.5, 6.6, 7.7], [8.8, 9.9], []])

    def test_jagged_compatible(self):
        a = JaggedArray.fromoffsets([0, 3, 3, 8, 10, 10], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        b = JaggedArray([0, 3, 3, 8, 10], [3, 3, 8, 10, 10], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        self.assertTrue(JaggedArray.compatible(a, b))

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

    def test_jagged_getempty(self):
        a = JaggedArray([], [], [0.0, 1.1, 2.2, 3.3, 4.4])
        self.assertEqual(a[:].tolist(), [])

    def test_jagged_jagged(self):
        a = JaggedArray.fromoffsets([0, 3, 3, 5], JaggedArray.fromoffsets([0, 3, 3, 8, 10, 10], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
        self.assertEqual([a[i].tolist() for i in range(len(a))], [[[0.0, 1.1, 2.2], [], [3.3, 4.4, 5.5, 6.6, 7.7]], [], [[8.8, 9.9], []]])
        self.assertEqual([x.tolist() for x in a], [[[0.0, 1.1, 2.2], [], [3.3, 4.4, 5.5, 6.6, 7.7]], [], [[8.8, 9.9], []]])
        self.assertEqual([x.tolist() for x in a[:]], [[[0.0, 1.1, 2.2], [], [3.3, 4.4, 5.5, 6.6, 7.7]], [], [[8.8, 9.9], []]])
        self.assertEqual([x.tolist() for x in a[:-1]], [[[0.0, 1.1, 2.2], [], [3.3, 4.4, 5.5, 6.6, 7.7]], []])
        self.assertEqual([x.tolist() for x in a[[2, 1, 0]]], [[[8.8, 9.9], []], [], [[0.0, 1.1, 2.2], [], [3.3, 4.4, 5.5, 6.6, 7.7]]])
        self.assertEqual([x.tolist() for x in a[[True, True, False]]], [[[0.0, 1.1, 2.2], [], [3.3, 4.4, 5.5, 6.6, 7.7]], []])

    def test_jagged_set(self):
        a = JaggedArray.fromoffsets([0, 3, 3, 8, 10, 10], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        a[3] = 999
        self.assertEqual([x.tolist() for x in a], [[0.0, 1.1, 2.2], [], [3.3, 4.4, 5.5, 6.6, 7.7], [999.0, 999.0], []])

        a = JaggedArray.fromoffsets([0, 3, 3, 8, 10, 10], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        a[3] = [999]
        self.assertEqual([x.tolist() for x in a], [[0.0, 1.1, 2.2], [], [3.3, 4.4, 5.5, 6.6, 7.7], [999.0, 999.0], []])

        a = JaggedArray.fromoffsets([0, 3, 3, 8, 10, 10], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        def quickie():
            a[3] = [123, 456, 789]
        self.assertRaises(ValueError, quickie)

        a = JaggedArray.fromoffsets([0, 3, 3, 8, 10, 10], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        a[0:3] = 999
        self.assertEqual([x.tolist() for x in a], [[999.0, 999.0, 999.0], [], [999.0, 999.0, 999.0, 999.0, 999.0], [8.8, 9.9], []])

        a = JaggedArray.fromoffsets([0, 3, 3, 8, 10, 10], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        a[0:3] = [999]
        self.assertEqual([x.tolist() for x in a], [[999.0, 999.0, 999.0], [], [999.0, 999.0, 999.0, 999.0, 999.0], [8.8, 9.9], []])

        a = JaggedArray.fromoffsets([0, 3, 3, 8, 10, 10], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        a[0:3] = [101, 102, 103, 104, 105, 106, 107, 108]
        self.assertEqual([x.tolist() for x in a], [[101.0, 102.0, 103.0], [], [104.0, 105.0, 106.0, 107.0, 108.0], [8.8, 9.9], []])

        a = JaggedArray.fromoffsets([0, 3, 3, 8, 10, 10], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        a[0:3] = JaggedArray.fromoffsets([0, 3, 3, 8], [101, 102, 103, 104, 105, 106, 107, 108])
        self.assertEqual([x.tolist() for x in a], [[101.0, 102.0, 103.0], [], [104.0, 105.0, 106.0, 107.0, 108.0], [8.8, 9.9], []])

    def test_bytejagged_offsets(self):
        a = ByteJaggedArray.fromoffsets([5, 17, 17, 25], b"\xff\x00\x00\x00\x00\x01\x00\x00\x00\x02\x00\x00\x00\x03\x00\x00\x00\x04\x00\x00\x00\x05\x00\x00\x00\xff\xff", numpy.int32)
        self.assertEqual([x.tolist() for x in a], [[1, 2, 3], [], [4, 5]])

        a = ByteJaggedArray([5, 17, 19], [17, 17, 27], b"\xff\x00\x00\x00\x00\x01\x00\x00\x00\x02\x00\x00\x00\x03\x00\x00\x00\xff\xff\x04\x00\x00\x00\x05\x00\x00\x00\xff", numpy.int32)
        self.assertEqual([x.tolist() for x in a], [[1, 2, 3], [], [4, 5]])

    def test_bytejagged_iterable(self):
        a = ByteJaggedArray.fromiterable([[1, 2, 3], [], [4, 5]])
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

    def test_bytejagged_set(self):
        a = ByteJaggedArray([5, 17, 19], [17, 17, 27], numpy.array([255, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 255, 255, 4, 0, 0, 0, 5, 0, 0, 0, 255], "u1").tobytes(), numpy.int32)
        a[2] = 123
        self.assertEqual(a.content.tobytes(), b"\xff\x00\x00\x00\x00\x01\x00\x00\x00\x02\x00\x00\x00\x03\x00\x00\x00\xff\xff{\x00\x00\x00{\x00\x00\x00\xff")
        self.assertEqual([a[i].tolist() for i in range(len(a))], [[1, 2, 3], [], [123, 123]])

        a = ByteJaggedArray([5, 17, 19], [17, 17, 27], numpy.array([255, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 255, 255, 4, 0, 0, 0, 5, 0, 0, 0, 255], "u1").tobytes(), numpy.int32)
        a[2] = [123]
        self.assertEqual(a.content.tobytes(), b"\xff\x00\x00\x00\x00\x01\x00\x00\x00\x02\x00\x00\x00\x03\x00\x00\x00\xff\xff{\x00\x00\x00{\x00\x00\x00\xff")
        self.assertEqual([a[i].tolist() for i in range(len(a))], [[1, 2, 3], [], [123, 123]])

        a = ByteJaggedArray([5, 17, 19], [17, 17, 27], numpy.array([255, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 255, 255, 4, 0, 0, 0, 5, 0, 0, 0, 255], "u1").tobytes(), numpy.int32)
        a[2] = 123, 125
        self.assertEqual(a.content.tobytes(), b"\xff\x00\x00\x00\x00\x01\x00\x00\x00\x02\x00\x00\x00\x03\x00\x00\x00\xff\xff{\x00\x00\x00}\x00\x00\x00\xff")
        self.assertEqual([a[i].tolist() for i in range(len(a))], [[1, 2, 3], [], [123, 125]])

        a = ByteJaggedArray([5, 17, 19], [17, 17, 27], numpy.array([255, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 255, 255, 4, 0, 0, 0, 5, 0, 0, 0, 255], "u1").tobytes(), numpy.int32)
        a[:] = 123
        self.assertEqual(a.content.tobytes(), b"\xff\x00\x00\x00\x00{\x00\x00\x00{\x00\x00\x00{\x00\x00\x00\xff\xff{\x00\x00\x00{\x00\x00\x00\xff")
        self.assertEqual([a[i].tolist() for i in range(len(a))], [[123, 123, 123], [], [123, 123]])

        a = ByteJaggedArray([5, 17, 19], [17, 17, 27], numpy.array([255, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 255, 255, 4, 0, 0, 0, 5, 0, 0, 0, 255], "u1").tobytes(), numpy.int32)
        a[:] = [123]
        self.assertEqual(a.content.tobytes(), b"\xff\x00\x00\x00\x00{\x00\x00\x00{\x00\x00\x00{\x00\x00\x00\xff\xff{\x00\x00\x00{\x00\x00\x00\xff")
        self.assertEqual([a[i].tolist() for i in range(len(a))], [[123, 123, 123], [], [123, 123]])

        a = ByteJaggedArray([5, 17, 19], [17, 17, 27], numpy.array([255, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 255, 255, 4, 0, 0, 0, 5, 0, 0, 0, 255], "u1").tobytes(), numpy.int32)
        a[:] = [3, 2, 1, 5, 4]
        self.assertEqual(a.content.tobytes(), b"\xff\x00\x00\x00\x00\x03\x00\x00\x00\x02\x00\x00\x00\x01\x00\x00\x00\xff\xff\x05\x00\x00\x00\x04\x00\x00\x00\xff")
        self.assertEqual([a[i].tolist() for i in range(len(a))], [[3, 2, 1], [], [5, 4]])

        a = ByteJaggedArray([5, 17, 19], [17, 17, 27], numpy.array([255, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 255, 255, 4, 0, 0, 0, 5, 0, 0, 0, 255], "u1").tobytes(), numpy.int32)
        a[:] = JaggedArray.fromiterable([[3, 2, 1], [], [5, 4]])
        self.assertEqual(a.content.tobytes(), b"\xff\x00\x00\x00\x00\x03\x00\x00\x00\x02\x00\x00\x00\x01\x00\x00\x00\xff\xff\x05\x00\x00\x00\x04\x00\x00\x00\xff")
        self.assertEqual([a[i].tolist() for i in range(len(a))], [[3, 2, 1], [], [5, 4]])
