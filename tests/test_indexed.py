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

    ################### old tests

    # def test_indexed_get(self):
    #     a = IndexedArray([3, 2, 4, 2, 2, 4, 0], [0.0, 1.1, 2.2, 3.3, 4.4])
    #     self.assertEqual([x for x in a], [3.3, 2.2, 4.4, 2.2, 2.2, 4.4, 0.0])
    #     self.assertEqual([a[i] for i in range(len(a))], [3.3, 2.2, 4.4, 2.2, 2.2, 4.4, 0.0])
    #     self.assertEqual([a[i : i + 1].tolist() for i in range(len(a))], [[3.3], [2.2], [4.4], [2.2], [2.2], [4.4], [0.0]])
    #     self.assertEqual([a[i : i + 2].tolist() for i in range(len(a) - 1)], [[3.3, 2.2], [2.2, 4.4], [4.4, 2.2], [2.2, 2.2], [2.2, 4.4], [4.4, 0.0]])
    #     self.assertEqual(a[:].tolist(), [3.3, 2.2, 4.4, 2.2, 2.2, 4.4, 0.0])
    #     self.assertEqual(a[[6, 5, 4, 3, 2, 1, 0]].tolist(), [0.0, 4.4, 2.2, 2.2, 4.4, 2.2, 3.3])
    #     self.assertEqual(a[[True, False, True, False, True, False, True]].tolist(), [3.3, 4.4, 2.2, 0.0])
    #     self.assertEqual(a[[-1, -2, -3, -4, -5, -6, -7]].tolist(), [0.0, 4.4, 2.2, 2.2, 4.4, 2.2, 3.3])

    # def test_indexed_get2d(self):
    #     a = IndexedArray([3, 2, 4, 2, 2, 4, 0], [[0.0, 0.0], [1.1, 1.1], [2.2, 2.2], [3.3, 3.3], [4.4, 4.4]])
    #     self.assertEqual([x.tolist() for x in a], [[3.3, 3.3], [2.2, 2.2], [4.4, 4.4], [2.2, 2.2], [2.2, 2.2], [4.4, 4.4], [0.0, 0.0]])
    #     self.assertEqual([a[i].tolist() for i in range(len(a))], [[3.3, 3.3], [2.2, 2.2], [4.4, 4.4], [2.2, 2.2], [2.2, 2.2], [4.4, 4.4], [0.0, 0.0]])
    #     self.assertEqual(a[:].tolist(), [[3.3, 3.3], [2.2, 2.2], [4.4, 4.4], [2.2, 2.2], [2.2, 2.2], [4.4, 4.4], [0.0, 0.0]])
    #     self.assertEqual(a[[6, 5, 4, 3, 2, 1, 0]].tolist(), [[0.0, 0.0], [4.4, 4.4], [2.2, 2.2], [2.2, 2.2], [4.4, 4.4], [2.2, 2.2], [3.3, 3.3]])
    #     self.assertEqual(a[[True, False, True, False, True, False, True]].tolist(), [[3.3, 3.3], [4.4, 4.4], [2.2, 2.2], [0.0, 0.0]])

    # def test_indexed_getstruct(self):
    #     a = IndexedArray([3, 2, 4, 2, 2, 4, 0], numpy.array([(0.0, 0.0), (1.1, 1.1), (2.2, 2.2), (3.3, 3.3), (4.4, 4.4)], dtype=[("a", float), ("b", float)]))
    #     self.assertEqual([x.tolist() for x in a], [(3.3, 3.3), (2.2, 2.2), (4.4, 4.4), (2.2, 2.2), (2.2, 2.2), (4.4, 4.4), (0.0, 0.0)])
    #     self.assertEqual([a[i].tolist() for i in range(len(a))], [(3.3, 3.3), (2.2, 2.2), (4.4, 4.4), (2.2, 2.2), (2.2, 2.2), (4.4, 4.4), (0.0, 0.0)])
    #     self.assertEqual(a[:].tolist(), [(3.3, 3.3), (2.2, 2.2), (4.4, 4.4), (2.2, 2.2), (2.2, 2.2), (4.4, 4.4), (0.0, 0.0)])
    #     self.assertEqual(a[[6, 5, 4, 3, 2, 1, 0]].tolist(), [(0.0, 0.0), (4.4, 4.4), (2.2, 2.2), (2.2, 2.2), (4.4, 4.4), (2.2, 2.2), (3.3, 3.3)])
    #     self.assertEqual(a[[True, False, True, False, True, False, True]].tolist(), [(3.3, 3.3), (4.4, 4.4), (2.2, 2.2), (0.0, 0.0)])

    # def test_indexed_getempty(self):
    #     a = IndexedArray([], [0.0, 1.1, 2.2, 3.3, 4.4])
    #     self.assertEqual(a[:].tolist(), [])

    # def test_indexed_indexed(self):
    #     a = IndexedArray([6, 5, 4, 3, 2, 1, 0], IndexedArray([3, 2, 4, 2, 2, 4, 0], [0.0, 1.1, 2.2, 3.3, 4.4]))
    #     self.assertEqual([x for x in a], [0.0, 4.4, 2.2, 2.2, 4.4, 2.2, 3.3])
    #     self.assertEqual([a[i] for i in range(len(a))], [0.0, 4.4, 2.2, 2.2, 4.4, 2.2, 3.3])
    #     self.assertEqual(a[:].tolist(), [0.0, 4.4, 2.2, 2.2, 4.4, 2.2, 3.3])

    #     a = IndexedArray([6, 5, 4, 3, 6], IndexedArray([3, 2, 4, 2, 2, 4, 0], [0.0, 1.1, 2.2, 3.3, 4.4]))
    #     self.assertEqual([x for x in a], [0.0, 4.4, 2.2, 2.2, 0.0])
    #     self.assertEqual([a[i] for i in range(len(a))], [0.0, 4.4, 2.2, 2.2, 0.0])
    #     self.assertEqual(a[:].tolist(), [0.0, 4.4, 2.2, 2.2, 0.0])

    # def test_byteindexed_get(self):
    #     a = ByteIndexedArray([12, 8, 4, 0], b"\x00\x00\x00\x00\x01\x00\x00\x00\x02\x00\x00\x00\x03\x00\x00\x00", numpy.int32)
    #     self.assertEqual([x for x in a], [3, 2, 1, 0])
    #     self.assertEqual([a[i] for i in range(len(a))], [3, 2, 1, 0])
    #     self.assertEqual(a[:].tolist(), [3, 2, 1, 0])
    #     self.assertEqual(a[[3, 2, 1, 0]].tolist(), [0, 1, 2, 3])
    #     self.assertEqual(a[[True, False, True, False]].tolist(), [3, 1])

    # def test_byteindexed_get5byte(self):
    #     a = ByteIndexedArray([15, 10, 5, 1], b"\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x02\x00\x00\x00\x00\x03\x00\x00\x00\x00", numpy.int32)
    #     self.assertEqual([x for x in a], [3, 2, 1, 0])
    #     self.assertEqual([a[i] for i in range(len(a))], [3, 2, 1, 0])
    #     self.assertEqual(a[:].tolist(), [3, 2, 1, 0])
    #     self.assertEqual(a[[3, 2, 1, 0]].tolist(), [0, 1, 2, 3])
    #     self.assertEqual(a[[True, False, True, False]].tolist(), [3, 1])

    # def test_indexed_byteindexed(self):
    #     a = IndexedArray([1, 2, 3], ByteIndexedArray([12, 8, 4, 0], b"\x00\x00\x00\x00\x01\x00\x00\x00\x02\x00\x00\x00\x03\x00\x00\x00", numpy.int32))
    #     self.assertEqual([a[i] for i in range(len(a))], [2, 1, 0])
    #     self.assertEqual(a[:].tolist(), [2, 1, 0])

    # def test_union_get(self):
    #     a = UnionArray([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [[0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], [0, 100, 200, 300, 400, 500, 600, 700, 800, 900]])
    #     self.assertEqual(a.tolist(), [0.0, 100, 2.2, 300, 4.4, 500, 6.6, 700, 8.8, 900])
