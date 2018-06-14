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

class TestChunked(unittest.TestCase):
    def runTest(self):
        pass

    def test_chunked_iteration(self):
        a = ChunkedArray([[], [0, 1, 2, 3, 4], [5, 6], [], [7, 8, 9], []])
        self.assertEqual(a.tolist(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

        a = ChunkedArray([[0, 1, 2, 3, 4], [5, 6], [], [7, 8, 9], []])
        self.assertEqual(a.tolist(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

        a = ChunkedArray([[], [0, 1, 2, 3, 4], [5, 6], [], [7, 8, 9]])
        self.assertEqual(a.tolist(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

        a = ChunkedArray([[]])
        self.assertEqual(a.tolist(), [])

        a = ChunkedArray([])
        self.assertEqual(a.tolist(), [])

    def test_chunked_dtype(self):
        a = ChunkedArray([[], [0, 1, 2, 3, 4], [5, 6], [], [7, 8, 9], []])
        self.assertEqual(a.dtype, numpy.dtype(float))

        a = ChunkedArray([[0, 1, 2, 3, 4], [5, 6], [], [7, 8, 9], []])
        self.assertEqual(a.dtype, numpy.dtype(int))

        a = ChunkedArray([])
        self.assertRaises(ValueError, lambda: a.dtype)

    def test_chunked_get(self):
        a = ChunkedArray([[], [0, 1, 2, 3, 4], [5, 6], [], [7, 8, 9], []])
        self.assertEqual([a[i] for i in range(10)], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

        self.assertEqual([a[i : i + 1].tolist() for i in range(10)], [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]])
        self.assertEqual([a[i : i + 2].tolist() for i in range(10 - 1)], [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9]])
        self.assertEqual([a[i : i + 3].tolist() for i in range(10 - 2)], [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8], [7, 8, 9]])
        self.assertEqual([a[i : i + 4].tolist() for i in range(10 - 3)], [[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7], [5, 6, 7, 8], [6, 7, 8, 9]])
        self.assertEqual([a[i : i + 4 : 2].tolist() for i in range(10 - 3)], [[0, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7], [6, 8]])
        self.assertEqual([a[i : i + 4 : 3].tolist() for i in range(10 - 3)], [[0, 3], [1, 4], [2, 5], [3, 6], [4, 7], [5, 8], [6, 9]])
        self.assertEqual([a[i + 4 : i : -1].tolist() for i in range(10 - 4)], [[4, 3, 2, 1], [5, 4, 3, 2], [6, 5, 4, 3], [7, 6, 5, 4], [8, 7, 6, 5], [9, 8, 7, 6]])
        self.assertEqual([a[i + 4 : i : -2].tolist() for i in range(10 - 4)], [[4, 2], [5, 3], [6, 4], [7, 5], [8, 6], [9, 7]])
        self.assertEqual([a[i + 4 : i : -3].tolist() for i in range(10 - 4)], [[4, 1], [5, 2], [6, 3], [7, 4], [8, 5], [9, 6]])

        self.assertEqual(a[4:].tolist(), [4, 5, 6, 7, 8, 9])
        self.assertEqual(a[5:].tolist(), [5, 6, 7, 8, 9])
        self.assertEqual(a[6:].tolist(), [6, 7, 8, 9])
        self.assertEqual(a[7:].tolist(), [7, 8, 9])
        self.assertEqual(a[8:].tolist(), [8, 9])
        self.assertEqual(a[:4].tolist(), [0, 1, 2, 3])
        self.assertEqual(a[:5].tolist(), [0, 1, 2, 3, 4])
        self.assertEqual(a[:6].tolist(), [0, 1, 2, 3, 4, 5])
        self.assertEqual(a[:7].tolist(), [0, 1, 2, 3, 4, 5, 6])
        self.assertEqual(a[:8].tolist(), [0, 1, 2, 3, 4, 5, 6, 7])

        self.assertEqual(a[4::2].tolist(), [4, 6, 8])
        self.assertEqual(a[5::2].tolist(), [5, 7, 9])
        self.assertEqual(a[6::2].tolist(), [6, 8])
        self.assertEqual(a[7::2].tolist(), [7, 9])
        self.assertEqual(a[8::2].tolist(), [8])
        self.assertEqual(a[:4:2].tolist(), [0, 2])
        self.assertEqual(a[:5:2].tolist(), [0, 2, 4])
        self.assertEqual(a[:6:2].tolist(), [0, 2, 4])
        self.assertEqual(a[:7:2].tolist(), [0, 2, 4, 6])
        self.assertEqual(a[:8:2].tolist(), [0, 2, 4, 6])

        self.assertEqual(a[4::-1].tolist(), [4, 3, 2, 1, 0])
        self.assertEqual(a[5::-1].tolist(), [5, 4, 3, 2, 1, 0])
        self.assertEqual(a[6::-1].tolist(), [6, 5, 4, 3, 2, 1, 0])
        self.assertEqual(a[7::-1].tolist(), [7, 6, 5, 4, 3, 2, 1, 0])
        self.assertEqual(a[8::-1].tolist(), [8, 7, 6, 5, 4, 3, 2, 1, 0])
        self.assertEqual(a[:4:-1].tolist(), [9, 8, 7, 6, 5])
        self.assertEqual(a[:5:-1].tolist(), [9, 8, 7, 6])
        self.assertEqual(a[:6:-1].tolist(), [9, 8, 7])
        self.assertEqual(a[:7:-1].tolist(), [9, 8])
        self.assertEqual(a[:8:-1].tolist(), [9])

        self.assertEqual(a[4::-2].tolist(), [4, 2, 0])
        self.assertEqual(a[5::-2].tolist(), [5, 3, 1])
        self.assertEqual(a[6::-2].tolist(), [6, 4, 2, 0])
        self.assertEqual(a[7::-2].tolist(), [7, 5, 3, 1])
        self.assertEqual(a[8::-2].tolist(), [8, 6, 4, 2, 0])
        self.assertEqual(a[:4:-2].tolist(), [9, 7, 5])
        self.assertEqual(a[:5:-2].tolist(), [9, 7])
        self.assertEqual(a[:6:-2].tolist(), [9, 7])
        self.assertEqual(a[:7:-2].tolist(), [9])
        self.assertEqual(a[:8:-2].tolist(), [9])

        self.assertEqual(a[[8, 6, 4, 5, 0]].tolist(), [8, 6, 4, 5, 0])
        self.assertEqual(a[[6, 4, 5, 0]].tolist(), [6, 4, 5, 0])
        self.assertEqual(a[[5, 6, 4, 5, 5, 5, 0]].tolist(), [5, 6, 4, 5, 5, 5, 0])
        self.assertRaises(IndexError, lambda: a[[8, 6, 4, 5, 0, 99]])


