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
        self.assertEqual(a.dtype, numpy.dtype(int))

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

        self.assertEqual(a[[True, False, True, False, True, False, True, False, True, False]].tolist(), [0, 2, 4, 6, 8])
        self.assertEqual(a[[False, False, False, False, False, False, False, False, False, False]].tolist(), [])
        self.assertEqual(a[[True, True, True, True, True, True, True, True, True, True]].tolist(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        self.assertRaises(IndexError, lambda: a[[True, True, True, True, True, True, True, True, True, True, True]])
        self.assertRaises(IndexError, lambda: a[[True, True, True, True, True, True, True, True, True]])

    def test_chunked_get2d(self):
        a = ChunkedArray([[], [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]], [[5, 5], [6, 6]], [], [[7, 7], [8, 8], [9, 9]], []])
        self.assertEqual([a[i].tolist() for i in range(10)], [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9]])
        self.assertEqual(a[4:].tolist(), [[4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9]])
        self.assertEqual(a[[8, 6, 4, 5, 0]].tolist(), [[8, 8], [6, 6], [4, 4], [5, 5], [0, 0]])
        self.assertEqual(a[[True, False, True, False, True, False, True, False, True, False]].tolist(), [[0, 0], [2, 2], [4, 4], [6, 6], [8, 8]])

        a = ChunkedArray([[], [[0.0, 0.0], [1.0, 1.1], [2.0, 2.2], [3.0, 3.3], [4.0, 4.4]], [[5.0, 5.5], [6.0, 6.6]], [], [[7.0, 7.7], [8.0, 8.8], [9.0, 9.9]], []])
        self.assertEqual([a[i, 0].tolist() for i in range(10)], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
        self.assertEqual([a[i, 1].tolist() for i in range(10)], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        self.assertEqual(a[4:, 0].tolist(), [4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
        self.assertEqual(a[4:, 1].tolist(), [4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        self.assertEqual(a[[8, 6, 4, 5, 0], 0].tolist(), [8.0, 6.0, 4.0, 5.0, 0.0])
        self.assertEqual(a[[8, 6, 4, 5, 0], 1].tolist(), [8.8, 6.6, 4.4, 5.5, 0.0])
        self.assertEqual(a[[True, False, True, False, True, False, True, False, True, False], 0].tolist(), [0.0, 2.0, 4.0, 6.0, 8.0])
        self.assertEqual(a[[True, False, True, False, True, False, True, False, True, False], 1].tolist(), [0.0, 2.2, 4.4, 6.6, 8.8])

    def test_chunked_set_const(self):
        a = ChunkedArray([[], [0, 1, 2, 3, 4], [5, 6], [], [7, 8, 9], []])
        a[0] = 999
        self.assertEqual(a.tolist(), [999, 1, 2, 3, 4, 5, 6, 7, 8, 9])

        a = ChunkedArray([[], [0, 1, 2, 3, 4], [5, 6], [], [7, 8, 9], []])
        a[9] = 999
        self.assertEqual(a.tolist(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 999])

        a = ChunkedArray([[], [0, 1, 2, 3, 4], [5, 6], [], [7, 8, 9], []])
        a[5] = 999
        self.assertEqual(a.tolist(), [0, 1, 2, 3, 4, 999, 6, 7, 8, 9])

        a = ChunkedArray([[], [0, 1, 2, 3, 4], [5, 6], [], [7, 8, 9], []])
        a[:] = 999
        self.assertEqual(a.tolist(), [999, 999, 999, 999, 999, 999, 999, 999, 999, 999])

        a = ChunkedArray([[], [0, 1, 2, 3, 4], [5, 6], [], [7, 8, 9], []])
        a[5:] = 999
        self.assertEqual(a.tolist(), [0, 1, 2, 3, 4, 999, 999, 999, 999, 999])

        a = ChunkedArray([[], [0, 1, 2, 3, 4], [5, 6], [], [7, 8, 9], []])
        a[:5] = 999
        self.assertEqual(a.tolist(), [999, 999, 999, 999, 999, 5, 6, 7, 8, 9])

        a = ChunkedArray([[], [0, 1, 2, 3, 4], [5, 6], [], [7, 8, 9], []])
        a[3:8] = 999
        self.assertEqual(a.tolist(), [0, 1, 2, 999, 999, 999, 999, 999, 8, 9])

        a = ChunkedArray([[], [0, 1, 2, 3, 4], [5, 6], [], [7, 8, 9], []])
        a[3:8:2] = 999
        self.assertEqual(a.tolist(), [0, 1, 2, 999, 4, 999, 6, 999, 8, 9])

        a = ChunkedArray([[], [0, 1, 2, 3, 4], [5, 6], [], [7, 8, 9], []])
        a[::3] = 999
        self.assertEqual(a.tolist(), [999, 1, 2, 999, 4, 5, 999, 7, 8, 999])

        a = ChunkedArray([[], [0, 1, 2, 3, 4], [5, 6], [], [7, 8, 9], []])
        a[1::3] = 999
        self.assertEqual(a.tolist(), [0, 999, 2, 3, 999, 5, 6, 999, 8, 9])

        a = ChunkedArray([[], [0, 1, 2, 3, 4], [5, 6], [], [7, 8, 9], []])
        a[7:2:-1] = 999
        self.assertEqual(a.tolist(), [0, 1, 2, 999, 999, 999, 999, 999, 8, 9])

        a = ChunkedArray([[], [0, 1, 2, 3, 4], [5, 6], [], [7, 8, 9], []])
        a[7:2:-2] = 999
        self.assertEqual(a.tolist(), [0, 1, 2, 999, 4, 999, 6, 999, 8, 9])

        a = ChunkedArray([[], [0, 1, 2, 3, 4], [5, 6], [], [7, 8, 9], []])
        a[::4] = 999
        self.assertEqual(a.tolist(), [999, 1, 2, 3, 999, 5, 6, 7, 999, 9])

        a = ChunkedArray([[], [0, 1, 2, 3, 4], [5, 6], [], [7, 8, 9], []])
        a[1::4] = 999
        self.assertEqual(a.tolist(), [0, 999, 2, 3, 4, 999, 6, 7, 8, 999])

        a = ChunkedArray([[], [0, 1, 2, 3, 4], [5, 6], [], [7, 8, 9], []])
        a[numpy.empty(0, dtype=int)] = 999
        self.assertEqual(a.tolist(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

        a = ChunkedArray([[], [0, 1, 2, 3, 4], [5, 6], [], [7, 8, 9], []])
        a[[5]] = 999
        self.assertEqual(a.tolist(), [0, 1, 2, 3, 4, 999, 6, 7, 8, 9])

        a = ChunkedArray([[], [0, 1, 2, 3, 4], [5, 6], [], [7, 8, 9], []])
        a[[8, 6, 3, 4, 5]] = 999
        self.assertEqual(a.tolist(), [0, 1, 2, 999, 999, 999, 999, 7, 999, 9])

        a = ChunkedArray([[], [0, 1, 2, 3, 4], [5, 6], [], [7, 8, 9], []])
        a[[True, False, True, False, True, False, True, False, True, False]] = 999
        self.assertEqual(a.tolist(), [999, 1, 999, 3, 999, 5, 999, 7, 999, 9])

        a = ChunkedArray([[], [0, 1, 2, 3, 4], [5, 6], [], [7, 8, 9], []])
        a[[True, False, False, False, True, False, False, False, False, False]] = 999
        self.assertEqual(a.tolist(), [999, 1, 2, 3, 999, 5, 6, 7, 8, 9])

    def test_chunked_set_singleton(self):
        a = ChunkedArray([[], [0, 1, 2, 3, 4], [5, 6], [], [7, 8, 9], []])
        a[:] = [999]
        self.assertEqual(a.tolist(), [999, 999, 999, 999, 999, 999, 999, 999, 999, 999])

        a = ChunkedArray([[], [0, 1, 2, 3, 4], [5, 6], [], [7, 8, 9], []])
        a[5:] = [999]
        self.assertEqual(a.tolist(), [0, 1, 2, 3, 4, 999, 999, 999, 999, 999])

        a = ChunkedArray([[], [0, 1, 2, 3, 4], [5, 6], [], [7, 8, 9], []])
        a[:5] = [999]
        self.assertEqual(a.tolist(), [999, 999, 999, 999, 999, 5, 6, 7, 8, 9])

        a = ChunkedArray([[], [0, 1, 2, 3, 4], [5, 6], [], [7, 8, 9], []])
        a[3:8] = [999]
        self.assertEqual(a.tolist(), [0, 1, 2, 999, 999, 999, 999, 999, 8, 9])

        a = ChunkedArray([[], [0, 1, 2, 3, 4], [5, 6], [], [7, 8, 9], []])
        a[3:8:2] = [999]
        self.assertEqual(a.tolist(), [0, 1, 2, 999, 4, 999, 6, 999, 8, 9])

        a = ChunkedArray([[], [0, 1, 2, 3, 4], [5, 6], [], [7, 8, 9], []])
        a[::3] = [999]
        self.assertEqual(a.tolist(), [999, 1, 2, 999, 4, 5, 999, 7, 8, 999])

        a = ChunkedArray([[], [0, 1, 2, 3, 4], [5, 6], [], [7, 8, 9], []])
        a[1::3] = [999]
        self.assertEqual(a.tolist(), [0, 999, 2, 3, 999, 5, 6, 999, 8, 9])

        a = ChunkedArray([[], [0, 1, 2, 3, 4], [5, 6], [], [7, 8, 9], []])
        a[7:2:-1] = [999]
        self.assertEqual(a.tolist(), [0, 1, 2, 999, 999, 999, 999, 999, 8, 9])

        a = ChunkedArray([[], [0, 1, 2, 3, 4], [5, 6], [], [7, 8, 9], []])
        a[7:2:-2] = [999]
        self.assertEqual(a.tolist(), [0, 1, 2, 999, 4, 999, 6, 999, 8, 9])

        a = ChunkedArray([[], [0, 1, 2, 3, 4], [5, 6], [], [7, 8, 9], []])
        a[::4] = [999]
        self.assertEqual(a.tolist(), [999, 1, 2, 3, 999, 5, 6, 7, 999, 9])

        a = ChunkedArray([[], [0, 1, 2, 3, 4], [5, 6], [], [7, 8, 9], []])
        a[1::4] = [999]
        self.assertEqual(a.tolist(), [0, 999, 2, 3, 4, 999, 6, 7, 8, 999])

        a = ChunkedArray([[], [0, 1, 2, 3, 4], [5, 6], [], [7, 8, 9], []])
        a[numpy.empty(0, dtype=int)] = [999]
        self.assertEqual(a.tolist(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

        a = ChunkedArray([[], [0, 1, 2, 3, 4], [5, 6], [], [7, 8, 9], []])
        a[[5]] = [999]
        self.assertEqual(a.tolist(), [0, 1, 2, 3, 4, 999, 6, 7, 8, 9])

        a = ChunkedArray([[], [0, 1, 2, 3, 4], [5, 6], [], [7, 8, 9], []])
        a[[8, 6, 3, 4, 5]] = [999]
        self.assertEqual(a.tolist(), [0, 1, 2, 999, 999, 999, 999, 7, 999, 9])

        a = ChunkedArray([[], [0, 1, 2, 3, 4], [5, 6], [], [7, 8, 9], []])
        a[[True, False, True, False, True, False, True, False, True, False]] = [999]
        self.assertEqual(a.tolist(), [999, 1, 999, 3, 999, 5, 999, 7, 999, 9])

        a = ChunkedArray([[], [0, 1, 2, 3, 4], [5, 6], [], [7, 8, 9], []])
        a[[True, False, False, False, True, False, False, False, False, False]] = [999]
        self.assertEqual(a.tolist(), [999, 1, 2, 3, 999, 5, 6, 7, 8, 9])

    def test_chunked_set_sequence(self):
        a = ChunkedArray([[], [0, 1, 2, 3, 4], [5, 6], [], [7, 8, 9], []])
        a[:] = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
        self.assertEqual(a.tolist(), [100, 101, 102, 103, 104, 105, 106, 107, 108, 109])

        a = ChunkedArray([[], [0, 1, 2, 3, 4], [5, 6], [], [7, 8, 9], []])
        a[5:] = [101, 102, 103, 104, 105]
        self.assertEqual(a.tolist(), [0, 1, 2, 3, 4, 101, 102, 103, 104, 105])

        a = ChunkedArray([[], [0, 1, 2, 3, 4], [5, 6], [], [7, 8, 9], []])
        a[:5] = [101, 102, 103, 104, 105]
        self.assertEqual(a.tolist(), [101, 102, 103, 104, 105, 5, 6, 7, 8, 9])

        a = ChunkedArray([[], [0, 1, 2, 3, 4], [5, 6], [], [7, 8, 9], []])
        a[3:8] = [101, 102, 103, 104, 105]
        self.assertEqual(a.tolist(), [0, 1, 2, 101, 102, 103, 104, 105, 8, 9])

        a = ChunkedArray([[], [0, 1, 2, 3, 4], [5, 6], [], [7, 8, 9], []])
        a[3:8:2] = [101, 102, 103]
        self.assertEqual(a.tolist(), [0, 1, 2, 101, 4, 102, 6, 103, 8, 9])

        a = ChunkedArray([[], [0, 1, 2, 3, 4], [5, 6], [], [7, 8, 9], []])
        a[::3] = [101, 102, 103, 104]
        self.assertEqual(a.tolist(), [101, 1, 2, 102, 4, 5, 103, 7, 8, 104])

        a = ChunkedArray([[], [0, 1, 2, 3, 4], [5, 6], [], [7, 8, 9], []])
        a[1::3] = [101, 102, 103]
        self.assertEqual(a.tolist(), [0, 101, 2, 3, 102, 5, 6, 103, 8, 9])

        a = ChunkedArray([[], [0, 1, 2, 3, 4], [5, 6], [], [7, 8, 9], []])
        a[7:2:-1] = [101, 102, 103, 104, 105]
        self.assertEqual(a.tolist(), [0, 1, 2, 105, 104, 103, 102, 101, 8, 9])

        a = ChunkedArray([[], [0, 1, 2, 3, 4], [5, 6], [], [7, 8, 9], []])
        a[7:2:-2] = [101, 102, 103]
        self.assertEqual(a.tolist(), [0, 1, 2, 103, 4, 102, 6, 101, 8, 9])

        a = ChunkedArray([[], [0, 1, 2, 3, 4], [5, 6], [], [7, 8, 9], []])
        a[::4] = [101, 102, 103]
        self.assertEqual(a.tolist(), [101, 1, 2, 3, 102, 5, 6, 7, 103, 9])

        a = ChunkedArray([[], [0, 1, 2, 3, 4], [5, 6], [], [7, 8, 9], []])
        a[1::4] = [101, 102, 103]
        self.assertEqual(a.tolist(), [0, 101, 2, 3, 4, 102, 6, 7, 8, 103])

        a = ChunkedArray([[], [0, 1, 2, 3, 4], [5, 6], [], [7, 8, 9], []])
        def quickie():
            a[numpy.empty(0, dtype=int)] = [101, 102, 103]
        self.assertRaises(ValueError, quickie)

        a = ChunkedArray([[], [0, 1, 2, 3, 4], [5, 6], [], [7, 8, 9], []])
        def quickie():
            a[[5]] = [101, 102, 103]
        self.assertRaises(ValueError, quickie)

        a = ChunkedArray([[], [0, 1, 2, 3, 4], [5, 6], [], [7, 8, 9], []])
        a[[8, 6, 3, 4, 5]] = [101, 102, 103, 104, 105]
        self.assertEqual(a.tolist(), [0, 1, 2, 103, 104, 105, 102, 7, 101, 9])

        a = ChunkedArray([[], [0, 1, 2, 3, 4], [5, 6], [], [7, 8, 9], []])
        a[[True, False, True, False, True, False, True, False, True, False]] = [101, 102, 103, 104, 105]
        self.assertEqual(a.tolist(), [101, 1, 102, 3, 103, 5, 104, 7, 105, 9])

        a = ChunkedArray([[], [0, 1, 2, 3, 4], [5, 6], [], [7, 8, 9], []])
        a[[True, False, False, False, True, False, False, False, False, False]] = [101, 102]
        self.assertEqual(a.tolist(), [101, 1, 2, 3, 102, 5, 6, 7, 8, 9])

    def test_partitioned(self):
        a = PartitionedArray([0, 3, 3, 5], [[0, 1, 2], [], [3, 4]])
        self.assertEqual(a.tolist(), [0, 1, 2, 3, 4])

        a = ChunkedArray([[0, 1, 2], [], [3, 4]]).topartitioned()
        self.assertEqual(a.tolist(), [0, 1, 2, 3, 4])
        self.assertEqual(a.offsets.tolist(), [0, 3, 3, 5])

    def test_partitioned_get(self):
        a = ChunkedArray([[], [0, 1, 2, 3, 4], [5, 6], [], [7, 8, 9], []]).topartitioned()
        self.assertEqual([a[-i] for i in range(1, 11)], [9, 8, 7, 6, 5, 4, 3, 2, 1, 0])
        self.assertEqual([a[-i : -i - 1 : -1].tolist() for i in range(1, 11)], [[9], [8], [7], [6], [5], [4], [3], [2], [1], [0]])
        self.assertEqual([a[-i : -i - 2 : -1].tolist() for i in range(1, 10)], [[9, 8], [8, 7], [7, 6], [6, 5], [5, 4], [4, 3], [3, 2], [2, 1], [1, 0]])
        self.assertEqual([a[-i : -i + 1].tolist() for i in range(2, 11)], [[8], [7], [6], [5], [4], [3], [2], [1], [0]])
        self.assertEqual([a[-i : -i + 2].tolist() for i in range(3, 11)], [[7, 8], [6, 7], [5, 6], [4, 5], [3, 4], [2, 3], [1, 2], [0, 1]])
        self.assertEqual(a[[-2, -4, 7, 5, 3, -6, 5]].tolist(), [8, 6, 7, 5, 3, 4, 5])
        self.assertEqual(a[[-2, -4, 7, 5, 3, -6, -5]].tolist(), [8, 6, 7, 5, 3, 4, 5])

    def test_partitioned_set_const(self):
        a = ChunkedArray([[], [0, 1, 2, 3, 4], [5, 6], [], [7, 8, 9], []]).topartitioned()
        a[-4] = 999
        self.assertEqual(a.tolist(), [0, 1, 2, 3, 4, 5, 999, 7, 8, 9])

        a = ChunkedArray([[], [0, 1, 2, 3, 4], [5, 6], [], [7, 8, 9], []]).topartitioned()
        a[-4:-6:-1] = 999
        self.assertEqual(a.tolist(), [0, 1, 2, 3, 4, 999, 999, 7, 8, 9])

        a = ChunkedArray([[], [0, 1, 2, 3, 4], [5, 6], [], [7, 8, 9], []]).topartitioned()
        a[-4::-2] = 999
        self.assertEqual(a.tolist(), [999, 1, 999, 3, 999, 5, 999, 7, 8, 9])

        a = ChunkedArray([[], [0, 1, 2, 3, 4], [5, 6], [], [7, 8, 9], []]).topartitioned()
        a[:-6:-2] = 999
        self.assertEqual(a.tolist(), [0, 1, 2, 3, 4, 999, 6, 999, 8, 999])

        a = ChunkedArray([[], [0, 1, 2, 3, 4], [5, 6], [], [7, 8, 9], []]).topartitioned()
        a[[-2, -4, 7, 5, 3, -6, 5]] = 999
        self.assertEqual(a.tolist(), [0, 1, 2, 999, 999, 999, 999, 999, 999, 9])

        a = ChunkedArray([[], [0, 1, 2, 3, 4], [5, 6], [], [7, 8, 9], []]).topartitioned()
        a[[-2, -4, 7, 5, 3, -6, -5]] = 999
        self.assertEqual(a.tolist(), [0, 1, 2, 999, 999, 999, 999, 999, 999, 9])

    def test_partitioned_set_singleton(self):
        a = ChunkedArray([[], [0, 1, 2, 3, 4], [5, 6], [], [7, 8, 9], []]).topartitioned()
        a[-4:-6:-1] = [999]
        self.assertEqual(a.tolist(), [0, 1, 2, 3, 4, 999, 999, 7, 8, 9])

        a = ChunkedArray([[], [0, 1, 2, 3, 4], [5, 6], [], [7, 8, 9], []]).topartitioned()
        a[-4::-2] = [999]
        self.assertEqual(a.tolist(), [999, 1, 999, 3, 999, 5, 999, 7, 8, 9])

        a = ChunkedArray([[], [0, 1, 2, 3, 4], [5, 6], [], [7, 8, 9], []]).topartitioned()
        a[:-6:-2] = [999]
        self.assertEqual(a.tolist(), [0, 1, 2, 3, 4, 999, 6, 999, 8, 999])

        a = ChunkedArray([[], [0, 1, 2, 3, 4], [5, 6], [], [7, 8, 9], []]).topartitioned()
        a[[-2, -4, 7, 5, 3, -6, 5]] = [999]
        self.assertEqual(a.tolist(), [0, 1, 2, 999, 999, 999, 999, 999, 999, 9])

        a = ChunkedArray([[], [0, 1, 2, 3, 4], [5, 6], [], [7, 8, 9], []]).topartitioned()
        a[[-2, -4, 7, 5, 3, -6, -5]] = [999]
        self.assertEqual(a.tolist(), [0, 1, 2, 999, 999, 999, 999, 999, 999, 9])

    def test_partitioned_set_sequence(self):
        a = ChunkedArray([[], [0, 1, 2, 3, 4], [5, 6], [], [7, 8, 9], []]).topartitioned()
        a[-4:-6:-1] = [101, 102]
        self.assertEqual(a.tolist(), [0, 1, 2, 3, 4, 102, 101, 7, 8, 9])

        a = ChunkedArray([[], [0, 1, 2, 3, 4], [5, 6], [], [7, 8, 9], []]).topartitioned()
        a[-4::-2] = [101, 102, 103, 104]
        self.assertEqual(a.tolist(), [104, 1, 103, 3, 102, 5, 101, 7, 8, 9])

        a = ChunkedArray([[], [0, 1, 2, 3, 4], [5, 6], [], [7, 8, 9], []]).topartitioned()
        a[:-6:-2] = [101, 102, 103]
        self.assertEqual(a.tolist(), [0, 1, 2, 3, 4, 103, 6, 102, 8, 101])

        a = ChunkedArray([[], [0, 1, 2, 3, 4], [5, 6], [], [7, 8, 9], []]).topartitioned()
        a[[-2, -4, 7, 5, 3, -6]] = [101, 102, 103, 104, 105, 106]
        self.assertEqual(a.tolist(), [0, 1, 2, 105, 106, 104, 102, 103, 101, 9])
