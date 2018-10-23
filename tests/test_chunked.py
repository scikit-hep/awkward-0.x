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

    def test_chunked_allslices(self):
        chunked = ChunkedArray([[], [0.0, 1.1, 2.2, 3.3, 4.4], [5.5, 6.6], [], [7.7], [8.8, 9.9], []])
        regular = numpy.concatenate(chunked.chunks).tolist()
        for start in [None] + list(range(-12, 12 + 1)):
            for stop in [None] + list(range(-12, 12 + 1)):
                for step in [None, 1, 2, 3, 4, 5, 9, 10, 11, -1, -2, -3, -4, -5, -9, -10, -11]:
                    # print(start, stop, step)
                    assert numpy.concatenate(chunked[start:stop:step].chunks).tolist() == regular[start:stop:step]

    def test_chunked_iteration(self):
        a = ChunkedArray([[], [0, 1, 2, 3, 4], [5, 6], [], [7, 8, 9], []])
        assert a.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        a = ChunkedArray([[0, 1, 2, 3, 4], [5, 6], [], [7, 8, 9], []])
        assert a.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        a = ChunkedArray([[], [0, 1, 2, 3, 4], [5, 6], [], [7, 8, 9]])
        assert a.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        a = ChunkedArray([[]])
        assert a.tolist() == []

        a = ChunkedArray([])
        assert a.tolist() == []

    def test_chunked_dtype(self):
        a = ChunkedArray([[], [0, 1, 2, 3, 4], [5, 6], [], [7, 8, 9], []])
        assert a.dtype == numpy.dtype(int)

        a = ChunkedArray([[0, 1, 2, 3, 4], [5, 6], [], [7, 8, 9], []])
        assert a.dtype == numpy.dtype(int)

        a = ChunkedArray([])
        assert a.dtype == numpy.dtype(float)

    def test_chunked_get(self):
        a = ChunkedArray([[], [0, 1, 2, 3, 4], [5, 6], [], [7, 8, 9], []])
        assert [a[i] for i in range(10)] == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        assert [a[i : i + 1].tolist() for i in range(10)] == [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]]
        assert [a[i : i + 2].tolist() for i in range(10 - 1)] == [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9]]
        assert [a[i : i + 3].tolist() for i in range(10 - 2)] == [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8], [7, 8, 9]]
        assert [a[i : i + 4].tolist() for i in range(10 - 3)] == [[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7], [5, 6, 7, 8], [6, 7, 8, 9]]
        assert [a[i : i + 4 : 2].tolist() for i in range(10 - 3)] == [[0, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7], [6, 8]]
        assert [a[i : i + 4 : 3].tolist() for i in range(10 - 3)] == [[0, 3], [1, 4], [2, 5], [3, 6], [4, 7], [5, 8], [6, 9]]
        assert [a[i + 4 : i : -1].tolist() for i in range(10 - 4)] == [[4, 3, 2, 1], [5, 4, 3, 2], [6, 5, 4, 3], [7, 6, 5, 4], [8, 7, 6, 5], [9, 8, 7, 6]]
        assert [a[i + 4 : i : -2].tolist() for i in range(10 - 4)] == [[4, 2], [5, 3], [6, 4], [7, 5], [8, 6], [9, 7]]
        assert [a[i + 4 : i : -3].tolist() for i in range(10 - 4)] == [[4, 1], [5, 2], [6, 3], [7, 4], [8, 5], [9, 6]]

        assert a[4:].tolist() == [4, 5, 6, 7, 8, 9]
        assert a[5:].tolist() == [5, 6, 7, 8, 9]
        assert a[6:].tolist() == [6, 7, 8, 9]
        assert a[7:].tolist() == [7, 8, 9]
        assert a[8:].tolist() == [8, 9]
        assert a[:4].tolist() == [0, 1, 2, 3]
        assert a[:5].tolist() == [0, 1, 2, 3, 4]
        assert a[:6].tolist() == [0, 1, 2, 3, 4, 5]
        assert a[:7].tolist() == [0, 1, 2, 3, 4, 5, 6]
        assert a[:8].tolist() == [0, 1, 2, 3, 4, 5, 6, 7]

        assert a[4::2].tolist() == [4, 6, 8]
        assert a[5::2].tolist() == [5, 7, 9]
        assert a[6::2].tolist() == [6, 8]
        assert a[7::2].tolist() == [7, 9]
        assert a[8::2].tolist() == [8]
        assert a[:4:2].tolist() == [0, 2]
        assert a[:5:2].tolist() == [0, 2, 4]
        assert a[:6:2].tolist() == [0, 2, 4]
        assert a[:7:2].tolist() == [0, 2, 4, 6]
        assert a[:8:2].tolist() == [0, 2, 4, 6]

        assert a[4::-1].tolist() == [4, 3, 2, 1, 0]
        assert a[5::-1].tolist() == [5, 4, 3, 2, 1, 0]
        assert a[6::-1].tolist() == [6, 5, 4, 3, 2, 1, 0]
        assert a[7::-1].tolist() == [7, 6, 5, 4, 3, 2, 1, 0]
        assert a[8::-1].tolist() == [8, 7, 6, 5, 4, 3, 2, 1, 0]
        assert a[:4:-1].tolist() == [9, 8, 7, 6, 5]
        assert a[:5:-1].tolist() == [9, 8, 7, 6]
        assert a[:6:-1].tolist() == [9, 8, 7]
        assert a[:7:-1].tolist() == [9, 8]
        assert a[:8:-1].tolist() == [9]

        assert a[4::-2].tolist() == [4, 2, 0]
        assert a[5::-2].tolist() == [5, 3, 1]
        assert a[6::-2].tolist() == [6, 4, 2, 0]
        assert a[7::-2].tolist() == [7, 5, 3, 1]
        assert a[8::-2].tolist() == [8, 6, 4, 2, 0]
        assert a[:4:-2].tolist() == [9, 7, 5]
        assert a[:5:-2].tolist() == [9, 7]
        assert a[:6:-2].tolist() == [9, 7]
        assert a[:7:-2].tolist() == [9]
        assert a[:8:-2].tolist() == [9]

        assert a[[8, 6, 4, 5, 0]].tolist() == [8, 6, 4, 5, 0]
        assert a[[6, 4, 5, 0]].tolist() == [6, 4, 5, 0]
        assert a[[5, 6, 4, 5, 5, 5, 0]].tolist() == [5, 6, 4, 5, 5, 5, 0]
        self.assertRaises(IndexError, lambda: a[[8, 6, 4, 5, 0, 99]])

        assert a[[True, False, True, False, True, False, True, False, True, False]].tolist() == [0, 2, 4, 6, 8]
        assert a[[False, False, False, False, False, False, False, False, False, False]].tolist() == []
        assert a[[True, True, True, True, True, True, True, True, True, True]].tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.assertRaises(IndexError, lambda: a[[True, True, True, True, True, True, True, True, True, True, True]])
        self.assertRaises(IndexError, lambda: a[[True, True, True, True, True, True, True, True, True]])

    def test_chunked_get2d(self):
        a = ChunkedArray([[], [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]], [[5, 5], [6, 6]], [], [[7, 7], [8, 8], [9, 9]], []])
        assert [a[i].tolist() for i in range(10)] == [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9]]
        assert a[4:].tolist() == [[4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9]]
        assert a[[8, 6, 4, 5, 0]].tolist() == [[8, 8], [6, 6], [4, 4], [5, 5], [0, 0]]
        assert a[[True, False, True, False, True, False, True, False, True, False]].tolist() == [[0, 0], [2, 2], [4, 4], [6, 6], [8, 8]]

        a = ChunkedArray([[], [[0.0, 0.0], [1.0, 1.1], [2.0, 2.2], [3.0, 3.3], [4.0, 4.4]], [[5.0, 5.5], [6.0, 6.6]], [], [[7.0, 7.7], [8.0, 8.8], [9.0, 9.9]], []])
        assert [a[i, 0].tolist() for i in range(10)] == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        assert [a[i, 1].tolist() for i in range(10)] == [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]
        assert a[4:, 0].tolist() == [4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        assert a[4:, 1].tolist() == [4.4, 5.5, 6.6, 7.7, 8.8, 9.9]
        assert a[[8, 6, 4, 5, 0], 0].tolist() == [8.0, 6.0, 4.0, 5.0, 0.0]
        assert a[[8, 6, 4, 5, 0], 1].tolist() == [8.8, 6.6, 4.4, 5.5, 0.0]
        assert a[[True, False, True, False, True, False, True, False, True, False], 0].tolist() == [0.0, 2.0, 4.0, 6.0, 8.0]
        assert a[[True, False, True, False, True, False, True, False, True, False], 1].tolist() == [0.0, 2.2, 4.4, 6.6, 8.8]

    def test_appendable_append(self):
        a = AppendableArray(3, numpy.float64)
        assert a.tolist() == []
        assert len(a.chunks) == 0
        assert a.offsets.tolist() == [0]

        a.append(0.0)
        assert a.tolist() == [0.0]
        assert len(a.chunks) == 1
        assert a.offsets.tolist() == [0, 1]

        a.append(1.1)
        assert a.tolist() == [0.0, 1.1]
        assert len(a.chunks) == 1
        assert a.offsets.tolist() == [0, 2]

        a.append(2.2)
        assert a.tolist() == [0.0, 1.1, 2.2]
        assert len(a.chunks) == 1
        assert a.offsets.tolist() == [0, 3]

        a.append(3.3)
        assert a.tolist() == [0.0, 1.1, 2.2, 3.3]
        assert len(a.chunks) == 2
        assert a.offsets.tolist() == [0, 3, 4]

        a.append(4.4)
        assert a.tolist() == [0.0, 1.1, 2.2, 3.3, 4.4]
        assert len(a.chunks) == 2
        assert a.offsets.tolist() == [0, 3, 5]

        a.append(5.5)
        assert a.tolist() == [0.0, 1.1, 2.2, 3.3, 4.4, 5.5]
        assert len(a.chunks) == 2
        assert a.offsets.tolist() == [0, 3, 6]

        a.append(6.6)
        assert a.tolist() == [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6]
        assert len(a.chunks) == 3
        assert a.offsets.tolist() == [0, 3, 6, 7]

        a.append(7.7)
        assert a.tolist() == [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7]
        assert len(a.chunks) == 3
        assert a.offsets.tolist() == [0, 3, 6, 8]

        a.append(8.8)
        assert a.tolist() == [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8]
        assert len(a.chunks) == 3
        assert a.offsets.tolist() == [0, 3, 6, 9]

        a.append(9.9)
        assert a.tolist() == [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]
        assert [a[i] for i in range(len(a))] == [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]
        assert len(a.chunks) == 4
        assert a.offsets.tolist() == [0, 3, 6, 9, 10]

    def test_appendable_extend(self):
        a = AppendableArray(3, numpy.float64)
        assert a.tolist() == []
        assert len(a.chunks) == 0
        assert a.offsets.tolist() == [0]

        a.extend([0.0, 1.1])
        assert a.tolist() == [0.0, 1.1]
        assert len(a.chunks) == 1
        assert a.offsets.tolist() == [0, 2]

        a.extend([2.2, 3.3])
        assert a.tolist() == [0.0, 1.1, 2.2, 3.3]
        assert len(a.chunks) == 2
        assert a.offsets.tolist() == [0, 3, 4]

        a.extend([4.4, 5.5])
        assert a.tolist() == [0.0, 1.1, 2.2, 3.3, 4.4, 5.5]
        assert len(a.chunks) == 2
        assert a.offsets.tolist() == [0, 3, 6]

        a.extend([6.6, 7.7])
        assert a.tolist() == [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7]
        assert len(a.chunks) == 3
        assert a.offsets.tolist() == [0, 3, 6, 8]

        a.extend([8.8, 9.9])
        assert a.tolist() == [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]
        assert [a[i] for i in range(len(a))] == [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]
        assert len(a.chunks) == 4
        assert a.offsets.tolist() == [0, 3, 6, 9, 10]
