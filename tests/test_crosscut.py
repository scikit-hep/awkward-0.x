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

from awkward import *
from awkward.type import *

class Test(unittest.TestCase):
    def runTest(self):
        pass

    def test_crosscut_asdtype(self):
        a = JaggedArray([0, 3, 3, 5], [3, 3, 5, 10], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        assert a.tolist() == [[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5, 6.6, 7.7, 8.8, 9.9]]
        assert a.astype(int).tolist() == [[0, 1, 2], [], [3, 4], [5, 6, 7, 8, 9]]

        a = ChunkedArray([[], [0.0, 1.1, 2.2, 3.3, 4.4], [5.5, 6.6], [], [7.7, 8.8, 9.9], []])
        assert a.tolist() == [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]
        assert a.astype(int).tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        a = AppendableArray(3, numpy.float64)
        a.append(0.0)
        a.append(1.1)
        a.append(2.2)
        a.append(3.3)
        assert a.offsets.tolist() == [0, 3, 4]
        assert a.tolist() == [0.0, 1.1, 2.2, 3.3]
        assert a.astype(int).tolist() == [0, 1, 2, 3]

        a = IndexedArray([3, 2, 4, 2, 2, 4, 0], [0.0, 1.1, 2.2, 3.3, 4.4])
        assert a.tolist() == [3.3, 2.2, 4.4, 2.2, 2.2, 4.4, 0.0]
        assert a.astype(int).tolist() == [3, 2, 4, 2, 2, 4, 0]

        a = SparseArray(10, [1, 3, 5, 7, 9], [100.0, 101.1, 102.2, 103.3, 104.4])
        assert a.tolist() == [0.0, 100.0, 0.0, 101.1, 0.0, 102.2, 0.0, 103.3, 0.0, 104.4]
        assert a.astype(int).tolist() == [0, 100, 0, 101, 0, 102, 0, 103, 0, 104]

        a = MaskedArray([True, False, True, False, True, False, True, False, True, False], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], maskedwhen=True)
        assert a.tolist() == [None, 1.1, None, 3.3, None, 5.5, None, 7.7, None, 9.9]
        assert a.astype(int).tolist() == [None, 1, None, 3, None, 5, None, 7, None, 9]

        a = BitMaskedArray.fromboolmask([True, False, True, False, True, False, True, False, True, False], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], maskedwhen=True, lsborder=True)
        assert a.tolist() == [None, 1.1, None, 3.3, None, 5.5, None, 7.7, None, 9.9]
        assert a.astype(int).tolist() == [None, 1, None, 3, None, 5, None, 7, None, 9]

        a = IndexedMaskedArray([-1, 0, -1, 1, -1, 2, -1, 4, -1, 3], [0.0, 1.1, 2.2, 3.3, 4.4])
        assert a.tolist() == [None, 0.0, None, 1.1, None, 2.2, None, 4.4, None, 3.3]
        assert a.astype(int).tolist() == [None, 0, None, 1, None, 2, None, 4, None, 3]

        class Point(object):
            def __init__(self, array):
                self.x, self.y, self.z = array
            def __repr__(self):
                return "<Point {0} {1} {2}>".format(self.x, self.y, self.z)
            def __eq__(self, other):
                return isinstance(other, Point) and self.x == other.x and self.y == other.y and self.z == other.z

        a = ObjectArray([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6], [7.7, 8.8, 9.9]], Point)
        assert a.tolist() == [Point([1.1, 2.2, 3.3]), Point([4.4, 5.5, 6.6]), Point([7.7, 8.8, 9.9])]
        assert a.astype(int).tolist() == [Point([1, 2, 3]), Point([4, 5, 6]), Point([7, 8, 9])]

        a = Table([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        assert a.tolist() == [{"0": 0, "1": 0.0}, {"0": 1, "1": 1.1}, {"0": 2, "1": 2.2}, {"0": 3, "1": 3.3}, {"0": 4, "1": 4.4}, {"0": 5, "1": 5.5}, {"0": 6, "1": 6.6}, {"0": 7, "1": 7.7}, {"0": 8, "1": 8.8}, {"0": 9, "1": 9.9}]
        assert a.astype(int).tolist() == [{"0": 0, "1": 0}, {"0": 1, "1": 1}, {"0": 2, "1": 2}, {"0": 3, "1": 3}, {"0": 4, "1": 4}, {"0": 5, "1": 5}, {"0": 6, "1": 6}, {"0": 7, "1": 7}, {"0": 8, "1": 8}, {"0": 9, "1": 9}]

        a = UnionArray([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [[0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], [0, 100, 200, 300, 400, 500, 600, 700, 800, 900]])
        assert a.tolist() == [0.0, 100, 2.2, 300, 4.4, 500, 6.6, 700, 8.8, 900]
        assert a.astype(int).tolist() == [0, 100, 2, 300, 4, 500, 6, 700, 8, 900]

        a = VirtualArray(lambda: [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        assert a.astype(int).tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        a = VirtualArray(lambda: [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        a.materialize()
        assert a.astype(int).tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    def test_crosscut_reduce(self):
        a = JaggedArray([0, 3, 3, 5], [3, 3, 5, 10], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        assert a.sum().tolist() == [3.3000000000000003, 0.0, 7.7, 38.5]

        a = ChunkedArray([[], [0.0, 1.1, 2.2, 3.3, 4.4], [5.5, 6.6], [], [7.7, 8.8, 9.9], []])
        assert a.sum() == 49.5
        a = ChunkedArray([JaggedArray.fromiter([[0.0, 1.1, 2.2], [], [3.3, 4.4]]), JaggedArray.fromiter([]), JaggedArray.fromiter([[5.5, 6.6]]), JaggedArray.fromiter([[7.7, 8.8], [9.9]])])
        assert a.sum().tolist() == [3.3000000000000003, 0.0, 7.7, 12.1, 16.5, 9.9]
        a = JaggedArray.fromcounts([3, 0, 3, 1, 2, 1], ChunkedArray([[], [0.0, 1.1, 2.2, 3.3, 4.4], [5.5, 6.6], [], [7.7, 8.8, 9.9], []]))
        assert a.sum().tolist() == [3.3000000000000003, 0.0, 13.2, 6.6, 16.5, 9.9]
        a = ChunkedArray([Table.named("tuple", [0.0, 1.1, 2.2], [0, 100, 200]), Table.named("tuple", []), Table.named("tuple", [3.3, 4.4, 5.5], [300, 400, 500])])
        assert a.sum().tolist() == {"0": 16.5, "1": 1500}

        a = AppendableArray(3, numpy.float64)
        a.append(0.0)
        a.append(1.1)
        a.append(2.2)
        a.append(3.3)
        assert a.sum() == 6.6
        a = AppendableArray(3, numpy.float64)
        a.append(0.0)
        a.append(1.1)
        a.append(2.2)
        a.append(3.3)
        a = JaggedArray.fromcounts([2, 0, 2], a)
        assert a.sum().tolist() == [1.1, 0.0, 5.5]

        a = IndexedArray([3, 2, 4, 2, 2, 4, 0], [0.0, 1.1, 2.2, 3.3, 4.4])
        assert a.sum() == 18.700000000000003
        a = JaggedArray.fromcounts([3, 0, 2], IndexedArray([3, 2, 4, 2, 2, 4, 0], [0.0, 1.1, 2.2, 3.3, 4.4]))
        assert a.sum().tolist() == [9.9, 0.0, 8.8]
        a = IndexedArray([3, 2, 4, 2, 2, 4, 0], JaggedArray.fromcounts([3, 0, 2, 1, 4], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
        assert a.sum().tolist() == [5.5, 7.7, 33.0, 7.7, 7.7, 33.0, 3.3000000000000003]
        a = IndexedArray([3, 2, 4, 2, 2, 4, 0], Table.named("tuple", [0.0, 1.1, 2.2, 3.3, 4.4], [0, 100, 200, 300, 400]))
        assert a.sum().tolist() == {"0": 18.700000000000003, "1": 1700}

        a = SparseArray(10, [1, 3, 5, 7, 9], [100.0, 101.1, 102.2, 103.3, 104.4])
        assert a.sum() == 511.0

        a = MaskedArray([True, False, True, False, True, False, True, False, True, False], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], maskedwhen=True)
        assert a.sum() == 27.5
        a = JaggedArray.fromcounts([3, 0, 2, 1, 1, 3], MaskedArray([True, False, True, False, True, False, True, False, True, False], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], maskedwhen=True))
        assert a.sum().tolist() == [1.1, 0.0, 3.3, 5.5, 0.0, 17.6]
        a = MaskedArray([False, False, False, True, True, False], JaggedArray.fromcounts([3, 0, 2, 1, 0, 4], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]), maskedwhen=True)
        assert a.sum().tolist() == [3.3000000000000003, 0.0, 7.7, None, None, 33.0]
        a = MaskedArray([True, False, True, False, True, False, True, False, True, False], Table.named("tuple", [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], [0, 100, 200, 300, 400, 500, 600, 700, 800, 900]), maskedwhen=True)
        assert a.sum().tolist() == {"0": 27.5, "1": 2500}

        a = BitMaskedArray.fromboolmask([True, False, True, False, True, False, True, False, True, False], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], maskedwhen=True, lsborder=True)
        assert a.sum() == 27.5
        a = JaggedArray.fromcounts([3, 0, 2, 1, 1, 3], BitMaskedArray.fromboolmask([True, False, True, False, True, False, True, False, True, False], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], maskedwhen=True, lsborder=True))
        assert a.sum().tolist() == [1.1, 0.0, 3.3, 5.5, 0.0, 17.6]
        a = BitMaskedArray.fromboolmask([False, False, False, True, True, False], JaggedArray.fromcounts([3, 0, 2, 1, 0, 4], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]), maskedwhen=True, lsborder=True)
        assert a.sum().tolist() == [3.3000000000000003, 0.0, 7.7, None, None, 33.0]
        a = BitMaskedArray.fromboolmask([True, False, True, False, True, False, True, False, True, False], Table.named("tuple", [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], [0, 100, 200, 300, 400, 500, 600, 700, 800, 900]), maskedwhen=True, lsborder=True)
        assert a.sum().tolist() == {"0": 27.5, "1": 2500}

        a = IndexedMaskedArray([-1, 1, -1, 3, -1, 5, -1, 7, -1, 8], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 9.9])
        assert a.sum() == 27.5
        a = JaggedArray.fromcounts([3, 0, 2, 1, 1, 3], IndexedMaskedArray([-1, 1, -1, 3, -1, 5, -1, 7, -1, 9], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
        assert a.sum().tolist() == [1.1, 0.0, 3.3, 5.5, 0.0, 17.6]
        a = IndexedMaskedArray([0, 1, 2, -1, -1, 5], JaggedArray.fromcounts([3, 0, 2, 1, 0, 4], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
        assert a.sum().tolist() == [3.3000000000000003, 0.0, 7.7, None, None, 33.0]
        a = IndexedMaskedArray([-1, 1, -1, 3, -1, 5, -1, 7, -1, 9], Table.named("tuple", [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], [0, 100, 200, 300, 400, 500, 600, 700, 800, 900]))
        assert a.sum().tolist() == {"0": 27.5, "1": 2500}

        a = Table.named("tuple", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        assert a.sum().tolist() == {"0": 45, "1": 49.50000000000001}
        a = JaggedArray.fromcounts([3, 0, 2, 1, 4], Table.named("tuple", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
        assert a.sum().tolist() == [(3, 3.3000000000000003), (0, 0.0), (7, 7.7), (5, 5.5), (30, 33.0)]
        a = Table.named("tuple", JaggedArray.fromcounts([3, 0, 2, 1, 4], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        assert a.sum().tolist() == {"0": [3, 0, 7, 5, 30], "1": 11.0}

        a = UnionArray([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [[0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], [0, 100, 200, 300, 400, 500, 600, 700, 800, 900]])
        assert a.sum() == 2522.0

        a = VirtualArray(lambda: [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        assert a.sum() == 49.50000000000001

        a = VirtualArray(lambda: [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        a.materialize()
        assert a.sum() == 49.50000000000001
