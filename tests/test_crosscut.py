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

