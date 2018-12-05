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

import pickle
import struct
import unittest
import zlib

import numpy

from awkward import *
from awkward.persist import *
from awkward.type import *

def makearray():
    return range(10)

class FloatPoint(object):
    def __init__(self, array):
        self.x, self.y, self.z = array
    def __repr__(self):
        return "<FloatPoint {0} {1} {2}>".format(self.x, self.y, self.z)
    def __eq__(self, other):
        return isinstance(other, FloatPoint) and self.x == other.x and self.y == other.y and self.z == other.z

class BytesPoint(object):
    def __init__(self, bytes):
        self.x, self.y, self.z = struct.unpack("ddd", bytes)
    def __repr__(self):
        return "<BytesPoint {0} {1} {2}>".format(self.x, self.y, self.z)
    def __eq__(self, other):
        return isinstance(other, BytesPoint) and self.x == other.x and self.y == other.y and self.z == other.z

class Test(unittest.TestCase):
    def runTest(self):
        pass

    def test_pickle(self):
        a = awkward.JaggedArray.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
        b = pickle.loads(pickle.dumps(a))
        assert a.tolist() == b.tolist()

    def test_uncompressed_numpy(self):
        storage = {}
        a = numpy.arange(100, dtype=">u2").reshape(-1, 5)
        serialize(a, storage, compression=None)
        b = deserialize(storage)
        assert numpy.array_equal(a, b)
        assert a.dtype == b.dtype
        assert a.shape == b.shape

    def test_compressed_numpy(self):
        storage = {}
        a = numpy.arange(100, dtype=">u2").reshape(-1, 5)
        serialize(a, storage, compression=zlib.compress)
        b = deserialize(storage)
        assert numpy.array_equal(a, b)
        assert a.dtype == b.dtype
        assert a.shape == b.shape

    def test_crossref(self):
        starts = [1, 0, 4, 0, 0]
        stops  = [4, 0, 5, 0, 0]
        a = awkward.JaggedArray(starts, stops, [])
        a.content = a
        a = awkward.IndexedArray([0, 4], a)
        storage = {}
        serialize(a, storage)
        b = deserialize(storage)
        assert a.tolist() == b.tolist()

    def test_two_in_one(self):
        storage = {}
        a1 = awkward.JaggedArray.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
        a2 = awkward.JaggedArray.fromiter([[], [1, 2], [3], []])
        serialize(a1, storage, "one")
        serialize(a2, storage, "two")
        b1 = deserialize(storage, "one")
        b2 = deserialize(storage, "two")
        assert a1.tolist() == b1.tolist()
        assert a2.tolist() == b2.tolist()

    def test_ChunkedArray(self):
        storage = {}
        a = awkward.ChunkedArray([[0.0, 1.1, 2.2], [], [3.3, 4.4]])
        serialize(a, storage)
        b = deserialize(storage)
        assert a.tolist() == b.tolist()

    def test_AppendableArray(self):
        storage = {}
        a = AppendableArray(3, numpy.float64)
        a.append(0.0)
        a.append(1.1)
        a.append(2.2)
        a.append(3.3)
        a.append(4.4)
        a.append(5.5)
        a.append(6.6)
        a.append(7.7)
        a.append(8.8)
        a.append(9.9)
        serialize(a, storage)
        b = deserialize(storage)
        assert a.tolist() == b.tolist()

    def test_IndexedArray(self):
        storage = {}
        a = awkward.IndexedArray([2, 3, 6, 3, 2, 2, 7], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        serialize(a, storage)
        b = deserialize(storage)
        assert a.tolist() == b.tolist()

    def test_SparseArray(self):
        storage = {}
        a = SparseArray(10, [1, 3, 5, 7, 9], [100, 101, 102, 103, 104])
        serialize(a, storage)
        b = deserialize(storage)
        assert a.tolist() == b.tolist()

    def test_JaggedArray(self):
        storage = {}
        a = awkward.JaggedArray([2, 1], [5, 2], [1.1, 2.2, 3.3, 4.4, 5.5])
        serialize(a, storage)
        b = deserialize(storage)
        assert a.tolist() == b.tolist()

    def test_JaggedArray_fromcounts(self):
        storage = {}
        a = awkward.JaggedArray.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
        serialize(a, storage)
        b = deserialize(storage)
        assert a.tolist() == b.tolist()

    def test_MaskedArray(self):
        storage = {}
        a = MaskedArray([True, False, True, False, True, False, True, False, True, False], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], maskedwhen=True)
        serialize(a, storage)
        b = deserialize(storage)
        assert a.tolist() == b.tolist()

    def test_BitMaskedArray(self):
        storage = {}
        a = BitMaskedArray.fromboolmask([True, False, True, False, True, False, True, False, True, False], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], maskedwhen=True, lsborder=True)
        serialize(a, storage)
        b = deserialize(storage)
        assert a.tolist() == b.tolist()
        
    def test_IndexedMaskedArray(self):
        storage = {}
        a = IndexedMaskedArray([-1, 0, -1, 1, -1, 2, -1, 4, -1, 3], [0.0, 1.1, 2.2, 3.3, 4.4])
        serialize(a, storage)
        b = deserialize(storage)
        assert a.tolist() == b.tolist()

    def test_ObjectArray(self):
        storage = {}
        a = ObjectArray([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6], [7.7, 8.8, 9.9]], FloatPoint)
        serialize(a, storage)
        b = deserialize(storage, whitelist="*")
        assert a.tolist() == b.tolist()

        storage = {}
        a = ObjectArray([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6], [7.7, 8.8, 9.9]], FloatPoint)
        serialize(a, storage)
        b = deserialize(storage, whitelist=whitelist + [["tests.test_persist", "FloatPoint"]])
        assert a.tolist() == b.tolist()

        a = ObjectArray(numpy.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]).view("u1").reshape(-1, 24), BytesPoint)
        serialize(a, storage)
        b = deserialize(storage, whitelist="*")
        assert a.tolist() == b.tolist()
        
    def test_Table(self):
        storage = {}
        a = Table([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], stuff=[5, 4, 3, 2, 1])
        serialize(a, storage)
        b = deserialize(storage)
        assert a.tolist() == b.tolist()

        storage = {}
        a = Table([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], stuff=[5, 4, 3, 2, 1])[5:]
        serialize(a, storage)
        b = deserialize(storage)
        assert a.tolist() == b.tolist()

        storage = {}
        a = Table([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], stuff=[5, 4, 3, 2, 1])[[True, False, False, True, False]]
        serialize(a, storage)
        b = deserialize(storage)
        assert a.tolist() == b.tolist()

    def test_UnionArray(self):
        storage = {}
        a = UnionArray([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [[0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], [0, 100, 200, 300, 400, 500, 600, 700, 800, 900]])
        serialize(a, storage)
        b = deserialize(storage)
        assert a.tolist() == b.tolist()

        storage = {}
        a = UnionArray.fromtags([0, 1, 1, 0, 0], [[100, 200, 300], [1.1, 2.2]])
        serialize(a, storage)
        b = deserialize(storage)
        assert a.tolist() == b.tolist()

    def test_VirtualArray(self):
        storage = {}
        a = awkward.VirtualArray(makearray)
        serialize(a, storage)
        cache = {}
        b = deserialize(storage, cache=cache, whitelist="*")
        assert a.tolist() == b.tolist()
        assert len(cache) == 1

        cache = {}
        b = deserialize(storage, cache=cache, whitelist=whitelist + [["tests.test_persist", "makearray"]])
        assert a.tolist() == b.tolist()
        assert len(cache) == 1

        storage = {}
        a = awkward.VirtualArray(makearray, persistentkey="find-me-again")
        serialize(a, storage)
        cache = {}
        b = deserialize(storage, cache=cache, whitelist="*")
        assert a.persistentkey == b.persistentkey
        assert a.tolist() == b.tolist()
        assert list(cache.keys()) == ["find-me-again"]

        storage = {}
        a = awkward.VirtualArray(makearray, type=ArrayType(10, numpy.dtype(int)))
        serialize(a, storage)
        b = deserialize(storage, whitelist="*")
        assert a.type == b.type

        storage = {}
        a = awkward.VirtualArray(makearray, persistvirtual=False)
        serialize(a, storage)
        b = deserialize(storage, whitelist="*")
        assert isinstance(b, numpy.ndarray)
        assert a.tolist() == b.tolist()
