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

import struct
import unittest

import numpy

from awkward import *

class Test(unittest.TestCase):
    def runTest(self):
        pass

    ################### old tests

    # def test_virtual_nocache(self):
    #     a = VirtualArray(lambda: [1, 2, 3])
    #     self.assertFalse(a.ismaterialized)
    #     self.assertTrue(numpy.array_equal(a[:], numpy.array([1, 2, 3])))
    #     self.assertTrue(a.ismaterialized)

    #     a = VirtualArray(lambda: range(10))
    #     self.assertFalse(a.ismaterialized)
    #     self.assertTrue(numpy.array_equal(a[::2], numpy.array([0, 2, 4, 6, 8])))
    #     self.assertTrue(a.ismaterialized)

    #     a = VirtualArray(lambda: range(10))
    #     self.assertFalse(a.ismaterialized)
    #     self.assertTrue(numpy.array_equal(a[[5, 3, 6, 0, 6]], numpy.array([5, 3, 6, 0, 6])))
    #     self.assertTrue(a.ismaterialized)

    #     a = VirtualArray(lambda: range(10))
    #     self.assertFalse(a.ismaterialized)
    #     self.assertTrue(numpy.array_equal(a[[True, False, True, False, True, False, True, False, True, False]], numpy.array([0, 2, 4, 6, 8])))
    #     self.assertTrue(a.ismaterialized)

    # def test_virtual_transientcache(self):
    #     cache = {}
    #     a = VirtualArray(lambda: [1, 2, 3], cache=cache)
    #     self.assertFalse(a.ismaterialized)
    #     a[:]
    #     self.assertTrue(a.ismaterialized)
    #     self.assertEqual(list(cache), [a.TransientKey(id(a))])
    #     self.assertEqual(list(cache), [a.key])
    #     self.assertTrue(numpy.array_equal(cache[a.key], numpy.array([1, 2, 3])))
    #     del a

    # def test_virtual_persistentcache(self):
    #     cache = {}
    #     a = VirtualArray(lambda: [1, 2, 3], cache=cache, persistentkey="find-me-again")
    #     self.assertFalse(a.ismaterialized)
    #     a[:]
    #     self.assertTrue(a.ismaterialized)
    #     self.assertEqual(list(cache), ["find-me-again"])
    #     self.assertEqual(list(cache), [a.key])
    #     self.assertTrue(numpy.array_equal(cache[a.key], numpy.array([1, 2, 3])))
    #     del a

    # def test_virtual_dontmaterialize(self):
    #     a = VirtualArray(lambda: [1, 2, 3], dtype=int, shape=(3,))
    #     self.assertFalse(a.ismaterialized)
    #     self.assertEqual(a.dtype, numpy.dtype(int))
    #     self.assertEqual(a.shape, (3,))
    #     self.assertEqual(len(a), 3)
    #     self.assertEqual(a._array, None)
    #     self.assertFalse(a.ismaterialized)
    #     self.assertTrue(numpy.array_equal(a[:], numpy.array([1, 2, 3])))
    #     self.assertTrue(a.ismaterialized)

    # def test_virtualobject_floats(self):
    #     class Point(object):
    #         def __init__(self, array):
    #             self.x, self.y, self.z = array
    #         def __repr__(self):
    #             return "<Point {0} {1} {2}>".format(self.x, self.y, self.z)
    #         def __eq__(self, other):
    #             return isinstance(other, Point) and self.x == other.x and self.y == other.y and self.z == other.z

    #     a = VirtualObjectArray(Point, [[1.1, 2.2, 3.3], [4.4, 5.5, 6.6], [7.7, 8.8, 9.9]])
    #     self.assertEqual(a[0], Point([1.1, 2.2, 3.3]))
    #     self.assertEqual(a[1], Point([4.4, 5.5, 6.6]))
    #     self.assertEqual(a[2], Point([7.7, 8.8, 9.9]))
    #     self.assertEqual(a[:], [Point([1.1, 2.2, 3.3]), Point([4.4, 5.5, 6.6]), Point([7.7, 8.8, 9.9])])
    #     self.assertEqual(a[::2], [Point([1.1, 2.2, 3.3]), Point([7.7, 8.8, 9.9])])
    #     self.assertEqual(a[[True, False, True]], [Point([1.1, 2.2, 3.3]), Point([7.7, 8.8, 9.9])])
    #     self.assertEqual(a[[2, 0]], [Point([7.7, 8.8, 9.9]), Point([1.1, 2.2, 3.3])])

    # def test_virtualobject_bytes(self):
    #     class Point(object):
    #         def __init__(self, bytes):
    #             self.x, self.y, self.z = struct.unpack("ddd", bytes)
    #         def __repr__(self):
    #             return "<Point {0} {1} {2}>".format(self.x, self.y, self.z)
    #         def __eq__(self, other):
    #             return isinstance(other, Point) and self.x == other.x and self.y == other.y and self.z == other.z

    #     a = VirtualObjectArray(Point, numpy.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]).view("u1").reshape(-1, 24))
    #     self.assertEqual(a[0], Point(numpy.array([1.1, 2.2, 3.3]).tobytes()))
    #     self.assertEqual(a[1], Point(numpy.array([4.4, 5.5, 6.6]).tobytes()))
    #     self.assertEqual(a[2], Point(numpy.array([7.7, 8.8, 9.9]).tobytes()))
    #     self.assertEqual(a[:], [Point(numpy.array([1.1, 2.2, 3.3]).tobytes()), Point(numpy.array([4.4, 5.5, 6.6]).tobytes()), Point(numpy.array([7.7, 8.8, 9.9]).tobytes())])
    #     self.assertEqual(a[::2], [Point(numpy.array([1.1, 2.2, 3.3]).tobytes()), Point(numpy.array([7.7, 8.8, 9.9]).tobytes())])
    #     self.assertEqual(a[[True, False, True]], [Point(numpy.array([1.1, 2.2, 3.3]).tobytes()), Point(numpy.array([7.7, 8.8, 9.9]).tobytes())])
    #     self.assertEqual(a[[2, 0]], [Point(numpy.array([7.7, 8.8, 9.9]).tobytes()), Point(numpy.array([1.1, 2.2, 3.3]).tobytes())])

    # def test_virtualobject_indexedbytes(self):
    #     class Point(object):
    #         def __init__(self, array):
    #             self.x, self.y, self.z = array
    #         def __repr__(self):
    #             return "<Point {0} {1} {2}>".format(self.x, self.y, self.z)
    #         def __eq__(self, other):
    #             return isinstance(other, Point) and self.x == other.x and self.y == other.y and self.z == other.z

    #     a = VirtualObjectArray(Point, ByteIndexedArray([0, 24, 48], numpy.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]).view("u1"), numpy.dtype((float, 3))))
    #     self.assertEqual(a[0], Point([1.1, 2.2, 3.3]))
    #     self.assertEqual(a[1], Point([4.4, 5.5, 6.6]))
    #     self.assertEqual(a[2], Point([7.7, 8.8, 9.9]))
    #     self.assertEqual(a[:], [Point([1.1, 2.2, 3.3]), Point([4.4, 5.5, 6.6]), Point([7.7, 8.8, 9.9])])
    #     self.assertEqual(a[::2], [Point([1.1, 2.2, 3.3]), Point([7.7, 8.8, 9.9])])
    #     self.assertEqual(a[[True, False, True]], [Point([1.1, 2.2, 3.3]), Point([7.7, 8.8, 9.9])])
    #     self.assertEqual(a[[2, 0]], [Point([7.7, 8.8, 9.9]), Point([1.1, 2.2, 3.3])])

    # def test_virtualobject_jaggedbytes(self):
    #     class Point(object):
    #         def __init__(self, array):
    #             self.x, self.y, self.z = array
    #         def __repr__(self):
    #             return "<Point {0} {1} {2}>".format(self.x, self.y, self.z)
    #         def __eq__(self, other):
    #             return isinstance(other, Point) and self.x == other.x and self.y == other.y and self.z == other.z

    #     a = VirtualObjectArray(Point, ByteJaggedArray.fromoffsets([0, 24, 48, 72], numpy.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]).view("u1"), float))
    #     self.assertEqual(a[0], Point([1.1, 2.2, 3.3]))
    #     self.assertEqual(a[1], Point([4.4, 5.5, 6.6]))
    #     self.assertEqual(a[2], Point([7.7, 8.8, 9.9]))
    #     self.assertEqual(a[:], [Point([1.1, 2.2, 3.3]), Point([4.4, 5.5, 6.6]), Point([7.7, 8.8, 9.9])])
    #     self.assertEqual(a[::2], [Point([1.1, 2.2, 3.3]), Point([7.7, 8.8, 9.9])])
    #     self.assertEqual(a[[True, False, True]], [Point([1.1, 2.2, 3.3]), Point([7.7, 8.8, 9.9])])
    #     self.assertEqual(a[[2, 0]], [Point([7.7, 8.8, 9.9]), Point([1.1, 2.2, 3.3])])
