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

import struct
import unittest

import numpy

from awkward import *

class Test(unittest.TestCase):
    def runTest(self):
        pass

    def test_object_nbytes(self):
        class Point(object):
            def __init__(self, array):
                self.x, self.y, self.z = array
            def __repr__(self):
                return "<Point {0} {1} {2}>".format(self.x, self.y, self.z)
            def __eq__(self, other):
                return isinstance(other, Point) and self.x == other.x and self.y == other.y and self.z == other.z

        assert isinstance(ObjectArray([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6], [7.7, 8.8, 9.9]], Point).nbytes, int)

    def test_object_floats(self):
        class Point(object):
            def __init__(self, array):
                self.x, self.y, self.z = array
            def __repr__(self):
                return "<Point {0} {1} {2}>".format(self.x, self.y, self.z)
            def __eq__(self, other):
                return isinstance(other, Point) and self.x == other.x and self.y == other.y and self.z == other.z

        a = ObjectArray([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6], [7.7, 8.8, 9.9]], Point)
        assert a[0] == Point([1.1, 2.2, 3.3])
        assert a[1] == Point([4.4, 5.5, 6.6])
        assert a[2] == Point([7.7, 8.8, 9.9])
        assert a[:].tolist() == [Point([1.1, 2.2, 3.3]), Point([4.4, 5.5, 6.6]), Point([7.7, 8.8, 9.9])]
        assert a[::2].tolist() == [Point([1.1, 2.2, 3.3]), Point([7.7, 8.8, 9.9])]
        assert a[[True, False, True]].tolist() == [Point([1.1, 2.2, 3.3]), Point([7.7, 8.8, 9.9])]
        assert a[[2, 0]].tolist() == [Point([7.7, 8.8, 9.9]), Point([1.1, 2.2, 3.3])]

    def test_object_bytes(self):
        class Point(object):
            def __init__(self, bytes):
                self.x, self.y, self.z = struct.unpack("ddd", bytes)
            def __repr__(self):
                return "<Point {0} {1} {2}>".format(self.x, self.y, self.z)
            def __eq__(self, other):
                return isinstance(other, Point) and self.x == other.x and self.y == other.y and self.z == other.z

        a = ObjectArray(numpy.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]).view("u1").reshape(-1, 24), Point)
        assert a[0] == Point(numpy.array([1.1, 2.2, 3.3]).tobytes())
        assert a[1] == Point(numpy.array([4.4, 5.5, 6.6]).tobytes())
        assert a[2] == Point(numpy.array([7.7, 8.8, 9.9]).tobytes())
        assert a[:].tolist() == [Point(numpy.array([1.1, 2.2, 3.3]).tobytes()), Point(numpy.array([4.4, 5.5, 6.6]).tobytes()), Point(numpy.array([7.7, 8.8, 9.9]).tobytes())]
        assert a[::2].tolist() == [Point(numpy.array([1.1, 2.2, 3.3]).tobytes()), Point(numpy.array([7.7, 8.8, 9.9]).tobytes())]
        assert a[[True, False, True]].tolist() == [Point(numpy.array([1.1, 2.2, 3.3]).tobytes()), Point(numpy.array([7.7, 8.8, 9.9]).tobytes())]
        assert a[[2, 0]].tolist() == [Point(numpy.array([7.7, 8.8, 9.9]).tobytes()), Point(numpy.array([1.1, 2.2, 3.3]).tobytes())]
