#!/usr/bin/env python

# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-array/blob/master/LICENSE

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
