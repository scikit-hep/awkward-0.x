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
import awkward.type

class Test(unittest.TestCase):
    def runTest(self):
        pass

    def test_virtual_nbytes(self):
        assert isinstance(VirtualArray(lambda: [1, 2, 3]).nbytes, int)
        assert VirtualArray(lambda: [1, 2, 3], nbytes=12345).nbytes == 12345

    def test_virtual_nocache(self):
        a = VirtualArray(lambda: [1, 2, 3])
        assert not a.ismaterialized
        assert numpy.array_equal(a[:], numpy.array([1, 2, 3]))
        assert a.ismaterialized

        a = VirtualArray(lambda: range(10))
        assert not a.ismaterialized
        assert numpy.array_equal(a[::2], numpy.array([0, 2, 4, 6, 8]))
        assert a.ismaterialized

        a = VirtualArray(lambda: range(10))
        assert not a.ismaterialized
        assert numpy.array_equal(a[[5, 3, 6, 0, 6]], numpy.array([5, 3, 6, 0, 6]))
        assert a.ismaterialized

        a = VirtualArray(lambda: range(10))
        assert not a.ismaterialized
        assert numpy.array_equal(a[[True, False, True, False, True, False, True, False, True, False]], numpy.array([0, 2, 4, 6, 8]))
        assert a.ismaterialized

    def test_virtual_transientcache(self):
        cache = {}
        a = VirtualArray(lambda: [1, 2, 3], cache=cache)
        assert not a.ismaterialized
        a[:]
        assert a.ismaterialized
        assert list(cache) == [a.TransientKey(id(a))]
        assert list(cache) == [a.key]
        assert numpy.array_equal(cache[a.key], numpy.array([1, 2, 3]))
        del a

    def test_virtual_persistentcache(self):
        cache = {}
        a = VirtualArray(lambda: [1, 2, 3], cache=cache, persistentkey="find-me-again")
        assert not a.ismaterialized
        a[:]
        assert a.ismaterialized
        assert list(cache) == ["find-me-again"]
        assert list(cache) == [a.key]
        assert numpy.array_equal(cache[a.key], numpy.array([1, 2, 3]))
        del a

    def test_virtual_dontmaterialize(self):
        a = VirtualArray(lambda: [1, 2, 3], type=awkward.type.fromnumpy(3, int))
        assert not a.ismaterialized
        assert a.dtype == numpy.dtype(int)
        assert a.shape == (3,)
        assert len(a) == 3
        assert a._array == None
        assert not a.ismaterialized
        assert numpy.array_equal(a[:], numpy.array([1, 2, 3]))
        assert a.ismaterialized
