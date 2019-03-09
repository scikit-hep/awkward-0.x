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

class Test(unittest.TestCase):
    def runTest(self):
        pass

    def test_union_nbytes(self):
        assert isinstance(UnionArray([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [[0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], [0, 100, 200, 300, 400, 500, 600, 700, 800, 900]]).nbytes, int)

    def test_union_get(self):
        a = UnionArray([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [[0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], [0, 100, 200, 300, 400, 500, 600, 700, 800, 900]])
        assert a.tolist() == [0.0, 100, 2.2, 300, 4.4, 500, 6.6, 700, 8.8, 900]
        assert [a[i] for i in range(len(a))] == [0.0, 100, 2.2, 300, 4.4, 500, 6.6, 700, 8.8, 900]
        assert a[:].tolist() == [0.0, 100, 2.2, 300, 4.4, 500, 6.6, 700, 8.8, 900]
        assert a[1:].tolist() == [100, 2.2, 300, 4.4, 500, 6.6, 700, 8.8, 900]
        assert a[2:].tolist() == [2.2, 300, 4.4, 500, 6.6, 700, 8.8, 900]
        assert a[:-1].tolist() == [0.0, 100, 2.2, 300, 4.4, 500, 6.6, 700, 8.8]
        assert a[1:-2].tolist() == [100, 2.2, 300, 4.4, 500, 6.6, 700]
        assert a[2:-2].tolist() == [2.2, 300, 4.4, 500, 6.6, 700]
        assert [a[2:-2][i] for i in range(6)] == [2.2, 300, 4.4, 500, 6.6, 700]

        assert a[[1, 2, 3, 8, 8, 1]].tolist() == [100, 2.2, 300, 8.8, 8.8, 100]
        assert [a[[1, 2, 3, 8, 8, 1]][i] for i in range(6)] == [100, 2.2, 300, 8.8, 8.8, 100]

        assert a[[False, True, True, True, False, True, False, False, False, False]].tolist() == [100, 2.2, 300, 500]
        assert [a[[False, True, True, True, False, True, False, False, False, False]][i] for i in range(4)] == [100, 2.2, 300, 500]

    def test_union_ufunc(self):
        a = UnionArray.fromtags([0, 1, 1, 0, 0], [[100, 200, 300], [1.1, 2.2]])
        b = UnionArray.fromtags([1, 1, 0, 1, 0], [[10.1, 20.2], [123, 456, 789]])
        assert (a + a).tolist() == [200, 2.2, 4.4, 400, 600]
        assert (a + b).tolist() == [223, 457.1, 12.3, 989, 320.2]
