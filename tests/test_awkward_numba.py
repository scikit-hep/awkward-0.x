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

import pytest
import numpy

awkward_numba = pytest.importorskip("awkward.numba")

class Test(unittest.TestCase):
    def runTest(self):
        pass

    def test_awkward_numba_init(self):
        a = awkward_numba.JaggedArray([0, 3, 3, 5], [3, 3, 5, 10], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        assert isinstance(a, awkward_numba.array.jagged.JaggedArrayNumba)
        assert a.JaggedArray is awkward_numba.array.jagged.JaggedArrayNumba

        assert isinstance(a[2:], awkward_numba.array.jagged.JaggedArrayNumba)
        assert isinstance(a[[False, True, True, False]], awkward_numba.array.jagged.JaggedArrayNumba)
        assert isinstance(a[[1, 2]], awkward_numba.array.jagged.JaggedArrayNumba)
        assert isinstance(a + 100, awkward_numba.array.jagged.JaggedArrayNumba)
        assert isinstance(numpy.sin(a), awkward_numba.array.jagged.JaggedArrayNumba)
        assert isinstance(a.cross(a), awkward_numba.array.jagged.JaggedArrayNumba)
        assert isinstance(a.pairs(), awkward_numba.array.jagged.JaggedArrayNumba)
        assert isinstance(a.distincts(), awkward_numba.array.jagged.JaggedArrayNumba)
        assert isinstance(a.cross(a, nested=True), awkward_numba.array.jagged.JaggedArrayNumba)
        assert isinstance(a.pairs(nested=True), awkward_numba.array.jagged.JaggedArrayNumba)
        assert isinstance(a.distincts(nested=True), awkward_numba.array.jagged.JaggedArrayNumba)
        assert isinstance(a.concatenate([a]), awkward_numba.array.jagged.JaggedArrayNumba)

        b = awkward_numba.JaggedArray([1, 4, 4, 6], [4, 4, 6, 11], [999, 0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        assert isinstance(a + b, awkward_numba.array.jagged.JaggedArrayNumba)

    def test_awkward_numba_argmin(self):
        a = awkward_numba.JaggedArray([0, 3, 3, 5], [3, 3, 5, 10], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        assert isinstance(a.argmin(), awkward_numba.array.jagged.JaggedArrayNumba)
        assert a.argmin().tolist() == [[0], [], [0], [0]]

        a = awkward_numba.JaggedArray([[0, 3], [3, 5]], [[3, 3], [5, 10]], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        assert isinstance(a.argmin(), awkward_numba.array.jagged.JaggedArrayNumba)
        assert a.argmin().tolist() == [[[0], []], [[0], [0]]]

    def test_awkward_numba_argmax(self):
        a = awkward_numba.JaggedArray([0, 3, 3, 5], [3, 3, 5, 10], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        assert isinstance(a.argmin(), awkward_numba.array.jagged.JaggedArrayNumba)
        assert a.argmax().tolist() == [[2], [], [1], [4]]

        a = awkward_numba.JaggedArray([[0, 3], [3, 5]], [[3, 3], [5, 10]], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        assert isinstance(a.argmin(), awkward_numba.array.jagged.JaggedArrayNumba)
        assert a.argmax().tolist() == [[[2], []], [[1], [4]]]
