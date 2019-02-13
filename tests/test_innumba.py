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

import sys

import unittest

import numpy
import pytest

numba = pytest.importorskip("numba")
awkward_numba = pytest.importorskip("awkward.numba")

from awkward import *

class Test(unittest.TestCase):
    def runTest(self):
        pass

    # def test_innumba_unbox(self):
    #     a = JaggedArray.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
    #     a2 = JaggedArray.fromcounts([2, 0, 1], a)
    #     @numba.njit
    #     def test(x):
    #         return 3.14
    #     test(a)
    #     test(a2)

    # def test_innumba_box(self):
    #     a = JaggedArray.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
    #     a2 = JaggedArray.fromcounts([2, 0, 1], a)
    #     @numba.njit
    #     def test(x):
    #         return x
    #     assert test(a).tolist() == a.tolist()
    #     assert test(a2).tolist() == a2.tolist()

    def test_innumba_getitem(self):
        a = JaggedArray.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
        a2 = JaggedArray.fromcounts([2, 0, 1], a)
        # @numba.njit
        # def test1(x, i, j):
        #     return x[i][j]
        # assert test1(a, 0, 0) == 1.1
        # assert test1(a, 0, 1) == 2.2
        # assert test1(a, 0, 2) == 3.3
        # assert test1(a, 2, 0) == 4.4
        # assert test1(a, 2, 1) == 5.5
        @numba.njit
        def test2(x, i):
            return x[i]
        # assert test2(a, 0).tolist() == [1.1, 2.2, 3.3]
        # assert test2(a, 1).tolist() == []
        # assert test2(a, 2).tolist() == [4.4, 5.5]
        assert test2(a2, 0).tolist() == [[1.1, 2.2, 3.3], []]
        # assert test2(a2, 1).tolist() == []
        # assert test2(a2, 2).tolist() == [[4.4, 5.5]]

