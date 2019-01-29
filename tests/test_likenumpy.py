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

import awkward

class Test(unittest.TestCase):
    def runTest(self):
        pass

    def test_likenumpy_slices(self):
        print()

        np = numpy.array([[1, 10, 100], [2, 20, 200], [3, 30, 300]])
        aw = awkward.fromiter(np)

        assert np.tolist() == aw.tolist()
        assert np[:2].tolist() == aw[:2].tolist()
        assert np[:2, :2].tolist() == aw[:2, :2].tolist()
        assert np[:2, 2].tolist() == aw[:2, 2].tolist()
        assert np[2, :2].tolist() == aw[2, :2].tolist()
        assert np[:2, [0, 1]].tolist() == aw[:2, [0, 1]].tolist()
        assert np[[0, 1], :2].tolist() == aw[[0, 1], :2].tolist()
        assert np[:2, [0, 1, 2]].tolist() == aw[:2, [0, 1, 2]].tolist()
        assert np[[0, 1, 2], :2].tolist() == aw[[0, 1, 2], :2].tolist()
        assert np[[0, 1], [0, 1]].tolist() == aw[[0, 1], [0, 1]].tolist()
        assert np[[0, 1, 2], [0, 1, 2]].tolist() == aw[[0, 1, 2], [0, 1, 2]].tolist()
        assert np[:2, [True, False, True]].tolist() == aw[:2, [True, False, True]].tolist()
        assert np[[True, False, True], :2].tolist() == aw[[True, False, True], :2].tolist()
        assert np[[True, False, True], [True, False, True]].tolist() == aw[[True, False, True], [True, False, True]].tolist()

        np = numpy.array([[[1, 10, 100], [2, 20, 200], [3, 30, 300]], [[4, 40, 400], [5, 50, 500], [6, 60, 600]], [[7, 70, 700], [8, 80, 800], [9, 90, 900]]])
        aw = awkward.fromiter(np)

        assert np.tolist() == aw.tolist()
        assert np[:2].tolist() == aw[:2].tolist()
        assert np[:2, :2].tolist() == aw[:2, :2].tolist()
        assert np[:2, 2].tolist() == aw[:2, 2].tolist()
        assert np[2, :2].tolist() == aw[2, :2].tolist()
        assert np[:2, [0, 1]].tolist() == aw[:2, [0, 1]].tolist()
        assert np[[0, 1], :2].tolist() == aw[[0, 1], :2].tolist()
        assert np[:2, [0, 1, 2]].tolist() == aw[:2, [0, 1, 2]].tolist()
        assert np[[0, 1, 2], :2].tolist() == aw[[0, 1, 2], :2].tolist()
        assert np[[0, 1], [0, 1]].tolist() == aw[[0, 1], [0, 1]].tolist()
        assert np[[0, 1, 2], [0, 1, 2]].tolist() == aw[[0, 1, 2], [0, 1, 2]].tolist()
        assert np[:2, [True, False, True]].tolist() == aw[:2, [True, False, True]].tolist()
        assert np[[True, False, True], :2].tolist() == aw[[True, False, True], :2].tolist()
        assert np[[True, False, True], [True, False, True]].tolist() == aw[[True, False, True], [True, False, True]].tolist()

        assert np[:2, :2, 0].tolist() == aw[:2, :2, 0].tolist()
        assert np[:2, 2, 0].tolist() == aw[:2, 2, 0].tolist()
        assert np[2, :2, 0].tolist() == aw[2, :2, 0].tolist()
        assert np[:2, [0, 1], 0].tolist() == aw[:2, [0, 1], 0].tolist()
        assert np[[0, 1], :2, 0].tolist() == aw[[0, 1], :2, 0].tolist()
        assert np[:2, [0, 1, 2], 0].tolist() == aw[:2, [0, 1, 2], 0].tolist()
        assert np[[0, 1, 2], :2, 0].tolist() == aw[[0, 1, 2], :2, 0].tolist()
        assert np[[0, 1], [0, 1], 0].tolist() == aw[[0, 1], [0, 1], 0].tolist()
        assert np[[0, 1, 2], [0, 1, 2], 0].tolist() == aw[[0, 1, 2], [0, 1, 2], 0].tolist()
        assert np[:2, [True, False, True], 0].tolist() == aw[:2, [True, False, True], 0].tolist()
        assert np[[True, False, True], :2, 0].tolist() == aw[[True, False, True], :2, 0].tolist()
        assert np[[True, False, True], [True, False, True], 0].tolist() == aw[[True, False, True], [True, False, True], 0].tolist()

        assert np[:2, :2, 1].tolist() == aw[:2, :2, 1].tolist()
        assert np[:2, 2, 1].tolist() == aw[:2, 2, 1].tolist()
        assert np[2, :2, 1].tolist() == aw[2, :2, 1].tolist()
        assert np[:2, [0, 1], 1].tolist() == aw[:2, [0, 1], 1].tolist()
        assert np[[0, 1], :2, 1].tolist() == aw[[0, 1], :2, 1].tolist()
        assert np[:2, [0, 1, 2], 1].tolist() == aw[:2, [0, 1, 2], 1].tolist()
        assert np[[0, 1, 2], :2, 1].tolist() == aw[[0, 1, 2], :2, 1].tolist()
        assert np[[0, 1], [0, 1], 1].tolist() == aw[[0, 1], [0, 1], 1].tolist()
        assert np[[0, 1, 2], [0, 1, 2], 1].tolist() == aw[[0, 1, 2], [0, 1, 2], 1].tolist()
        assert np[:2, [True, False, True], 1].tolist() == aw[:2, [True, False, True], 1].tolist()
        assert np[[True, False, True], :2, 1].tolist() == aw[[True, False, True], :2, 1].tolist()
        assert np[[True, False, True], [True, False, True], 1].tolist() == aw[[True, False, True], [True, False, True], 1].tolist()
