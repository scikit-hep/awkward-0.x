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

class Test(unittest.TestCase):
    def runTest(self):
        pass

    def test_table_get(self):
        a = Table([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])

        self.assertEqual(a[5]["0"], 5)
        self.assertEqual(a["0"][5], 5)

        self.assertEqual(a[5]["1"], 5.5)
        self.assertEqual(a["1"][5], 5.5)

        self.assertEqual(a[5:]["0"][0], 5)
        self.assertEqual(a["0"][5:][0], 5)
        self.assertEqual(a[5:][0]["0"], 5)

        self.assertEqual(a[::-2]["0"][-1], 1)
        self.assertEqual(a["0"][::-2][-1], 1)
        self.assertEqual(a[::-2][-1]["0"], 1)

        self.assertEqual(a[[5, 3, 7, 5]]["0"].tolist(), [5, 3, 7, 5])
        self.assertEqual(a["0"][[5, 3, 7, 5]].tolist(), [5, 3, 7, 5])

        self.assertEqual(a["0"][[True, False, True, False, True, False, True, False, True, False]].tolist(), [0, 2, 4, 6, 8])
        self.assertEqual(a[[True, False, True, False, True, False, True, False, True, False]]["0"].tolist(), [0, 2, 4, 6, 8])

    def test_table_slice_slice(self):
        a = Table([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        self.assertEqual(a[::2][2:4].tolist(), [{"0": 4, "1": 4.4}, {"0": 6, "1": 6.6}])
        self.assertEqual(a[1::2][2:100].tolist(), [{"0": 5, "1": 5.5}, {"0": 7, "1": 7.7}, {"0": 9, "1": 9.9}])
        self.assertEqual(a[5:][2:4].tolist(), [{"0": 7, "1": 7.7}, {"0": 8, "1": 8.8}])
        self.assertEqual(a[-5:][2:4].tolist(), [{"0": 7, "1": 7.7}, {"0": 8, "1": 8.8}])
        self.assertEqual(a[::2][-4:-2].tolist(), [{"0": 2, "1": 2.2}, {"0": 4, "1": 4.4}])
        self.assertEqual(a[::-2][2:4].tolist(), [{"0": 5, "1": 5.5}, {"0": 3, "1": 3.3}])
        self.assertEqual(a[::-2][2:100].tolist(), [{"0": 5, "1": 5.5}, {"0": 3, "1": 3.3}, {"0": 1, "1": 1.1}])

    def test_table_slice_fancy(self):
        a = Table([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        self.assertEqual(a[::2][[4, 3, 1, 1]].tolist(), [{"0": 8, "1": 8.8}, {"0": 6, "1": 6.6}, {"0": 2, "1": 2.2}, {"0": 2, "1": 2.2}])
        self.assertEqual(a[-5::-1][[0, 1, 5]].tolist(), [{"0": 5, "1": 5.5}, {"0": 4, "1": 4.4}, {"0": 0, "1": 0.0}])

    def test_table_slice_mask(self):
        a = Table([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        self.assertEqual(a[::2][[True, False, False, True, True]].tolist(), [{"0": 0, "1": 0.0}, {"0": 6, "1": 6.6}, {"0": 8, "1": 8.8}])
        self.assertEqual(a[-5::-1][[True, True, False, False, False, True]].tolist(), [{"0": 5, "1": 5.5}, {"0": 4, "1": 4.4}, {"0": 0, "1": 0.0}])

    def test_table_fancy_slice(self):
        a = Table([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        self.assertEqual(a[[7, 3, 3, 4, 4][2:]].tolist(), [{"0": 3, "1": 3.3}, {"0": 4, "1": 4.4}, {"0": 4, "1": 4.4}])

    def test_table_fancy_fancy(self):
        a = Table([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        self.assertEqual(a[[7, 3, 3, 4, 4]][[-2, 2, 0]].tolist(), [{"0": 4, "1": 4.4}, {"0": 3, "1": 3.3}, {"0": 7, "1": 7.7}])

    def test_table_fancy_mask(self):
        a = Table([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        self.assertEqual(a[[7, 3, 3, 4, 4]][[True, False, True, False, True]].tolist(), [{"0": 7, "1": 7.7}, {"0": 3, "1": 3.3}, {"0": 4, "1": 4.4}])

    def test_table_mask_slice(self):
        a = Table([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        self.assertEqual(a[[True, True, False, False, False, False, False, True, True, True]][1:4].tolist(), [{"0": 1, "1": 1.1}, {"0": 7, "1": 7.7}, {"0": 8, "1": 8.8}])

    def test_table_mask_fancy(self):
        a = Table([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        self.assertEqual(a[[True, True, False, False, False, False, False, True, True, True]][[1, 2, 3]].tolist(), [{"0": 1, "1": 1.1}, {"0": 7, "1": 7.7}, {"0": 8, "1": 8.8}])

    def test_table_mask_mask(self):
        a = Table([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        self.assertEqual(a[[True, True, False, False, False, False, False, True, True, True]][[False, True, True, True, False]].tolist(), [{"0": 1, "1": 1.1}, {"0": 7, "1": 7.7}, {"0": 8, "1": 8.8}])

    # def test_indexed_table(self):
    #     a = Table([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    #     self.assertEqual(IndexedArray([5, 3, 7, 5], a)["1"].tolist(), [5.5, 3.3, 7.7, 5.5])

    # def test_masked_table(self):
    #     a = Table([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    #     self.assertEqual(MaskedArray([False, True, True, True, True, False, False, False, False, True], a, maskedwhen=False)["1"].tolist(), [None, 1.1, 2.2, 3.3, 4.4, None, None, None, None, 9.9])

    def test_jagged_table(self):
        a = Table([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        self.assertEqual(JaggedArray.fromoffsets([0, 3, 5, 5, 10], a).tolist(), [[{"0": 0, "1": 0.0}, {"0": 1, "1": 1.1}, {"0": 2, "1": 2.2}], [{"0": 3, "1": 3.3}, {"0": 4, "1": 4.4}], [], [{"0": 5, "1": 5.5}, {"0": 6, "1": 6.6}, {"0": 7, "1": 7.7}, {"0": 8, "1": 8.8}, {"0": 9, "1": 9.9}]])
        self.assertEqual(JaggedArray.fromoffsets([0, 3, 5, 5, 10], a)["1"].tolist(), [[0.0, 1.1, 2.2], [3.3, 4.4], [], [5.5, 6.6, 7.7, 8.8, 9.9]])

    # def test_chunked_table(self):
    #     a = Table([0, 1, 2, 3], [0.0, 1.1, 2.2, 3.3])
    #     b = Table([4, 5, 6, 7, 8, 9], [4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    #     c = ChunkedArray([a, b])
    #     self.assertEqual(c["1"][6], 6.6)

    # def test_virtual_table(self):
    #     a = VirtualArray(lambda: Table([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
    #     self.assertEqual(a.tolist(), [{"0": 0, "1": 0.0}, {"0": 1, "1": 1.1}, {"0": 2, "1": 2.2}, {"0": 3, "1": 3.3}, {"0": 4, "1": 4.4}, {"0": 5, "1": 5.5}, {"0": 6, "1": 6.6}, {"0": 7, "1": 7.7}, {"0": 8, "1": 8.8}, {"0": 9, "1": 9.9}])
