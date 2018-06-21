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

class TestTable(unittest.TestCase):
    def runTest(self):
        pass

    def test_table_get(self):
        a = Table(10, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])

        self.assertEqual(a[5]["f0"], 5)
        self.assertEqual(a["f0"][5], 5)

        self.assertEqual(a[5]["f1"], 5.5)
        self.assertEqual(a["f1"][5], 5.5)

        self.assertEqual(a[5:]["f0"][0], 5)
        self.assertEqual(a["f0"][5:][0], 5)
        self.assertEqual(a[5:][0]["f0"], 5)

        self.assertEqual(a[::-2]["f0"][-1], 1)
        self.assertEqual(a["f0"][::-2][-1], 1)
        self.assertEqual(a[::-2][-1]["f0"], 1)

        self.assertEqual(a[[5, 3, 7, 5]]["f0"].tolist(), [5, 3, 7, 5])
        self.assertEqual(a["f0"][[5, 3, 7, 5]].tolist(), [5, 3, 7, 5])

        self.assertEqual(a["f0"][[True, False, True, False, True, False, True, False, True, False]].tolist(), [0, 2, 4, 6, 8])
        self.assertEqual(a[[True, False, True, False, True, False, True, False, True, False]]["f0"].tolist(), [0, 2, 4, 6, 8])

    def test_table_set(self):
        a = Table(10, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        self.assertEqual(a.tolist(), [{"f0": 0, "f1": 0.0}, {"f0": 1, "f1": 1.1}, {"f0": 2, "f1": 2.2}, {"f0": 3, "f1": 3.3}, {"f0": 4, "f1": 4.4}, {"f0": 5, "f1": 5.5}, {"f0": 6, "f1": 6.6}, {"f0": 7, "f1": 7.7}, {"f0": 8, "f1": 8.8}, {"f0": 9, "f1": 9.9}])

        a = Table(10, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        a[5]["f0"] = 999
        self.assertEqual(a.tolist(), [{"f0": 0, "f1": 0.0}, {"f0": 1, "f1": 1.1}, {"f0": 2, "f1": 2.2}, {"f0": 3, "f1": 3.3}, {"f0": 4, "f1": 4.4}, {"f0": 999, "f1": 5.5}, {"f0": 6, "f1": 6.6}, {"f0": 7, "f1": 7.7}, {"f0": 8, "f1": 8.8}, {"f0": 9, "f1": 9.9}])

        a = Table(10, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        a["f0"][5] = 999
        self.assertEqual(a.tolist(), [{"f0": 0, "f1": 0.0}, {"f0": 1, "f1": 1.1}, {"f0": 2, "f1": 2.2}, {"f0": 3, "f1": 3.3}, {"f0": 4, "f1": 4.4}, {"f0": 999, "f1": 5.5}, {"f0": 6, "f1": 6.6}, {"f0": 7, "f1": 7.7}, {"f0": 8, "f1": 8.8}, {"f0": 9, "f1": 9.9}])

        a = Table(10, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        a[5:]["f0"][0] = 999
        self.assertEqual(a.tolist(), [{"f0": 0, "f1": 0.0}, {"f0": 1, "f1": 1.1}, {"f0": 2, "f1": 2.2}, {"f0": 3, "f1": 3.3}, {"f0": 4, "f1": 4.4}, {"f0": 999, "f1": 5.5}, {"f0": 6, "f1": 6.6}, {"f0": 7, "f1": 7.7}, {"f0": 8, "f1": 8.8}, {"f0": 9, "f1": 9.9}])

        a = Table(10, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        a["f0"][5:][0] = 999
        self.assertEqual(a.tolist(), [{"f0": 0, "f1": 0.0}, {"f0": 1, "f1": 1.1}, {"f0": 2, "f1": 2.2}, {"f0": 3, "f1": 3.3}, {"f0": 4, "f1": 4.4}, {"f0": 999, "f1": 5.5}, {"f0": 6, "f1": 6.6}, {"f0": 7, "f1": 7.7}, {"f0": 8, "f1": 8.8}, {"f0": 9, "f1": 9.9}])

        a = Table(10, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        a[5:][0]["f0"] = 999
        self.assertEqual(a.tolist(), [{"f0": 0, "f1": 0.0}, {"f0": 1, "f1": 1.1}, {"f0": 2, "f1": 2.2}, {"f0": 3, "f1": 3.3}, {"f0": 4, "f1": 4.4}, {"f0": 999, "f1": 5.5}, {"f0": 6, "f1": 6.6}, {"f0": 7, "f1": 7.7}, {"f0": 8, "f1": 8.8}, {"f0": 9, "f1": 9.9}])

        a = Table(10, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        a[["f1", "f0"]] = 999
        self.assertEqual(a.tolist(), [{"f0": 999, "f1": 999.0}, {"f0": 999, "f1": 999.0}, {"f0": 999, "f1": 999.0}, {"f0": 999, "f1": 999.0}, {"f0": 999, "f1": 999.0}, {"f0": 999, "f1": 999.0}, {"f0": 999, "f1": 999.0}, {"f0": 999, "f1": 999.0}, {"f0": 999, "f1": 999.0}, {"f0": 999, "f1": 999.0}])

        a = Table(10, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        a[["f1", "f0"]] = [999]
        self.assertEqual(a.tolist(), [{"f0": 999, "f1": 999.0}, {"f0": 999, "f1": 999.0}, {"f0": 999, "f1": 999.0}, {"f0": 999, "f1": 999.0}, {"f0": 999, "f1": 999.0}, {"f0": 999, "f1": 999.0}, {"f0": 999, "f1": 999.0}, {"f0": 999, "f1": 999.0}, {"f0": 999, "f1": 999.0}, {"f0": 999, "f1": 999.0}])

        a = Table(10, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        a[["f1", "f0"]] = [123, 999]
        self.assertEqual(a.tolist(), [{"f0": 999, "f1": 123.0}, {"f0": 999, "f1": 123.0}, {"f0": 999, "f1": 123.0}, {"f0": 999, "f1": 123.0}, {"f0": 999, "f1": 123.0}, {"f0": 999, "f1": 123.0}, {"f0": 999, "f1": 123.0}, {"f0": 999, "f1": 123.0}, {"f0": 999, "f1": 123.0}, {"f0": 999, "f1": 123.0}])

        a = Table(10, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        a[[5, 3, 7, 5]] = 999
        self.assertEqual(a.tolist(), [{"f0": 0, "f1": 0.0}, {"f0": 1, "f1": 1.1}, {"f0": 2, "f1": 2.2}, {"f0": 999, "f1": 999.0}, {"f0": 4, "f1": 4.4}, {"f0": 999, "f1": 999.0}, {"f0": 6, "f1": 6.6}, {"f0": 999, "f1": 999.0}, {"f0": 8, "f1": 8.8}, {"f0": 9, "f1": 9.9}])

        a = Table(10, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        a[[5, 3, 7, 5]] = [999]
        self.assertEqual(a.tolist(), [{"f0": 0, "f1": 0.0}, {"f0": 1, "f1": 1.1}, {"f0": 2, "f1": 2.2}, {"f0": 999, "f1": 999.0}, {"f0": 4, "f1": 4.4}, {"f0": 999, "f1": 999.0}, {"f0": 6, "f1": 6.6}, {"f0": 999, "f1": 999.0}, {"f0": 8, "f1": 8.8}, {"f0": 9, "f1": 9.9}])

        a = Table(10, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        a[[5, 3, 7, 5]] = [999, 888, 777, 666]
        self.assertEqual(a.tolist(), [{"f0": 0, "f1": 0.0}, {"f0": 1, "f1": 1.1}, {"f0": 2, "f1": 2.2}, {"f0": 888, "f1": 888.0}, {"f0": 4, "f1": 4.4}, {"f0": 666, "f1": 666.0}, {"f0": 6, "f1": 6.6}, {"f0": 777, "f1": 777.0}, {"f0": 8, "f1": 8.8}, {"f0": 9, "f1": 9.9}])

    def test_indexed_table(self):
        a = Table(10, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        self.assertEqual(IndexedArray([5, 3, 7, 5], a)["f1"].tolist(), [5.5, 3.3, 7.7, 5.5])
        IndexedArray([5, 3, 7, 5], a)["f1"] = 999
        self.assertEqual(a.tolist(), [{"f0": 0, "f1": 0.0}, {"f0": 1, "f1": 1.1}, {"f0": 2, "f1": 2.2}, {"f0": 3, "f1": 999.0}, {"f0": 4, "f1": 4.4}, {"f0": 5, "f1": 999.0}, {"f0": 6, "f1": 6.6}, {"f0": 7, "f1": 999.0}, {"f0": 8, "f1": 8.8}, {"f0": 9, "f1": 9.9}])

    def test_masked_table(self):
        a = Table(10, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        self.assertEqual(MaskedArray([False, True, True, True, True, False, False, False, False, True], a, validwhen=True)["f1"].tolist(), [None, 1.1, 2.2, 3.3, 4.4, None, None, None, None, 9.9])
        MaskedArray([False, True, True, True, True, False, False, False, False, True], a)["f1"] = 999
        self.assertEqual(a.tolist(), [{"f0": 0, "f1": 999.0}, {"f0": 1, "f1": 999.0}, {"f0": 2, "f1": 999.0}, {"f0": 3, "f1": 999.0}, {"f0": 4, "f1": 999.0}, {"f0": 5, "f1": 999.0}, {"f0": 6, "f1": 999.0}, {"f0": 7, "f1": 999.0}, {"f0": 8, "f1": 999.0}, {"f0": 9, "f1": 999.0}])

    def test_jagged_table(self):
        a = Table(10, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        self.assertEqual(JaggedArray.fromoffsets([0, 3, 5, 5, 10], a).tolist(), [[{"f0": 0, "f1": 0.0}, {"f0": 1, "f1": 1.1}, {"f0": 2, "f1": 2.2}], [{"f0": 3, "f1": 3.3}, {"f0": 4, "f1": 4.4}], [], [{"f0": 5, "f1": 5.5}, {"f0": 6, "f1": 6.6}, {"f0": 7, "f1": 7.7}, {"f0": 8, "f1": 8.8}, {"f0": 9, "f1": 9.9}]])
        self.assertEqual(JaggedArray.fromoffsets([0, 3, 5, 5, 10], a)["f1"].tolist(), [[0.0, 1.1, 2.2], [3.3, 4.4], [], [5.5, 6.6, 7.7, 8.8, 9.9]])

        a = Table(10, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        JaggedArray.fromoffsets([0, 3, 5, 5, 10], a)["f1"][1] = 999.0
        self.assertEqual(a.tolist(), [{"f0": 0, "f1": 0.0}, {"f0": 1, "f1": 1.1}, {"f0": 2, "f1": 2.2}, {"f0": 3, "f1": 999.0}, {"f0": 4, "f1": 999.0}, {"f0": 5, "f1": 5.5}, {"f0": 6, "f1": 6.6}, {"f0": 7, "f1": 7.7}, {"f0": 8, "f1": 8.8}, {"f0": 9, "f1": 9.9}])

    def test_chunked_table(self):
        a = Table(4, [0, 1, 2, 3], [0.0, 1.1, 2.2, 3.3])
        b = Table(6, [4, 5, 6, 7, 8, 9], [4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
        c = ChunkedArray([a, b])
        self.assertEqual(c["f1"][6], 6.6)

        c["f1"][6] = 999
        self.assertEqual(c.tolist(), [{"f0": 0, "f1": 0.0}, {"f0": 1, "f1": 1.1}, {"f0": 2, "f1": 2.2}, {"f0": 3, "f1": 3.3}, {"f0": 4, "f1": 4.4}, {"f0": 5, "f1": 5.5}, {"f0": 6, "f1": 999.0}, {"f0": 7, "f1": 7.7}, {"f0": 8, "f1": 8.8}, {"f0": 9, "f1": 9.9}])

    def test_virtual_table(self):
        a = VirtualArray(lambda: Table(10, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
        self.assertEqual(a.tolist(), [{"f0": 0, "f1": 0.0}, {"f0": 1, "f1": 1.1}, {"f0": 2, "f1": 2.2}, {"f0": 3, "f1": 3.3}, {"f0": 4, "f1": 4.4}, {"f0": 5, "f1": 5.5}, {"f0": 6, "f1": 6.6}, {"f0": 7, "f1": 7.7}, {"f0": 8, "f1": 8.8}, {"f0": 9, "f1": 9.9}])
