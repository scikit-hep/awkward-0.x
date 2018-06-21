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

class TestMasked(unittest.TestCase):
    def runTest(self):
        pass

    def test_masked_get(self):
        a = MaskedArray([True, False, True, False, True, False, True, False, True, False], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], validwhen=False)
        self.assertEqual(a.tolist(), [None, 1.1, None, 3.3, None, 5.5, None, 7.7, None, 9.9])
        self.assertTrue(numpy.ma.is_masked(a[0]))
        self.assertFalse(numpy.ma.is_masked(a[1]))
        self.assertEqual(a[5:].tolist(), [5.5, None, 7.7, None, 9.9])
        self.assertFalse(numpy.ma.is_masked(a[5:][0]))
        self.assertTrue(numpy.ma.is_masked(a[5:][1]))
        self.assertEqual(a[[3, 2, 1]].tolist(), [3.3, None, 1.1])
        self.assertEqual(a[[True, True, True, True, True, False, False, False, False, False]].tolist(), [None, 1.1, None, 3.3, None])

    def test_masked_set(self):
        a = MaskedArray([True, False, True, False, True, False, True, False, True, False], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], validwhen=False)
        a[5] = 999
        self.assertEqual(a.tolist(), [None, 1.1, None, 3.3, None, 999.0, None, 7.7, None, 9.9])

        a = MaskedArray([True, False, True, False, True, False, True, False, True, False], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], validwhen=False)
        a[6] = 999
        self.assertEqual(a.tolist(), [None, 1.1, None, 3.3, None, 5.5, 999.0, 7.7, None, 9.9])

        a = MaskedArray([True, False, True, False, True, False, True, False, True, False], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], validwhen=False)
        a[5] = numpy.ma.masked
        self.assertEqual(a.tolist(), [None, 1.1, None, 3.3, None, None, None, 7.7, None, 9.9])

        a = MaskedArray([True, False, True, False, True, False, True, False, True, False], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], validwhen=False)
        a[6] = numpy.ma.masked
        self.assertEqual(a.tolist(), [None, 1.1, None, 3.3, None, 5.5, None, 7.7, None, 9.9])

        a = MaskedArray([True, False, True, False, True, False, True, False, True, False], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], validwhen=False)
        a[5:] = 999
        self.assertEqual(a.tolist(), [None, 1.1, None, 3.3, None, 999.0, 999.0, 999.0, 999.0, 999.0])

        a = MaskedArray([True, False, True, False, True, False, True, False, True, False], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], validwhen=False)
        a[5:] = [999]
        self.assertEqual(a.tolist(), [None, 1.1, None, 3.3, None, 999.0, 999.0, 999.0, 999.0, 999.0])

        a = MaskedArray([True, False, True, False, True, False, True, False, True, False], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], validwhen=False)
        a[5:] = [1, 2, 3, 4, 5]
        self.assertEqual(a.tolist(), [None, 1.1, None, 3.3, None, 1.0, 2.0, 3.0, 4.0, 5.0])

        a = MaskedArray([True, False, True, False, True, False, True, False, True, False], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], validwhen=False)
        a[5:] = numpy.ma.masked
        self.assertEqual(a.tolist(), [None, 1.1, None, 3.3, None, None, None, None, None, None])

        a = MaskedArray([True, False, True, False, True, False, True, False, True, False], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], validwhen=False)
        a[[3, 2, 1]] = [1, 2, 3]
        self.assertEqual(a.tolist(), [None, 3.0, 2.0, 1.0, None, 5.5, None, 7.7, None, 9.9])

        a = MaskedArray([True, False, True, False, True, False, True, False, True, False], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], validwhen=False)
        a[[3, 2, 1]] = [1, 2, numpy.ma.masked]
        self.assertEqual(a.tolist(), [None, None, 2.0, 1.0, None, 5.5, None, 7.7, None, 9.9])

        a = MaskedArray([True, False, True, False, True, False, True, False, True, False], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], validwhen=False)
        a[[True, True, True, True, True, False, False, False, False, False]] = [101, 102, 103, 104, 105]
        self.assertEqual(a.tolist(), [101.0, 102.0, 103.0, 104.0, 105.0, 5.5, None, 7.7, None, 9.9])

        a = MaskedArray([True, False, True, False, True, False, True, False, True, False], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], validwhen=False)
        a[[True, True, True, True, True, False, False, False, False, False]] = [101, 102, numpy.ma.masked, 104, 105]
        self.assertEqual(a.tolist(), [101.0, 102.0, None, 104.0, 105.0, 5.5, None, 7.7, None, 9.9])

        a = MaskedArray([True, False, True, False, True, False, True, False, True, False], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], validwhen=False)
        a[[3, 2, 1]] = MaskedArray([False, False, True], [1, 2, 3], validwhen=False)
        self.assertEqual(a.tolist(), [None, None, 2.0, 1.0, None, 5.5, None, 7.7, None, 9.9])

        a = MaskedArray([True, False, True, False, True, False, True, False, True, False], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], validwhen=False)
        a[[3, 2, 1]] = MaskedArray([True, True, False], [1, 2, 3], validwhen=True)
        self.assertEqual(a.tolist(), [None, None, 2.0, 1.0, None, 5.5, None, 7.7, None, 9.9])

    def test_masked_get_flip(self):
        a = MaskedArray([False, True, False, True, False, True, False, True, False, True], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], validwhen=True)
        self.assertEqual(a.tolist(), [None, 1.1, None, 3.3, None, 5.5, None, 7.7, None, 9.9])
        self.assertTrue(numpy.ma.is_masked(a[0]))
        self.assertFalse(numpy.ma.is_masked(a[1]))
        self.assertEqual(a[5:].tolist(), [5.5, None, 7.7, None, 9.9])
        self.assertFalse(numpy.ma.is_masked(a[5:][0]))
        self.assertTrue(numpy.ma.is_masked(a[5:][1]))
        self.assertEqual(a[[3, 2, 1]].tolist(), [3.3, None, 1.1])
        self.assertEqual(a[[True, True, True, True, True, False, False, False, False, False]].tolist(), [None, 1.1, None, 3.3, None])

    def test_masked_set_flip(self):
        a = MaskedArray([False, True, False, True, False, True, False, True, False, True], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], validwhen=True)
        a[5] = 999
        self.assertEqual(a.tolist(), [None, 1.1, None, 3.3, None, 999.0, None, 7.7, None, 9.9])

        a = MaskedArray([False, True, False, True, False, True, False, True, False, True], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], validwhen=True)
        a[6] = 999
        self.assertEqual(a.tolist(), [None, 1.1, None, 3.3, None, 5.5, 999.0, 7.7, None, 9.9])

        a = MaskedArray([False, True, False, True, False, True, False, True, False, True], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], validwhen=True)
        a[5] = numpy.ma.masked
        self.assertEqual(a.tolist(), [None, 1.1, None, 3.3, None, None, None, 7.7, None, 9.9])

        a = MaskedArray([False, True, False, True, False, True, False, True, False, True], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], validwhen=True)
        a[6] = numpy.ma.masked
        self.assertEqual(a.tolist(), [None, 1.1, None, 3.3, None, 5.5, None, 7.7, None, 9.9])

        a = MaskedArray([False, True, False, True, False, True, False, True, False, True], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], validwhen=True)
        a[5:] = 999
        self.assertEqual(a.tolist(), [None, 1.1, None, 3.3, None, 999.0, 999.0, 999.0, 999.0, 999.0])

        a = MaskedArray([False, True, False, True, False, True, False, True, False, True], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], validwhen=True)
        a[5:] = [999]
        self.assertEqual(a.tolist(), [None, 1.1, None, 3.3, None, 999.0, 999.0, 999.0, 999.0, 999.0])

        a = MaskedArray([False, True, False, True, False, True, False, True, False, True], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], validwhen=True)
        a[5:] = [1, 2, 3, 4, 5]
        self.assertEqual(a.tolist(), [None, 1.1, None, 3.3, None, 1.0, 2.0, 3.0, 4.0, 5.0])

        a = MaskedArray([False, True, False, True, False, True, False, True, False, True], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], validwhen=True)
        a[5:] = numpy.ma.masked
        self.assertEqual(a.tolist(), [None, 1.1, None, 3.3, None, None, None, None, None, None])

        a = MaskedArray([False, True, False, True, False, True, False, True, False, True], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], validwhen=True)
        a[[3, 2, 1]] = [1, 2, 3]
        self.assertEqual(a.tolist(), [None, 3.0, 2.0, 1.0, None, 5.5, None, 7.7, None, 9.9])

        a = MaskedArray([False, True, False, True, False, True, False, True, False, True], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], validwhen=True)
        a[[3, 2, 1]] = [1, 2, numpy.ma.masked]
        self.assertEqual(a.tolist(), [None, None, 2.0, 1.0, None, 5.5, None, 7.7, None, 9.9])

        a = MaskedArray([False, True, False, True, False, True, False, True, False, True], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], validwhen=True)
        a[[True, True, True, True, True, False, False, False, False, False]] = [101, 102, 103, 104, 105]
        self.assertEqual(a.tolist(), [101.0, 102.0, 103.0, 104.0, 105.0, 5.5, None, 7.7, None, 9.9])

        a = MaskedArray([False, True, False, True, False, True, False, True, False, True], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], validwhen=True)
        a[[True, True, True, True, True, False, False, False, False, False]] = [101, 102, numpy.ma.masked, 104, 105]
        self.assertEqual(a.tolist(), [101.0, 102.0, None, 104.0, 105.0, 5.5, None, 7.7, None, 9.9])

        a = MaskedArray([False, True, False, True, False, True, False, True, False, True], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], validwhen=True)
        a[[3, 2, 1]] = MaskedArray([False, False, True], [1, 2, 3], validwhen=False)
        self.assertEqual(a.tolist(), [None, None, 2.0, 1.0, None, 5.5, None, 7.7, None, 9.9])

        a = MaskedArray([False, True, False, True, False, True, False, True, False, True], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], validwhen=True)
        a[[3, 2, 1]] = MaskedArray([True, True, False], [1, 2, 3], validwhen=True)
        self.assertEqual(a.tolist(), [None, None, 2.0, 1.0, None, 5.5, None, 7.7, None, 9.9])

    def test_bitmasked_get(self):
        a = BitMaskedArray.fromboolmask([True, False, True, False, True, False, True, False, True, False], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], validwhen=False, lsb=True)
        self.assertEqual(a.tolist(), [None, 1.1, None, 3.3, None, 5.5, None, 7.7, None, 9.9])
        self.assertTrue(numpy.ma.is_masked(a[0]))
        self.assertFalse(numpy.ma.is_masked(a[1]))
        self.assertEqual(a[5:].tolist(), [5.5, None, 7.7, None, 9.9])
        self.assertFalse(numpy.ma.is_masked(a[5:][0]))
        self.assertTrue(numpy.ma.is_masked(a[5:][1]))
        self.assertEqual(a[[3, 2, 1]].tolist(), [3.3, None, 1.1])
        self.assertEqual(a[[True, True, True, True, True, False, False, False, False, False]].tolist(), [None, 1.1, None, 3.3, None])

        a = BitMaskedArray.fromboolmask([True, False, True, False, True, False, True, False, True, False], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], validwhen=False, lsb=False)
        self.assertEqual(a.tolist(), [None, 1.1, None, 3.3, None, 5.5, None, 7.7, None, 9.9])
        self.assertTrue(numpy.ma.is_masked(a[0]))
        self.assertFalse(numpy.ma.is_masked(a[1]))
        self.assertEqual(a[5:].tolist(), [5.5, None, 7.7, None, 9.9])
        self.assertFalse(numpy.ma.is_masked(a[5:][0]))
        self.assertTrue(numpy.ma.is_masked(a[5:][1]))
        self.assertEqual(a[[3, 2, 1]].tolist(), [3.3, None, 1.1])
        self.assertEqual(a[[True, True, True, True, True, False, False, False, False, False]].tolist(), [None, 1.1, None, 3.3, None])

    def test_bitmasked_get_flip(self):
        a = BitMaskedArray.fromboolmask([False, True, False, True, False, True, False, True, False, True], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], validwhen=True, lsb=True)
        self.assertEqual(a.tolist(), [None, 1.1, None, 3.3, None, 5.5, None, 7.7, None, 9.9])
        self.assertTrue(numpy.ma.is_masked(a[0]))
        self.assertFalse(numpy.ma.is_masked(a[1]))
        self.assertEqual(a[5:].tolist(), [5.5, None, 7.7, None, 9.9])
        self.assertFalse(numpy.ma.is_masked(a[5:][0]))
        self.assertTrue(numpy.ma.is_masked(a[5:][1]))
        self.assertEqual(a[[3, 2, 1]].tolist(), [3.3, None, 1.1])
        self.assertEqual(a[[True, True, True, True, True, False, False, False, False, False]].tolist(), [None, 1.1, None, 3.3, None])

        a = BitMaskedArray.fromboolmask([False, True, False, True, False, True, False, True, False, True], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], validwhen=True, lsb=False)
        self.assertEqual(a.tolist(), [None, 1.1, None, 3.3, None, 5.5, None, 7.7, None, 9.9])
        self.assertTrue(numpy.ma.is_masked(a[0]))
        self.assertFalse(numpy.ma.is_masked(a[1]))
        self.assertEqual(a[5:].tolist(), [5.5, None, 7.7, None, 9.9])
        self.assertFalse(numpy.ma.is_masked(a[5:][0]))
        self.assertTrue(numpy.ma.is_masked(a[5:][1]))
        self.assertEqual(a[[3, 2, 1]].tolist(), [3.3, None, 1.1])
        self.assertEqual(a[[True, True, True, True, True, False, False, False, False, False]].tolist(), [None, 1.1, None, 3.3, None])

    def test_bitmasked_arrow(self):
        # Apache Arrow layout example
        # https://github.com/apache/arrow/blob/master/format/Layout.md#null-bitmaps
        a = BitMaskedArray.fromboolmask([True, True, False, True, False, True], [0, 1, 999, 2, 999, 3], validwhen=True, lsb=True)
        self.assertEqual(a.tolist(), [0, 1, None, 2, None, 3])

        # extra gunk at the end of the array
        a = BitMaskedArray.fromboolmask([True, True, False, True, False, True, True, True], [0, 1, 999, 2, 999, 3], validwhen=True, lsb=True)
        self.assertEqual(a.tolist(), [0, 1, None, 2, None, 3])

        # opposite sign
        a = BitMaskedArray.fromboolmask([True, True, False, True, False, True, False, False], [0, 1, 999, 2, 999, 3], validwhen=True, lsb=True)
        self.assertEqual(a.tolist(), [0, 1, None, 2, None, 3])

        # doubled
        a = BitMaskedArray.fromboolmask([True, True, False, True, False, True, True, True, False, True, False, True], [0, 1, 999, 2, 999, 3, 0, 1, 999, 2, 999, 3], validwhen=True, lsb=True)
        self.assertEqual(a.tolist(), [0, 1, None, 2, None, 3, 0, 1, None, 2, None, 3])

    def test_bitmasked_set(self):
        a = BitMaskedArray.fromboolmask([True, False, True, False, True, False, True, False, True, False], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], validwhen=False, lsb=True)
        a[5] = 999
        self.assertEqual(a.tolist(), [None, 1.1, None, 3.3, None, 999.0, None, 7.7, None, 9.9])

        a = BitMaskedArray.fromboolmask([True, False, True, False, True, False, True, False, True, False], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], validwhen=False, lsb=True)
        a[6] = 999
        self.assertEqual(a.tolist(), [None, 1.1, None, 3.3, None, 5.5, 999.0, 7.7, None, 9.9])

        a = BitMaskedArray.fromboolmask([True, False, True, False, True, False, True, False, True, False], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], validwhen=False, lsb=True)
        a[5] = numpy.ma.masked
        self.assertEqual(a.tolist(), [None, 1.1, None, 3.3, None, None, None, 7.7, None, 9.9])

        a = BitMaskedArray.fromboolmask([True, False, True, False, True, False, True, False, True, False], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], validwhen=False, lsb=True)
        a[6] = numpy.ma.masked
        self.assertEqual(a.tolist(), [None, 1.1, None, 3.3, None, 5.5, None, 7.7, None, 9.9])

        a = BitMaskedArray.fromboolmask([True, False, True, False, True, False, True, False, True, False], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], validwhen=False, lsb=True)
        a[5:] = 999
        self.assertEqual(a.tolist(), [None, 1.1, None, 3.3, None, 999.0, 999.0, 999.0, 999.0, 999.0])

        a = BitMaskedArray.fromboolmask([True, False, True, False, True, False, True, False, True, False], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], validwhen=False, lsb=True)
        a[5:] = [999]
        self.assertEqual(a.tolist(), [None, 1.1, None, 3.3, None, 999.0, 999.0, 999.0, 999.0, 999.0])

        a = BitMaskedArray.fromboolmask([True, False, True, False, True, False, True, False, True, False], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], validwhen=False, lsb=True)
        a[5:] = [1, 2, 3, 4, 5]
        self.assertEqual(a.tolist(), [None, 1.1, None, 3.3, None, 1.0, 2.0, 3.0, 4.0, 5.0])

        a = BitMaskedArray.fromboolmask([True, False, True, False, True, False, True, False, True, False], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], validwhen=False, lsb=True)
        a[5:] = numpy.ma.masked
        self.assertEqual(a.tolist(), [None, 1.1, None, 3.3, None, None, None, None, None, None])

        a = BitMaskedArray.fromboolmask([True, False, True, False, True, False, True, False, True, False], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], validwhen=False, lsb=True)
        a[[3, 2, 1]] = [1, 2, 3]
        self.assertEqual(a.tolist(), [None, 3.0, 2.0, 1.0, None, 5.5, None, 7.7, None, 9.9])

        a = BitMaskedArray.fromboolmask([True, False, True, False, True, False, True, False, True, False], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], validwhen=False, lsb=True)
        a[[3, 2, 1]] = [1, 2, numpy.ma.masked]
        self.assertEqual(a.tolist(), [None, None, 2.0, 1.0, None, 5.5, None, 7.7, None, 9.9])

        a = BitMaskedArray.fromboolmask([True, False, True, False, True, False, True, False, True, False], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], validwhen=False, lsb=True)
        a[[True, True, True, True, True, False, False, False, False, False]] = [101, 102, 103, 104, 105]
        self.assertEqual(a.tolist(), [101.0, 102.0, 103.0, 104.0, 105.0, 5.5, None, 7.7, None, 9.9])

        a = BitMaskedArray.fromboolmask([True, False, True, False, True, False, True, False, True, False], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], validwhen=False, lsb=True)
        a[[True, True, True, True, True, False, False, False, False, False]] = [101, 102, numpy.ma.masked, 104, 105]
        self.assertEqual(a.tolist(), [101.0, 102.0, None, 104.0, 105.0, 5.5, None, 7.7, None, 9.9])

        a = BitMaskedArray.fromboolmask([True, False, True, False, True, False, True, False, True, False], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], validwhen=False, lsb=True)
        a[[3, 2, 1]] = BitMaskedArray.fromboolmask([False, False, True], [1, 2, 3], validwhen=False, lsb=True)
        self.assertEqual(a.tolist(), [None, None, 2.0, 1.0, None, 5.5, None, 7.7, None, 9.9])

        a = BitMaskedArray.fromboolmask([True, False, True, False, True, False, True, False, True, False], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], validwhen=False, lsb=True)
        a[[3, 2, 1]] = BitMaskedArray.fromboolmask([True, True, False], [1, 2, 3], validwhen=True, lsb=True)
        self.assertEqual(a.tolist(), [None, None, 2.0, 1.0, None, 5.5, None, 7.7, None, 9.9])

        a = BitMaskedArray.fromboolmask([True, False, True, False, True, False, True, False, True, False], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], validwhen=False, lsb=True)
        a[[3, 2, 1]] = BitMaskedArray.fromboolmask([False, False, True], [1, 2, 3], validwhen=False, lsb=False)
        self.assertEqual(a.tolist(), [None, None, 2.0, 1.0, None, 5.5, None, 7.7, None, 9.9])

        a = BitMaskedArray.fromboolmask([True, False, True, False, True, False, True, False, True, False], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], validwhen=False, lsb=True)
        a[[3, 2, 1]] = BitMaskedArray.fromboolmask([True, True, False], [1, 2, 3], validwhen=True, lsb=False)
        self.assertEqual(a.tolist(), [None, None, 2.0, 1.0, None, 5.5, None, 7.7, None, 9.9])

    def test_bitmasked_set_lsb(self):
        a = BitMaskedArray.fromboolmask([True, False, True, False, True, False, True, False, True, False], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], validwhen=False, lsb=False)
        a[5] = 999
        self.assertEqual(a.tolist(), [None, 1.1, None, 3.3, None, 999.0, None, 7.7, None, 9.9])

        a = BitMaskedArray.fromboolmask([True, False, True, False, True, False, True, False, True, False], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], validwhen=False, lsb=False)
        a[6] = 999
        self.assertEqual(a.tolist(), [None, 1.1, None, 3.3, None, 5.5, 999.0, 7.7, None, 9.9])

        a = BitMaskedArray.fromboolmask([True, False, True, False, True, False, True, False, True, False], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], validwhen=False, lsb=False)
        a[5] = numpy.ma.masked
        self.assertEqual(a.tolist(), [None, 1.1, None, 3.3, None, None, None, 7.7, None, 9.9])

        a = BitMaskedArray.fromboolmask([True, False, True, False, True, False, True, False, True, False], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], validwhen=False, lsb=False)
        a[6] = numpy.ma.masked
        self.assertEqual(a.tolist(), [None, 1.1, None, 3.3, None, 5.5, None, 7.7, None, 9.9])

        a = BitMaskedArray.fromboolmask([True, False, True, False, True, False, True, False, True, False], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], validwhen=False, lsb=False)
        a[5:] = 999
        self.assertEqual(a.tolist(), [None, 1.1, None, 3.3, None, 999.0, 999.0, 999.0, 999.0, 999.0])

        a = BitMaskedArray.fromboolmask([True, False, True, False, True, False, True, False, True, False], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], validwhen=False, lsb=False)
        a[5:] = [999]
        self.assertEqual(a.tolist(), [None, 1.1, None, 3.3, None, 999.0, 999.0, 999.0, 999.0, 999.0])

        a = BitMaskedArray.fromboolmask([True, False, True, False, True, False, True, False, True, False], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], validwhen=False, lsb=False)
        a[5:] = [1, 2, 3, 4, 5]
        self.assertEqual(a.tolist(), [None, 1.1, None, 3.3, None, 1.0, 2.0, 3.0, 4.0, 5.0])

        a = BitMaskedArray.fromboolmask([True, False, True, False, True, False, True, False, True, False], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], validwhen=False, lsb=False)
        a[5:] = numpy.ma.masked
        self.assertEqual(a.tolist(), [None, 1.1, None, 3.3, None, None, None, None, None, None])

        a = BitMaskedArray.fromboolmask([True, False, True, False, True, False, True, False, True, False], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], validwhen=False, lsb=False)
        a[[3, 2, 1]] = [1, 2, 3]
        self.assertEqual(a.tolist(), [None, 3.0, 2.0, 1.0, None, 5.5, None, 7.7, None, 9.9])

        a = BitMaskedArray.fromboolmask([True, False, True, False, True, False, True, False, True, False], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], validwhen=False, lsb=False)
        a[[3, 2, 1]] = [1, 2, numpy.ma.masked]
        self.assertEqual(a.tolist(), [None, None, 2.0, 1.0, None, 5.5, None, 7.7, None, 9.9])

        a = BitMaskedArray.fromboolmask([True, False, True, False, True, False, True, False, True, False], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], validwhen=False, lsb=False)
        a[[True, True, True, True, True, False, False, False, False, False]] = [101, 102, 103, 104, 105]
        self.assertEqual(a.tolist(), [101.0, 102.0, 103.0, 104.0, 105.0, 5.5, None, 7.7, None, 9.9])

        a = BitMaskedArray.fromboolmask([True, False, True, False, True, False, True, False, True, False], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], validwhen=False, lsb=False)
        a[[True, True, True, True, True, False, False, False, False, False]] = [101, 102, numpy.ma.masked, 104, 105]
        self.assertEqual(a.tolist(), [101.0, 102.0, None, 104.0, 105.0, 5.5, None, 7.7, None, 9.9])

        a = BitMaskedArray.fromboolmask([True, False, True, False, True, False, True, False, True, False], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], validwhen=False, lsb=False)
        a[[3, 2, 1]] = BitMaskedArray.fromboolmask([False, False, True], [1, 2, 3], validwhen=False, lsb=True)
        self.assertEqual(a.tolist(), [None, None, 2.0, 1.0, None, 5.5, None, 7.7, None, 9.9])

        a = BitMaskedArray.fromboolmask([True, False, True, False, True, False, True, False, True, False], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], validwhen=False, lsb=False)
        a[[3, 2, 1]] = BitMaskedArray.fromboolmask([True, True, False], [1, 2, 3], validwhen=True, lsb=True)
        self.assertEqual(a.tolist(), [None, None, 2.0, 1.0, None, 5.5, None, 7.7, None, 9.9])

        a = BitMaskedArray.fromboolmask([True, False, True, False, True, False, True, False, True, False], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], validwhen=False, lsb=False)
        a[[3, 2, 1]] = BitMaskedArray.fromboolmask([False, False, True], [1, 2, 3], validwhen=False, lsb=False)
        self.assertEqual(a.tolist(), [None, None, 2.0, 1.0, None, 5.5, None, 7.7, None, 9.9])

        a = BitMaskedArray.fromboolmask([True, False, True, False, True, False, True, False, True, False], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], validwhen=False, lsb=False)
        a[[3, 2, 1]] = BitMaskedArray.fromboolmask([True, True, False], [1, 2, 3], validwhen=True, lsb=False)
        self.assertEqual(a.tolist(), [None, None, 2.0, 1.0, None, 5.5, None, 7.7, None, 9.9])

    def test_bitmasked_set_flip(self):
        a = BitMaskedArray.fromboolmask([False, True, False, True, False, True, False, True, False, True], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], validwhen=True, lsb=True)
        a[5] = 999
        self.assertEqual(a.tolist(), [None, 1.1, None, 3.3, None, 999.0, None, 7.7, None, 9.9])

        a = BitMaskedArray.fromboolmask([False, True, False, True, False, True, False, True, False, True], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], validwhen=True, lsb=True)
        a[6] = 999
        self.assertEqual(a.tolist(), [None, 1.1, None, 3.3, None, 5.5, 999.0, 7.7, None, 9.9])

        a = BitMaskedArray.fromboolmask([False, True, False, True, False, True, False, True, False, True], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], validwhen=True, lsb=True)
        a[5] = numpy.ma.masked
        self.assertEqual(a.tolist(), [None, 1.1, None, 3.3, None, None, None, 7.7, None, 9.9])

        a = BitMaskedArray.fromboolmask([False, True, False, True, False, True, False, True, False, True], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], validwhen=True, lsb=True)
        a[6] = numpy.ma.masked
        self.assertEqual(a.tolist(), [None, 1.1, None, 3.3, None, 5.5, None, 7.7, None, 9.9])

        a = BitMaskedArray.fromboolmask([False, True, False, True, False, True, False, True, False, True], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], validwhen=True, lsb=True)
        a[5:] = 999
        self.assertEqual(a.tolist(), [None, 1.1, None, 3.3, None, 999.0, 999.0, 999.0, 999.0, 999.0])

        a = BitMaskedArray.fromboolmask([False, True, False, True, False, True, False, True, False, True], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], validwhen=True, lsb=True)
        a[5:] = [999]
        self.assertEqual(a.tolist(), [None, 1.1, None, 3.3, None, 999.0, 999.0, 999.0, 999.0, 999.0])

        a = BitMaskedArray.fromboolmask([False, True, False, True, False, True, False, True, False, True], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], validwhen=True, lsb=True)
        a[5:] = [1, 2, 3, 4, 5]
        self.assertEqual(a.tolist(), [None, 1.1, None, 3.3, None, 1.0, 2.0, 3.0, 4.0, 5.0])

        a = BitMaskedArray.fromboolmask([False, True, False, True, False, True, False, True, False, True], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], validwhen=True, lsb=True)
        a[5:] = numpy.ma.masked
        self.assertEqual(a.tolist(), [None, 1.1, None, 3.3, None, None, None, None, None, None])

        a = BitMaskedArray.fromboolmask([False, True, False, True, False, True, False, True, False, True], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], validwhen=True, lsb=True)
        a[[3, 2, 1]] = [1, 2, 3]
        self.assertEqual(a.tolist(), [None, 3.0, 2.0, 1.0, None, 5.5, None, 7.7, None, 9.9])

        a = BitMaskedArray.fromboolmask([False, True, False, True, False, True, False, True, False, True], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], validwhen=True, lsb=True)
        a[[3, 2, 1]] = [1, 2, numpy.ma.masked]
        self.assertEqual(a.tolist(), [None, None, 2.0, 1.0, None, 5.5, None, 7.7, None, 9.9])

        a = BitMaskedArray.fromboolmask([False, True, False, True, False, True, False, True, False, True], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], validwhen=True, lsb=True)
        a[[True, True, True, True, True, False, False, False, False, False]] = [101, 102, 103, 104, 105]
        self.assertEqual(a.tolist(), [101.0, 102.0, 103.0, 104.0, 105.0, 5.5, None, 7.7, None, 9.9])

        a = BitMaskedArray.fromboolmask([False, True, False, True, False, True, False, True, False, True], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], validwhen=True, lsb=True)
        a[[True, True, True, True, True, False, False, False, False, False]] = [101, 102, numpy.ma.masked, 104, 105]
        self.assertEqual(a.tolist(), [101.0, 102.0, None, 104.0, 105.0, 5.5, None, 7.7, None, 9.9])

        a = BitMaskedArray.fromboolmask([False, True, False, True, False, True, False, True, False, True], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], validwhen=True, lsb=True)
        a[[3, 2, 1]] = BitMaskedArray.fromboolmask([False, False, True], [1, 2, 3], validwhen=False, lsb=True)
        self.assertEqual(a.tolist(), [None, None, 2.0, 1.0, None, 5.5, None, 7.7, None, 9.9])

        a = BitMaskedArray.fromboolmask([False, True, False, True, False, True, False, True, False, True], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], validwhen=True, lsb=True)
        a[[3, 2, 1]] = BitMaskedArray.fromboolmask([True, True, False], [1, 2, 3], validwhen=True, lsb=True)
        self.assertEqual(a.tolist(), [None, None, 2.0, 1.0, None, 5.5, None, 7.7, None, 9.9])

        a = BitMaskedArray.fromboolmask([False, True, False, True, False, True, False, True, False, True], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], validwhen=True, lsb=True)
        a[[3, 2, 1]] = BitMaskedArray.fromboolmask([False, False, True], [1, 2, 3], validwhen=False, lsb=False)
        self.assertEqual(a.tolist(), [None, None, 2.0, 1.0, None, 5.5, None, 7.7, None, 9.9])

        a = BitMaskedArray.fromboolmask([False, True, False, True, False, True, False, True, False, True], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], validwhen=True, lsb=True)
        a[[3, 2, 1]] = BitMaskedArray.fromboolmask([True, True, False], [1, 2, 3], validwhen=True, lsb=False)
        self.assertEqual(a.tolist(), [None, None, 2.0, 1.0, None, 5.5, None, 7.7, None, 9.9])

    def test_bitmasked_set_lsb_flip(self):
        a = BitMaskedArray.fromboolmask([False, True, False, True, False, True, False, True, False, True], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], validwhen=True, lsb=False)
        a[5] = 999
        self.assertEqual(a.tolist(), [None, 1.1, None, 3.3, None, 999.0, None, 7.7, None, 9.9])

        a = BitMaskedArray.fromboolmask([False, True, False, True, False, True, False, True, False, True], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], validwhen=True, lsb=False)
        a[6] = 999
        self.assertEqual(a.tolist(), [None, 1.1, None, 3.3, None, 5.5, 999.0, 7.7, None, 9.9])

        a = BitMaskedArray.fromboolmask([False, True, False, True, False, True, False, True, False, True], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], validwhen=True, lsb=False)
        a[5] = numpy.ma.masked
        self.assertEqual(a.tolist(), [None, 1.1, None, 3.3, None, None, None, 7.7, None, 9.9])

        a = BitMaskedArray.fromboolmask([False, True, False, True, False, True, False, True, False, True], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], validwhen=True, lsb=False)
        a[6] = numpy.ma.masked
        self.assertEqual(a.tolist(), [None, 1.1, None, 3.3, None, 5.5, None, 7.7, None, 9.9])

        a = BitMaskedArray.fromboolmask([False, True, False, True, False, True, False, True, False, True], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], validwhen=True, lsb=False)
        a[5:] = 999
        self.assertEqual(a.tolist(), [None, 1.1, None, 3.3, None, 999.0, 999.0, 999.0, 999.0, 999.0])

        a = BitMaskedArray.fromboolmask([False, True, False, True, False, True, False, True, False, True], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], validwhen=True, lsb=False)
        a[5:] = [999]
        self.assertEqual(a.tolist(), [None, 1.1, None, 3.3, None, 999.0, 999.0, 999.0, 999.0, 999.0])

        a = BitMaskedArray.fromboolmask([False, True, False, True, False, True, False, True, False, True], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], validwhen=True, lsb=False)
        a[5:] = [1, 2, 3, 4, 5]
        self.assertEqual(a.tolist(), [None, 1.1, None, 3.3, None, 1.0, 2.0, 3.0, 4.0, 5.0])

        a = BitMaskedArray.fromboolmask([False, True, False, True, False, True, False, True, False, True], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], validwhen=True, lsb=False)
        a[5:] = numpy.ma.masked
        self.assertEqual(a.tolist(), [None, 1.1, None, 3.3, None, None, None, None, None, None])

        a = BitMaskedArray.fromboolmask([False, True, False, True, False, True, False, True, False, True], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], validwhen=True, lsb=False)
        a[[3, 2, 1]] = [1, 2, 3]
        self.assertEqual(a.tolist(), [None, 3.0, 2.0, 1.0, None, 5.5, None, 7.7, None, 9.9])

        a = BitMaskedArray.fromboolmask([False, True, False, True, False, True, False, True, False, True], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], validwhen=True, lsb=False)
        a[[3, 2, 1]] = [1, 2, numpy.ma.masked]
        self.assertEqual(a.tolist(), [None, None, 2.0, 1.0, None, 5.5, None, 7.7, None, 9.9])

        a = BitMaskedArray.fromboolmask([False, True, False, True, False, True, False, True, False, True], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], validwhen=True, lsb=False)
        a[[True, True, True, True, True, False, False, False, False, False]] = [101, 102, 103, 104, 105]
        self.assertEqual(a.tolist(), [101.0, 102.0, 103.0, 104.0, 105.0, 5.5, None, 7.7, None, 9.9])

        a = BitMaskedArray.fromboolmask([False, True, False, True, False, True, False, True, False, True], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], validwhen=True, lsb=False)
        a[[True, True, True, True, True, False, False, False, False, False]] = [101, 102, numpy.ma.masked, 104, 105]
        self.assertEqual(a.tolist(), [101.0, 102.0, None, 104.0, 105.0, 5.5, None, 7.7, None, 9.9])

        a = BitMaskedArray.fromboolmask([False, True, False, True, False, True, False, True, False, True], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], validwhen=True, lsb=False)
        a[[3, 2, 1]] = BitMaskedArray.fromboolmask([False, False, True], [1, 2, 3], validwhen=False, lsb=True)
        self.assertEqual(a.tolist(), [None, None, 2.0, 1.0, None, 5.5, None, 7.7, None, 9.9])

        a = BitMaskedArray.fromboolmask([False, True, False, True, False, True, False, True, False, True], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], validwhen=True, lsb=False)
        a[[3, 2, 1]] = BitMaskedArray.fromboolmask([True, True, False], [1, 2, 3], validwhen=True, lsb=True)
        self.assertEqual(a.tolist(), [None, None, 2.0, 1.0, None, 5.5, None, 7.7, None, 9.9])

        a = BitMaskedArray.fromboolmask([False, True, False, True, False, True, False, True, False, True], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], validwhen=True, lsb=False)
        a[[3, 2, 1]] = BitMaskedArray.fromboolmask([False, False, True], [1, 2, 3], validwhen=False, lsb=False)
        self.assertEqual(a.tolist(), [None, None, 2.0, 1.0, None, 5.5, None, 7.7, None, 9.9])

        a = BitMaskedArray.fromboolmask([False, True, False, True, False, True, False, True, False, True], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], validwhen=True, lsb=False)
        a[[3, 2, 1]] = BitMaskedArray.fromboolmask([True, True, False], [1, 2, 3], validwhen=True, lsb=False)
        self.assertEqual(a.tolist(), [None, None, 2.0, 1.0, None, 5.5, None, 7.7, None, 9.9])
