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
try:
    import pyarrow
except ImportError:
    pyarrow = None

from awkward import *

class Test(unittest.TestCase):
    def runTest(self):
        pass

    def test_array(self):
        if pyarrow is not None:
            arr = pyarrow.array([1.1, 2.2, 3.3, 4.4, 5.5])

    def test_boolean(self):
        if pyarrow is not None:
            arr = pyarrow.array([True, True, False, False, True])

    def test_array_null(self):
        if pyarrow is not None:
            arr = pyarrow.array([1.1, 2.2, 3.3, None, 4.4, 5.5])

    def test_nested_array(self):
        if pyarrow is not None:
            arr = pyarrow.array([[1.1, 2.2, 3.3], [], [4.4, 5.5]])

    def test_nested_array_null(self):
        if pyarrow is not None:
            arr = pyarrow.array([[1.1, 2.2, None], [], [4.4, 5.5]])

    def test_null_nested_array_null(self):
        if pyarrow is not None:
            arr = pyarrow.array([[1.1, 2.2, None], [], None, [4.4, 5.5]])

    def test_chunked_array(self):
        if pyarrow is not None:
            arr = pyarrow.chunked_array([pyarrow.array([1.1, 2.2, 3.3, 4.4, 5.5]), pyarrow.array([]), pyarrow.array([6.6, 7.7, 8.8])])

    def test_struct(self):
        if pyarrow is not None:
            arr = pyarrow.array([{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}])

    def test_struct_null(self):
        if pyarrow is not None:
            arr = pyarrow.array([{"x": 1, "y": 1.1}, {"x": 2, "y": None}, {"x": 3, "y": 3.3}])

    def test_null_struct(self):
        if pyarrow is not None:
            arr = pyarrow.array([{"x": 1, "y": 1.1}, None, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}])

    def test_null_struct_null(self):
        if pyarrow is not None:
            arr = pyarrow.array([{"x": 1, "y": 1.1}, None, {"x": 2, "y": None}, {"x": 3, "y": 3.3}])

    def test_chunked_struct(self):
        if pyarrow is not None:
            arr = pyarrow.chunked_array([pyarrow.array([{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}]), pyarrow.array([]), pyarrow.array([{"x": 4, "y": 4.4}, {"x": 5, "y": 5.5}])])

    def test_nested_struct(self):
        if pyarrow is not None:
            arr = pyarrow.array([[{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}], [], [{"x": 4, "y": 4.4}, {"x": 5, "y": 5.5}]])

    def test_nested_struct_null(self):
        if pyarrow is not None:
            arr = pyarrow.array([[{"x": 1, "y": 1.1}, {"x": 2, "y": None}, {"x": 3, "y": 3.3}], [], [{"x": 4, "y": 4.4}, {"x": 5, "y": 5.5}]])

    def test_null_nested_struct(self):
        if pyarrow is not None:
            arr = pyarrow.array([[{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}], None, [], [{"x": 4, "y": 4.4}, {"x": 5, "y": 5.5}]])

    def test_null_nested_struct_null(self):
        if pyarrow is not None:
            arr = pyarrow.array([[{"x": 1, "y": 1.1}, {"x": 2, "y": None}, {"x": 3, "y": 3.3}], None, [], [{"x": 4, "y": 4.4}, {"x": 5, "y": 5.5}]])

    def test_struct_nested(self):
        if pyarrow is not None:
            arr = pyarrow.array([{"x": [], "y": 1.1}, {"x": [2], "y": 2.2}, {"x": [3, 3], "y": 3.3}])

    def test_struct_nested_null(self):
        if pyarrow is not None:
            arr = pyarrow.array([{"x": [], "y": 1.1}, {"x": [2], "y": 2.2}, {"x": [None, 3], "y": 3.3}])

    def test_nested_struct_nested(self):
        if pyarrow is not None:
            arr = pyarrow.array([[{"x": [], "y": 1.1}, {"x": [2], "y": 2.2}, {"x": [3, 3], "y": 3.3}], [], [{"x": [4, 4, 4], "y": 4.4}, {"x": [5, 5, 5, 5], "y": 5.5}]])

    def test_null_nested_struct_nested_null(self):
        if pyarrow is not None:
            arr = pyarrow.array([[{"x": [], "y": 1.1}, {"x": [2], "y": 2.2}, {"x": [None, 3], "y": 3.3}], None, [], [{"x": [4, 4, 4], "y": 4.4}, {"x": [5, 5, 5, 5], "y": 5.5}]])

    def test_strings(self):
        if pyarrow is not None:
            arr = pyarrow.array(["one", "two", "three", u"fo\u2014ur", "five"])

    def test_strings_null(self):
        if pyarrow is not None:
            arr = pyarrow.array(["one", "two", None, u"fo\u2014ur", "five"])

    def test_binary(self):
        if pyarrow is not None:
            arr = pyarrow.array([b"one", b"two", b"three", b"four", b"five"])

    def test_binary_null(self):
        if pyarrow is not None:
            arr = pyarrow.array([b"one", b"two", None, b"four", b"five"])

    def test_chunked_strings(self):
        if pyarrow is not None:
            arr = pyarrow.chunked_array([pyarrow.array(["one", "two", "three", "four", "five"]), pyarrow.array(["six", "seven", "eight"])])

    def test_nested_strings(self):
        if pyarrow is not None:
            arr = pyarrow.array([["one", "two", "three"], [], ["four", "five"]])

    def test_nested_strings_null(self):
        if pyarrow is not None:
            arr = pyarrow.array([["one", "two", None], [], ["four", "five"]])

    def test_null_nested_strings_null(self):
        if pyarrow is not None:
            arr = pyarrow.array([["one", "two", None], [], None, ["four", "five"]])

    def test_union_sparse(self):
        if pyarrow is not None:
            arr = pyarrow.UnionArray.from_sparse(pyarrow.array([0, 1, 0, 0, 1], type=pyarrow.int8()), [pyarrow.array([0.0, 1.1, 2.2, 3.3, 4.4]), pyarrow.array([True, True, False, True, False])])

    def test_union_sparse_null(self):
        if pyarrow is not None:
            arr = pyarrow.UnionArray.from_sparse(pyarrow.array([0, 1, 0, 0, 1], type=pyarrow.int8()), [pyarrow.array([0.0, 1.1, None, 3.3, 4.4]), pyarrow.array([True, True, False, True, False])])

    def test_union_sparse_null_null(self):
        if pyarrow is not None:
            arr = pyarrow.UnionArray.from_sparse(pyarrow.array([0, 1, 0, 0, 1], type=pyarrow.int8()), [pyarrow.array([0.0, 1.1, None, 3.3, 4.4]), pyarrow.array([True, None, False, True, False])])

    def test_union_dense(self):
        if pyarrow is not None:
            arr = pyarrow.UnionArray.from_dense(pyarrow.array([0, 1, 0, 0, 0, 1, 1], type=pyarrow.int8()), pyarrow.array([0, 0, 1, 2, 3, 1, 2], type=pyarrow.int32()), [pyarrow.array([0.0, 1.1, 2.2, 3.3]), pyarrow.array([True, True, False])])

    def test_union_dense_null(self):
        if pyarrow is not None:
            arr = pyarrow.UnionArray.from_dense(pyarrow.array([0, 1, 0, 0, 0, 1, 1], type=pyarrow.int8()), pyarrow.array([0, 0, 1, 2, 3, 1, 2], type=pyarrow.int32()), [pyarrow.array([0.0, 1.1, None, 3.3]), pyarrow.array([True, True, False])])

    def test_union_dense_null_null(self):
        if pyarrow is not None:
            arr = pyarrow.UnionArray.from_dense(pyarrow.array([0, 1, 0, 0, 0, 1, 1], type=pyarrow.int8()), pyarrow.array([0, 0, 1, 2, 3, 1, 2], type=pyarrow.int32()), [pyarrow.array([0.0, 1.1, None, 3.3]), pyarrow.array([True, None, False])])

    def test_dictarray(self):
        if pyarrow is not None:
            arr = pyarrow.DictionaryArray.from_arrays(pyarrow.array([0, 0, 2, 2, 1, 0, 2, 1, 1]), pyarrow.array(["one", "two", "three"]))

    def test_dictarray_null(self):
        if pyarrow is not None:
            arr = pyarrow.DictionaryArray.from_arrays(pyarrow.array([0, 0, 2, None, 1, None, 2, 1, 1]), pyarrow.array(["one", "two", "three"]))

    def test_null_dictarray(self):
        if pyarrow is not None:
            arr = pyarrow.DictionaryArray.from_arrays(pyarrow.array([0, 0, 2, 2, 1, 0, 2, 1, 1]), pyarrow.array(["one", None, "three"]))

    def test_batch(self):
        if pyarrow is not None:
            arr = pyarrow.RecordBatch.from_arrays(
                [pyarrow.array([1.1, 2.2, 3.3, None, 5.5]),
                 pyarrow.array([[1, 2, 3], [], [4, 5], [None], [6]]),
                 pyarrow.array([{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}, {"x": 4, "y": None}, {"x": 5, "y": 5.5}]),
                 pyarrow.array([{"x": 1, "y": 1.1}, None, None, {"x": 4, "y": None}, {"x": 5, "y": 5.5}]),
                 pyarrow.array([[{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}], [], [{"x": 4, "y": None}, {"x": 5, "y": 5.5}], [None], [{"x": 6, "y": 6.6}]])],
                ["a", "b", "c", "d", "e"])

    def test_table(self):
        if pyarrow is not None:
            arr = pyarrow.Table.from_batches([
                pyarrow.RecordBatch.from_arrays(
                [pyarrow.array([1.1, 2.2, 3.3, None, 5.5]),
                 pyarrow.array([[1, 2, 3], [], [4, 5], [None], [6]]),
                 pyarrow.array([{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}, {"x": 4, "y": None}, {"x": 5, "y": 5.5}]),
                 pyarrow.array([{"x": 1, "y": 1.1}, None, None, {"x": 4, "y": None}, {"x": 5, "y": 5.5}]),
                 pyarrow.array([[{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}], [], [{"x": 4, "y": None}, {"x": 5, "y": 5.5}], [None], [{"x": 6, "y": 6.6}]])],
                ["a", "b", "c", "d", "e"]),
                pyarrow.RecordBatch.from_arrays(
                [pyarrow.array([1.1, 2.2, 3.3, None, 5.5]),
                 pyarrow.array([[1, 2, 3], [], [4, 5], [None], [6]]),
                 pyarrow.array([{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}, {"x": 4, "y": None}, {"x": 5, "y": 5.5}]),
                 pyarrow.array([{"x": 1, "y": 1.1}, None, None, {"x": 4, "y": None}, {"x": 5, "y": 5.5}]),
                 pyarrow.array([[{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}], [], [{"x": 4, "y": None}, {"x": 5, "y": 5.5}], [None], [{"x": 6, "y": 6.6}]])],
                ["a", "b", "c", "d", "e"])])
