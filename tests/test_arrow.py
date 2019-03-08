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

import os.path
import sys
import unittest

import pytest

import numpy
try:
    import pyarrow
    import pyarrow.parquet
except ImportError:
    pyarrow = None

import awkward.arrow
from awkward import *

class Test(unittest.TestCase):
    def runTest(self):
        pass

    def test_arrow_toarrow(self):
        if pyarrow is None:
            pytest.skip("unable to import pyarrow")
        else:
            try:
                import uproot_methods
            except ImportError:
                pytest.skip("unable to import uproot_methods")
            else:
                jet_m   = awkward.fromiter([[60.0, 70.0, 80.0], [], [90.0, 100.0]])
                jet_pt  = awkward.fromiter([[10.0, 20.0, 30.0], [], [40.0, 50.0]])
                jet_eta = awkward.fromiter([[-3.0, -2.0, 2.0],  [], [-1.0, 1.0]])
                jet_phi = awkward.fromiter([[-1.5,  0.0, 1.5],  [], [0.78, -0.78]])
                jets    = uproot_methods.TLorentzVectorArray.from_ptetaphim(jet_pt, jet_eta, jet_phi, jet_m)

                assert list(awkward.arrow.toarrow(jets)) == [[{"fPt": 10.0, "fEta": -3.0, "fPhi": -1.5, "fMass": 60.0}, {"fPt": 20.0, "fEta": -2.0, "fPhi": 0.0, "fMass": 70.0}, {"fPt": 30.0, "fEta": 2.0, "fPhi": 1.5, "fMass": 80.0}], [], [{"fPt": 40.0, "fEta": -1.0, "fPhi": 0.78, "fMass": 90.0}, {"fPt": 50.0, "fEta": 1.0, "fPhi": -0.78, "fMass": 100.0}]]

                # FIXME: it might be possible to avoid this "mask push-down" in Arrow, but I don't know how
                maskedjets = awkward.MaskedArray([False, False, True], jets, maskedwhen=True)
                assert list(awkward.arrow.toarrow(maskedjets)) == [[{"fPt": 10.0, "fEta": -3.0, "fPhi": -1.5, "fMass": 60.0}, {"fPt": 20.0, "fEta": -2.0, "fPhi": 0.0, "fMass": 70.0}, {"fPt": 30.0, "fEta": 2.0, "fPhi": 1.5, "fMass": 80.0}], [], [{"fPt": None, "fEta": None, "fPhi": None, "fMass": None}, {"fPt": None, "fEta": None, "fPhi": None, "fMass": None}]]

    def test_arrow_array(self):
        if pyarrow is None:
            pytest.skip("unable to import pyarrow")
        else:
            a = pyarrow.array([1.1, 2.2, 3.3, 4.4, 5.5])
            assert awkward.arrow.fromarrow(a).tolist() == [1.1, 2.2, 3.3, 4.4, 5.5]

    def test_arrow_boolean(self):
        if pyarrow is None:
            pytest.skip("unable to import pyarrow")
        else:
            a = pyarrow.array([True, True, False, False, True])
            assert awkward.arrow.fromarrow(a).tolist() == [True, True, False, False, True]

    def test_arrow_array_null(self):
        if pyarrow is None:
            pytest.skip("unable to import pyarrow")
        else:
            a = pyarrow.array([1.1, 2.2, 3.3, None, 4.4, 5.5])
            assert awkward.arrow.fromarrow(a).tolist() == [1.1, 2.2, 3.3, None, 4.4, 5.5]

    def test_arrow_nested_array(self):
        if pyarrow is None:
            pytest.skip("unable to import pyarrow")
        else:
            a = pyarrow.array([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
            assert awkward.arrow.fromarrow(a).tolist() == [[1.1, 2.2, 3.3], [], [4.4, 5.5]]

    def test_arrow_nested_nested_array(self):
        if pyarrow is None:
            pytest.skip("unable to import pyarrow")
        else:
            a = pyarrow.array([[[1.1, 2.2], [3.3], []], [], [[4.4, 5.5]]])
            assert awkward.arrow.fromarrow(a).tolist() == [[[1.1, 2.2], [3.3], []], [], [[4.4, 5.5]]]

    def test_arrow_nested_array_null(self):
        if pyarrow is None:
            pytest.skip("unable to import pyarrow")
        else:
            a = pyarrow.array([[1.1, 2.2, None], [], [4.4, 5.5]])
            assert awkward.arrow.fromarrow(a).tolist() == [[1.1, 2.2, None], [], [4.4, 5.5]]

    def test_arrow_null_nested_array_null(self):
        if pyarrow is None:
            pytest.skip("unable to import pyarrow")
        else:
            a = pyarrow.array([[1.1, 2.2, None], [], None, [4.4, 5.5]])
            assert awkward.arrow.fromarrow(a).tolist() == [[1.1, 2.2, None], [], None, [4.4, 5.5]]

    def test_arrow_chunked_array(self):
        if pyarrow is None:
            pytest.skip("unable to import pyarrow")
        else:
            a = pyarrow.chunked_array([pyarrow.array([1.1, 2.2, 3.3, 4.4, 5.5]), pyarrow.array([]), pyarrow.array([6.6, 7.7, 8.8])])
            assert awkward.arrow.fromarrow(a).tolist() == [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8]

    def test_arrow_struct(self):
        if pyarrow is None:
            pytest.skip("unable to import pyarrow")
        else:
            a = pyarrow.array([{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}])
            assert awkward.arrow.fromarrow(a).tolist() == [{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}]

    def test_arrow_struct_null(self):
        if pyarrow is None:
            pytest.skip("unable to import pyarrow")
        else:
            a = pyarrow.array([{"x": 1, "y": 1.1}, {"x": 2, "y": None}, {"x": 3, "y": 3.3}])
            assert awkward.arrow.fromarrow(a).tolist() == [{"x": 1, "y": 1.1}, {"x": 2, "y": None}, {"x": 3, "y": 3.3}]

    def test_arrow_null_struct(self):
        if pyarrow is None:
            pytest.skip("unable to import pyarrow")
        else:
            a = pyarrow.array([{"x": 1, "y": 1.1}, None, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}])
            assert awkward.arrow.fromarrow(a).tolist() == [{"x": 1, "y": 1.1}, None, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}]

    def test_arrow_null_struct_null(self):
        if pyarrow is None:
            pytest.skip("unable to import pyarrow")
        else:
            a = pyarrow.array([{"x": 1, "y": 1.1}, None, {"x": 2, "y": None}, {"x": 3, "y": 3.3}])
            assert awkward.arrow.fromarrow(a).tolist() == [{"x": 1, "y": 1.1}, None, {"x": 2, "y": None}, {"x": 3, "y": 3.3}]

    def test_arrow_chunked_struct(self):
        if pyarrow is None:
            pytest.skip("unable to import pyarrow")
        else:
            a = pyarrow.chunked_array([pyarrow.array([{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}]), pyarrow.array([]), pyarrow.array([{"x": 4, "y": 4.4}, {"x": 5, "y": 5.5}])])
            assert awkward.arrow.fromarrow(a).tolist() == [{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}, {"x": 4, "y": 4.4}, {"x": 5, "y": 5.5}]

    def test_arrow_nested_struct(self):
        if pyarrow is None:
            pytest.skip("unable to import pyarrow")
        else:
            a = pyarrow.array([[{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}], [], [{"x": 4, "y": 4.4}, {"x": 5, "y": 5.5}]])
            assert awkward.arrow.fromarrow(a).tolist() == [[{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}], [], [{"x": 4, "y": 4.4}, {"x": 5, "y": 5.5}]]

    def test_arrow_nested_struct_null(self):
        if pyarrow is None:
            pytest.skip("unable to import pyarrow")
        else:
            a = pyarrow.array([[{"x": 1, "y": 1.1}, {"x": 2, "y": None}, {"x": 3, "y": 3.3}], [], [{"x": 4, "y": 4.4}, {"x": 5, "y": 5.5}]])
            assert awkward.arrow.fromarrow(a).tolist() == [[{"x": 1, "y": 1.1}, {"x": 2, "y": None}, {"x": 3, "y": 3.3}], [], [{"x": 4, "y": 4.4}, {"x": 5, "y": 5.5}]]

    def test_arrow_null_nested_struct(self):
        if pyarrow is None:
            pytest.skip("unable to import pyarrow")
        else:
            a = pyarrow.array([[{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}], None, [], [{"x": 4, "y": 4.4}, {"x": 5, "y": 5.5}]])
            assert awkward.arrow.fromarrow(a).tolist() == [[{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}], None, [], [{"x": 4, "y": 4.4}, {"x": 5, "y": 5.5}]]

    def test_arrow_null_nested_struct_null(self):
        if pyarrow is None:
            pytest.skip("unable to import pyarrow")
        else:
            a = pyarrow.array([[{"x": 1, "y": 1.1}, {"x": 2, "y": None}, {"x": 3, "y": 3.3}], None, [], [{"x": 4, "y": 4.4}, {"x": 5, "y": 5.5}]])
            assert awkward.arrow.fromarrow(a).tolist() == [[{"x": 1, "y": 1.1}, {"x": 2, "y": None}, {"x": 3, "y": 3.3}], None, [], [{"x": 4, "y": 4.4}, {"x": 5, "y": 5.5}]]

    def test_arrow_struct_nested(self):
        if pyarrow is None:
            pytest.skip("unable to import pyarrow")
        else:
            a = pyarrow.array([{"x": [], "y": 1.1}, {"x": [2], "y": 2.2}, {"x": [3, 3], "y": 3.3}])
            assert awkward.arrow.fromarrow(a).tolist() == [{"x": [], "y": 1.1}, {"x": [2], "y": 2.2}, {"x": [3, 3], "y": 3.3}]

    def test_arrow_struct_nested_null(self):
        if pyarrow is None:
            pytest.skip("unable to import pyarrow")
        else:
            a = pyarrow.array([{"x": [], "y": 1.1}, {"x": [2], "y": 2.2}, {"x": [None, 3], "y": 3.3}])
            assert awkward.arrow.fromarrow(a).tolist() == [{"x": [], "y": 1.1}, {"x": [2], "y": 2.2}, {"x": [None, 3], "y": 3.3}]

    def test_arrow_nested_struct_nested(self):
        if pyarrow is None:
            pytest.skip("unable to import pyarrow")
        else:
            a = pyarrow.array([[{"x": [], "y": 1.1}, {"x": [2], "y": 2.2}, {"x": [3, 3], "y": 3.3}], [], [{"x": [4, 4, 4], "y": 4.4}, {"x": [5, 5, 5, 5], "y": 5.5}]])
            assert awkward.arrow.fromarrow(a).tolist() == [[{"x": [], "y": 1.1}, {"x": [2], "y": 2.2}, {"x": [3, 3], "y": 3.3}], [], [{"x": [4, 4, 4], "y": 4.4}, {"x": [5, 5, 5, 5], "y": 5.5}]]

    def test_arrow_null_nested_struct_nested_null(self):
        if pyarrow is None:
            pytest.skip("unable to import pyarrow")
        else:
            a = pyarrow.array([[{"x": [], "y": 1.1}, {"x": [2], "y": 2.2}, {"x": [None, 3], "y": 3.3}], None, [], [{"x": [4, 4, 4], "y": 4.4}, {"x": [5, 5, 5, 5], "y": 5.5}]])
            assert awkward.arrow.fromarrow(a).tolist() == [[{"x": [], "y": 1.1}, {"x": [2], "y": 2.2}, {"x": [None, 3], "y": 3.3}], None, [], [{"x": [4, 4, 4], "y": 4.4}, {"x": [5, 5, 5, 5], "y": 5.5}]]

    def test_arrow_strings(self):
        if pyarrow is None:
            pytest.skip("unable to import pyarrow")
        elif sys.version_info[0] < 3:
            pytest.skip("skipping strings test in Python 2")
        else:
            a = pyarrow.array(["one", "two", "three", u"fo\u2014ur", "five"])
            assert awkward.arrow.fromarrow(a).tolist() == ["one", "two", "three", u"fo\u2014ur", "five"]

    def test_arrow_strings_null(self):
        if pyarrow is None:
            pytest.skip("unable to import pyarrow")
        elif sys.version_info[0] < 3:
            pytest.skip("skipping strings test in Python 2")
        else:
            a = pyarrow.array(["one", "two", None, u"fo\u2014ur", "five"])
            assert awkward.arrow.fromarrow(a).tolist() == ["one", "two", None, u"fo\u2014ur", "five"]

    def test_arrow_binary(self):
        if pyarrow is None:
            pytest.skip("unable to import pyarrow")
        else:
            a = pyarrow.array([b"one", b"two", b"three", b"four", b"five"])
            assert awkward.arrow.fromarrow(a).tolist() == [b"one", b"two", b"three", b"four", b"five"]

    def test_arrow_binary_null(self):
        if pyarrow is None:
            pytest.skip("unable to import pyarrow")
        else:
            a = pyarrow.array([b"one", b"two", None, b"four", b"five"])
            assert awkward.arrow.fromarrow(a).tolist() == [b"one", b"two", None, b"four", b"five"]

    def test_arrow_chunked_strings(self):
        if pyarrow is None:
            pytest.skip("unable to import pyarrow")
        else:
            a = pyarrow.chunked_array([pyarrow.array(["one", "two", "three", "four", "five"]), pyarrow.array(["six", "seven", "eight"])])
            assert awkward.arrow.fromarrow(a).tolist() == ["one", "two", "three", "four", "five", "six", "seven", "eight"]

    def test_arrow_nested_strings(self):
        if pyarrow is None:
            pytest.skip("unable to import pyarrow")
        else:
            a = pyarrow.array([["one", "two", "three"], [], ["four", "five"]])
            assert awkward.arrow.fromarrow(a).tolist() == [["one", "two", "three"], [], ["four", "five"]]

    def test_arrow_nested_strings_null(self):
        if pyarrow is None:
            pytest.skip("unable to import pyarrow")
        else:
            a = pyarrow.array([["one", "two", None], [], ["four", "five"]])
            assert awkward.arrow.fromarrow(a).tolist() == [["one", "two", None], [], ["four", "five"]]

    def test_arrow_null_nested_strings_null(self):
        if pyarrow is None:
            pytest.skip("unable to import pyarrow")
        else:
            a = pyarrow.array([["one", "two", None], [], None, ["four", "five"]])
            assert awkward.arrow.fromarrow(a).tolist() == [["one", "two", None], [], None, ["four", "five"]]

    def test_arrow_union_sparse(self):
        if pyarrow is None:
            pytest.skip("unable to import pyarrow")
        else:
            a = pyarrow.UnionArray.from_sparse(pyarrow.array([0, 1, 0, 0, 1], type=pyarrow.int8()), [pyarrow.array([0.0, 1.1, 2.2, 3.3, 4.4]), pyarrow.array([True, True, False, True, False])])
            assert awkward.arrow.fromarrow(a).tolist() == [0.0, True, 2.2, 3.3, False]

    def test_arrow_union_sparse_null(self):
        if pyarrow is None:
            pytest.skip("unable to import pyarrow")
        else:
            a = pyarrow.UnionArray.from_sparse(pyarrow.array([0, 1, 0, 0, 1], type=pyarrow.int8()), [pyarrow.array([0.0, 1.1, None, 3.3, 4.4]), pyarrow.array([True, True, False, True, False])])
            assert awkward.arrow.fromarrow(a).tolist() == [0.0, True, None, 3.3, False]

    def test_arrow_union_sparse_null_null(self):
        if pyarrow is None:
            pytest.skip("unable to import pyarrow")
        else:
            a = pyarrow.UnionArray.from_sparse(pyarrow.array([0, 1, 0, 0, 1], type=pyarrow.int8()), [pyarrow.array([0.0, 1.1, None, 3.3, 4.4]), pyarrow.array([True, None, False, True, False])])
            assert awkward.arrow.fromarrow(a).tolist() == [0.0, None, None, 3.3, False]

    def test_arrow_union_dense(self):
        if pyarrow is None:
            pytest.skip("unable to import pyarrow")
        else:
            a = pyarrow.UnionArray.from_dense(pyarrow.array([0, 1, 0, 0, 0, 1, 1], type=pyarrow.int8()), pyarrow.array([0, 0, 1, 2, 3, 1, 2], type=pyarrow.int32()), [pyarrow.array([0.0, 1.1, 2.2, 3.3]), pyarrow.array([True, True, False])])
            assert awkward.arrow.fromarrow(a).tolist() == [0.0, True, 1.1, 2.2, 3.3, True, False]

    def test_arrow_union_dense_null(self):
        if pyarrow is None:
            pytest.skip("unable to import pyarrow")
        else:
            a = pyarrow.UnionArray.from_dense(pyarrow.array([0, 1, 0, 0, 0, 1, 1], type=pyarrow.int8()), pyarrow.array([0, 0, 1, 2, 3, 1, 2], type=pyarrow.int32()), [pyarrow.array([0.0, 1.1, None, 3.3]), pyarrow.array([True, True, False])])
            assert awkward.arrow.fromarrow(a).tolist() == [0.0, True, 1.1, None, 3.3, True, False]

    def test_arrow_union_dense_null_null(self):
        if pyarrow is None:
            pytest.skip("unable to import pyarrow")
        else:
            a = pyarrow.UnionArray.from_dense(pyarrow.array([0, 1, 0, 0, 0, 1, 1], type=pyarrow.int8()), pyarrow.array([0, 0, 1, 2, 3, 1, 2], type=pyarrow.int32()), [pyarrow.array([0.0, 1.1, None, 3.3]), pyarrow.array([True, None, False])])
            assert awkward.arrow.fromarrow(a).tolist() == [0.0, True, 1.1, None, 3.3, None, False]

    def test_arrow_dictarray(self):
        if pyarrow is None:
            pytest.skip("unable to import pyarrow")
        else:
            a = pyarrow.DictionaryArray.from_arrays(pyarrow.array([0, 0, 2, 2, 1, 0, 2, 1, 1]), pyarrow.array(["one", "two", "three"]))
            assert awkward.arrow.fromarrow(a).tolist() == ["one", "one", "three", "three", "two", "one", "three", "two", "two"]

    def test_arrow_dictarray_null(self):
        if pyarrow is None:
            pytest.skip("unable to import pyarrow")
        else:
            a = pyarrow.DictionaryArray.from_arrays(pyarrow.array([0, 0, 2, None, 1, None, 2, 1, 1]), pyarrow.array(["one", "two", "three"]))
            assert awkward.arrow.fromarrow(a).tolist() == ["one", "one", "three", None, "two", None, "three", "two", "two"]

    def test_arrow_null_dictarray(self):
        if pyarrow is None:
            pytest.skip("unable to import pyarrow")
        else:
            a = pyarrow.DictionaryArray.from_arrays(pyarrow.array([0, 0, 2, 2, 1, 0, 2, 1, 1]), pyarrow.array(["one", None, "three"]))
            assert awkward.arrow.fromarrow(a).tolist() == ["one", "one", "three", "three", None, "one", "three", None, None]

    def test_arrow_batch(self):
        if pyarrow is None:
            pytest.skip("unable to import pyarrow")
        else:
            a = pyarrow.RecordBatch.from_arrays(
                [pyarrow.array([1.1, 2.2, 3.3, None, 5.5]),
                 pyarrow.array([[1, 2, 3], [], [4, 5], [None], [6]]),
                 pyarrow.array([{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}, {"x": 4, "y": None}, {"x": 5, "y": 5.5}]),
                 pyarrow.array([{"x": 1, "y": 1.1}, None, None, {"x": 4, "y": None}, {"x": 5, "y": 5.5}]),
                 pyarrow.array([[{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}], [], [{"x": 4, "y": None}, {"x": 5, "y": 5.5}], [None], [{"x": 6, "y": 6.6}]])],
                ["a", "b", "c", "d", "e"])
            assert awkward.arrow.fromarrow(a).tolist() == [{"a": 1.1, "b": [1, 2, 3], "c": {"x": 1, "y": 1.1}, "d": {"x": 1, "y": 1.1}, "e": [{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}]}, {"a": 2.2, "b": [], "c": {"x": 2, "y": 2.2}, "d": None, "e": []}, {"a": 3.3, "b": [4, 5], "c": {"x": 3, "y": 3.3}, "d": None, "e": [{"x": 4, "y": None}, {"x": 5, "y": 5.5}]}, {"a": None, "b": [None], "c": {"x": 4, "y": None}, "d":{"x": 4, "y": None}, "e": [None]}, {"a": 5.5, "b": [6], "c": {"x": 5, "y": 5.5}, "d": {"x": 5, "y": 5.5}, "e": [{"x": 6, "y": 6.6}]}]

    def test_arrow_table(self):
        if pyarrow is None:
            pytest.skip("unable to import pyarrow")
        else:
            a = pyarrow.Table.from_batches([
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
            assert awkward.arrow.fromarrow(a).tolist() == [{"a": 1.1, "b": [1, 2, 3], "c": {"x": 1, "y": 1.1}, "d": {"x": 1, "y": 1.1}, "e": [{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}]}, {"a": 2.2, "b": [], "c": {"x": 2, "y": 2.2}, "d": None, "e": []}, {"a": 3.3, "b": [4, 5], "c": {"x": 3, "y": 3.3}, "d": None, "e": [{"x": 4, "y": None}, {"x": 5, "y": 5.5}]}, {"a": None, "b": [None], "c": {"x": 4, "y": None}, "d": {"x": 4, "y": None}, "e": [None]}, {"a": 5.5, "b": [6], "c": {"x": 5, "y": 5.5}, "d": {"x": 5, "y": 5.5}, "e": [{"x": 6, "y": 6.6}]}, {"a": 1.1, "b": [1, 2, 3], "c": {"x": 1, "y": 1.1}, "d": {"x": 1, "y": 1.1}, "e": [{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}]}, {"a": 2.2, "b": [], "c": {"x": 2, "y": 2.2}, "d": None, "e": []}, {"a": 3.3, "b": [4, 5], "c": {"x": 3, "y": 3.3}, "d": None, "e": [{"x": 4, "y": None}, {"x": 5, "y": 5.5}]}, {"a": None, "b": [None], "c": {"x": 4, "y": None}, "d": {"x": 4, "y": None}, "e": [None]}, {"a": 5.5, "b": [6], "c": {"x": 5, "y": 5.5}, "d": {"x": 5, "y": 5.5}, "e": [{"x": 6, "y": 6.6}]}]

    def test_arrow_nonnullable_table(self):
        if pyarrow is None:
            pytest.skip("unable to import pyarrow")
        else:
            x = pyarrow.array([1, 2, 3])
            y = pyarrow.array([1.1, 2.2, 3.3])
            table = pyarrow.Table.from_arrays([x], ["x"])
            table2 = table.add_column(1, pyarrow.column(pyarrow.field("y", y.type, False), numpy.array([1.1, 2.2, 3.3])))
            assert awkward.arrow.fromarrow(table2).tolist() == [{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}]

    # def test_arrow_writeparquet(self):
    #     if pyarrow is None:
    #         pytest.skip("unable to import pyarrow")
    #     else:
    #         a = pyarrow.Table.from_batches([
    #             pyarrow.RecordBatch.from_arrays(
    #             [pyarrow.array([1.1, 2.2, 3.3, None, 5.5]),
    #              pyarrow.array([[1, 2, 3], [], [None], None, [4, 5, 6]]),
    #              pyarrow.array([[[1.1, 2.2]], None, [[3.3, None], []], [], [None, [4.4, 5.5]]])],
    #             ["a", "b", "c"]),
    #             pyarrow.RecordBatch.from_arrays(
    #             [pyarrow.array([2.2, 1.1, 3.3, None, 5.5]),
    #              pyarrow.array([[2, 1, 3], [], [None], None, [4, 5, 6]]),
    #              pyarrow.array([[[2.2, 1.1]], None, [[3.3, None], []], [], [None, [4.4, 5.5]]])],
    #             ["a", "b", "c"])])
    #         writer = pyarrow.parquet.ParquetWriter("tests/samples/features-0_11_1.parquet", a.schema)
    #         writer.write_table(a)
    #         writer.write_table(a)
    #         writer.close()

    def test_arrow_readparquet(self):
        if pyarrow is None:
            pytest.skip("unable to import pyarrow")
        else:
            a = awkward.arrow.fromparquet("tests/samples/features-0_11_1.parquet", persistvirtual=True)
            assert a["a"].tolist() == [1.1, 2.2, 3.3, None, 5.5, 2.2, 1.1, 3.3, None, 5.5, 1.1, 2.2, 3.3, None, 5.5, 2.2, 1.1, 3.3, None, 5.5]
            storage = {}
            awkward.serialize(a, storage)
            b = awkward.deserialize(storage)
            assert b["b"].tolist() == [[1, 2, 3], [], [None], None, [4, 5, 6], [2, 1, 3], [], [None], None, [4, 5, 6], [1, 2, 3], [], [None], None, [4, 5, 6], [2, 1, 3], [], [None], None, [4, 5, 6]] 
            assert a["c"].tolist() == [[[1.1, 2.2]], None, [[3.3, None], []], [], [None, [4.4, 5.5]], [[2.2, 1.1]], None, [[3.3, None], []], [], [None, [4.4, 5.5]], [[1.1, 2.2]], None, [[3.3, None], []], [], [None, [4.4, 5.5]], [[2.2 , 1.1]], None, [[3.3, None], []], [], [None, [4.4, 5.5]]]

def test_arrow_writeparquet2(tmpdir):
    if pyarrow is None:
        pytest.skip("unable to import pyarrow")
    else:
        filename = os.path.join(str(tmpdir), "tmp.parquet")

    a = awkward.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
    b = awkward.fromiter([100, 200, None])

    awkward.toparquet([b, b, b], filename)
    assert awkward.fromparquet(filename).tolist() == [100, 200, None, 100, 200, None, 100, 200, None]

    awkward.toparquet(a, filename)
    assert awkward.fromparquet(filename).tolist() == [[1.1, 2.2, 3.3], [], [4.4, 5.5]]

    awkward.toparquet(awkward.Table(x=a, y=b), filename)
    c = awkward.fromparquet(filename)
    assert c.tolist() == [{'x': [1.1, 2.2, 3.3], 'y': 100}, {'x': [], 'y': 200}, {'x': [4.4, 5.5], 'y': None}]

    awkward.toparquet(c, filename)
    d = awkward.fromparquet(filename)
    assert c.tolist() == d.tolist()
    assert isinstance(c, awkward.ChunkedArray) and isinstance(d, awkward.ChunkedArray)
    assert len(c.chunks) == 1 and len(d.chunks) == 1
    assert isinstance(c.chunks[0], awkward.Table) and isinstance(d.chunks[0], awkward.Table)
    assert c.chunks[0].columns == d.chunks[0].columns
    assert isinstance(c.chunks[0]["x"], awkward.BitMaskedArray) and isinstance(d.chunks[0]["x"], awkward.BitMaskedArray)
    assert c.chunks[0]["x"].boolmask().tolist() == d.chunks[0]["x"].boolmask().tolist()
    assert isinstance(c.chunks[0]["x"].content, awkward.JaggedArray) and isinstance(d.chunks[0]["x"].content, awkward.JaggedArray)
    assert isinstance(c.chunks[0]["x"].content.content, awkward.BitMaskedArray) and isinstance(d.chunks[0]["x"].content.content, awkward.BitMaskedArray)
    assert c.chunks[0]["x"].content.content.boolmask().tolist() == d.chunks[0]["x"].content.content.boolmask().tolist()
    assert isinstance(c.chunks[0]["x"].content.content.content, numpy.ndarray) and isinstance(d.chunks[0]["x"].content.content.content, numpy.ndarray)
