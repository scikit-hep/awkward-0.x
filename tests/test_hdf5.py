#!/usr/bin/env python

# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-array/blob/master/LICENSE

import pytest
import numpy

import awkward

h5py = pytest.importorskip("h5py")

array_norm = [[1.1, 2.2, 3.3], [], [4.4, 5.5]]
array_int = [[1, 2, 3], [], [4, 5]]
array_f32 = [numpy.array([1.3,3.2], dtype=numpy.float32), [], numpy.array([7.6], dtype=numpy.float32)]
array_lg = [numpy.array([1,2,3])]*10000

@pytest.mark.parametrize("input_arr", (array_norm, array_int, array_f32, array_lg))
def test_read_write_hdf(tmpdir, input_arr):
    tmp_file = tmpdir / "example.h5"

    # Write
    with h5py.File(str(tmp_file), "w") as hf:
        a = awkward.JaggedArray.fromiter(input_arr)
        ah5 = awkward.hdf5(hf)
        ah5["example"] = a

    # Read
    with h5py.File(str(tmp_file), "r") as hf:
        ah5 = awkward.hdf5(hf)
        b = ah5["example"]

    assert a.tolist() == b.tolist()
