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
    with h5py.File(str(tmp_file)) as hf:
        ah5 = awkward.hdf5(hf)
        b = ah5["example"]

    assert a.tolist() == b.tolist()
