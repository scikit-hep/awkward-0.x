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
try:
    __import__('pkg_resources').declare_namespace(__name__)
except ImportError:
    __path__ = __import__('pkgutil').extend_path(__path__, __name__)

from awkward.array.chunked import ChunkedArray
from awkward.array.chunked import AppendableArray
from awkward.array.indexed import IndexedArray, SparseArray
from awkward_numba.array.jagged import JaggedArray   #from awkward_numba.array.jagged import JaggedArrayNumba as JaggedArray
from awkward.array.masked import MaskedArray, BitMaskedArray, IndexedMaskedArray
from awkward.array.objects import Methods, ObjectArray, StringArray
from awkward.array.table import Table
from awkward.array.union import UnionArray
from awkward.array.virtual import VirtualArray
import numba
import numpy as np
import awkward

__all__ = ["ChunkedArray", "AppendableArray", "IndexedArray", "SparseArray", "JaggedArray", "MaskedArray", "BitMaskedArray", "IndexedMaskedArray", "Methods", "ObjectArray", "Table", "UnionArray", "VirtualArray", "StringArray", "fromiter", "fromiterchunks", "serialize", "deserialize", "save", "load", "hdf5", "__version__"]

@numba.jit
def _argminmax_general_new(self, ismin):
    if len(self._content.shape) != 1:
        raise ValueError("cannot compute arg{0} because content is not one-dimensional".format("min" if ismin else "max"))

    if ismin:
        optimum = awkward.util.numpy.argmin
    else:
        optimum = awkward.util.numpy.argmax

    out = awkward.util.numpy.empty(self._starts.shape + self._content.shape[1:], dtype=self.INDEXTYPE)

    flatout = out.reshape((-1,) + self._content.shape[1:])
    flatstarts = self._starts.reshape(-1)
    flatstops = self._stops.reshape(-1)

    content = self._content
    for i, flatstart in enumerate(flatstarts):
        flatstop = flatstops[i]
        if flatstart != flatstop:
            flatout[i] = optimum(content[flatstart:flatstop], axis=0)
    newstarts = awkward.util.numpy.arange(len(flatstarts), dtype=self.INDEXTYPE).reshape(self._starts.shape)
    newstops = awkward.util.numpy.array(newstarts)
    newstops.reshape(-1)[flatstarts != flatstops] += 1
    return self.copy(starts=newstarts, stops=newstops, content=flatout)


JaggedArray._argminmax_general = _argminmax_general_new