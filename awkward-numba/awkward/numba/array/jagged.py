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

import numba
import numba.typing.arraydecl

import awkward.array.jagged
from .base import NumbaMethods
from .base import AwkwardArrayType
from .base import clsrepr
from .base import ISADVANCED
from .base import NOTADVANCED

######################################################################## optimized functions (hidden)

@numba.njit
def _offsets2parents_fill(offsets, parents):
    j = 0
    k = -1
    for i in offsets:
        while j < i:
            parents[j] = k
            j += 1
        k += 1

@numba.njit
def _argminmax_fillmin(starts, stops, content, output):
    k = 0
    for i in range(len(starts)):
        if stops[i] != starts[i]:
            best = content[starts[i]]
            bestj = 0
            for j in range(starts[i] + 1, stops[i]):
                if content[j] < best:
                    best = content[j]
                    bestj = j - starts[i]
            output[k] = bestj
            k += 1

@numba.njit
def _argminmax_fillmax(starts, stops, content, output):
    k = 0
    for i in range(len(starts)):
        if stops[i] != starts[i]:
            best = content[starts[i]]
            bestj = 0
            for j in range(starts[i] + 1, stops[i]):
                if content[j] > best:
                    best = content[j]
                    bestj = j - starts[i]
            output[k] = bestj
            k += 1

######################################################################## Numba-accelerated interface

class JaggedArrayNumba(NumbaMethods, awkward.array.jagged.JaggedArray):
    @classmethod
    def offsets2parents(cls, offsets):
        if len(offsets) == 0:
            raise ValueError("offsets must have at least one element")
        parents = cls.numpy.empty(offsets[-1], dtype=cls.JaggedArray.fget(None).INDEXTYPE)
        _offsets2parents_fill(offsets, parents)
        return parents

    def _argminmax(self, ismin):
        if len(self._starts) == len(self._stops) == 0:
            return self.copy()

        if len(self._content.shape) != 1:
            raise ValueError("cannot compute arg{0} because content is not one-dimensional".format("min" if ismin else "max"))

        # subarray with counts > 0 --> counts = 1
        counts = (self.counts != 0).astype(self.INDEXTYPE).reshape(-1)
        # offsets for these 0 or 1 counts (involves a cumsum)
        offsets = self.counts2offsets(counts)
        # starts and stops derived from offsets and reshaped to original starts and stops (see specification)
        starts, stops = offsets[:-1], offsets[1:]

        starts = starts.reshape(self._starts.shape[:-1] + (-1,))
        stops = stops.reshape(self._stops.shape[:-1] + (-1,))

        # content to fit the new offsets
        content = awkward.util.numpy.empty(offsets[-1], dtype=self.INDEXTYPE)

        # fill the new content
        if ismin:
            _argminmax_fillmin(self._starts.reshape(-1), self._stops.reshape(-1), self._content, content)
        else:
            _argminmax_fillmax(self._starts.reshape(-1), self._stops.reshape(-1), self._content, content)

        return self.copy(starts=starts, stops=stops, content=content)

    def _argminmax_general(self, ismin):
        raise RuntimeError("helper function not needed in JaggedArrayNumba")

######################################################################## register types in Numba

class JaggedArrayType(AwkwardArrayType):
    def __init__(self, starts, stops, content, special=awkward.array.jagged.JaggedArray):
        super(JaggedArrayType, self).__init__(name="JaggedArrayType({0}, {1}, {2}{3})".format(starts.name, stops.name, content.name, "" if special is awkward.array.jagged.JaggedArray else clsrepr(special)))
        if starts.ndim != stops.ndim:
            raise ValueError("JaggedArray.starts must have the same number of dimensions as JaggedArray.stops")
        self.starts = starts
        self.stops = stops
        self.content = content
        self.special = special

    def getitem(self, where):
        head = numba.types.Tuple(where.types[0])
        if isinstance(head, numba.types.Integer) and isinstance(self.content, numba.types.Array):
            return numba.typing.arraydecl.get_array_index_type(self.content, tail).result

        fake = _JaggedArray_getitem_typer(JaggedArrayType(numba.types.int64[:], numba.types.int64[:], self), where, NOTADVANCED)
        if isinstance(fake, numba.types.Array):
            return fake.dtype
        else:
            return fake.content

def _JaggedArray_getitem_typer(array, where, advanced):
    if len(where.types) == 0:
        return array

    assert array.starts.ndim == array.stops.ndim == 1

    isarray = (isinstance(where.types[0], numba.types.Array) and where.types[0].ndim == 1)

    content = _JaggedArray_getitem_typer(array.content, numba.types.Tuple(where.types[1:]), ISADVANCED if isarray else advanced)

    if isinstance(where.types[0], numba.types.Integer) or (advanced == ISADVANCED and isarray):
        return content
    elif isinstance(where.types[0], numba.types.SliceType) or (advanced == NOTADVANCED and isarray):
        return JaggedArrayType(array.starts, array.stops, content, special=array.special)
    else:
        raise TypeError("cannot be used for indexing: {0}".format(where))





######################################################################## model and boxing

######################################################################## lower methods in Numba
