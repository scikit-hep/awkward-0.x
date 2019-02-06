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

import awkward.array.jagged
from .base import NumbaMethods

@numba.njit(["void(i8[:], i8[:])"])
def _offsets2parents_fill(offsets, parents):
    j = 0
    k = -1
    for i in offsets:
        while j < i:
            parents[j] = k
            j += 1
        k += 1

@numba.njit(["void(i8[:], i8[:], f8[:], i8[:])",
             "void(i8[:], i8[:], f4[:], i8[:])",
             "void(i8[:], i8[:], i8[:], i8[:])"])
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

@numba.njit(["void(i8[:], i8[:], f8[:], i8[:])",
             "void(i8[:], i8[:], f4[:], i8[:])",
             "void(i8[:], i8[:], i8[:], i8[:])"])
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
