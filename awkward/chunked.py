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

import numbers

import numpy

import awkward.base

class ChunkedArray(awkward.base.AwkwardArray):
    def __init__(self, chunks, writeable=True, appendable=True):
        self._appendable = appendable
        self.chunks = chunks
        self.writeable = writeable

    @property
    def chunks(self):
        return self._chunks

    @chunks.setter
    def chunks(self, value):
        try:
            iter(value)
        except TypeError:
            raise TypeError("chunks must be iterable")

        if self._appendable and not (hasattr(value, "append") and callable(value.append)):
            raise TypeError("because appendable=True, chunks must have an append method")

        self._chunks = value

    @property
    def writeable(self):
        return self._writeable

    @writeable.setter
    def writeable(self, value):
        self._writeable = bool(value)

    @property
    def dtype(self):
        chunk = None
        for chunk in self._chunks:
            break
        if chunk is None:
            raise ValueError("chunks is empty; cannot determine dtype")
        else:
            return numpy.dtype((chunk.dtype, chunk.shape[1:]))

    def __iter__(self):
        dtype = None
        for chunk in self._chunks:
            if dtype is None:
                dtype = chunk.dtype
            elif dtype != chunk.dtype:
                raise TypeError("chunk dtypes disagree: {0} versus {1}".format(dtype, chunk.dtype))

            for x in chunk:
                yield x

    def __str__(self):
        values = []
        for x in self:
            if len(values) == 7:
                return "[{0} ...]".format(" ".join(str(x) for x in values))
            values.append(x)
        return "[{0}]".format(" ".join(str(x) for x in values))

    def __getitem__(self, where):
        if not isinstance(where, tuple):
            where = (where,)
        head, rest = where[0], where[1:]

        if isinstance(head, (numbers.Integral, numpy.integer)):
            if head < 0:
                raise IndexError("negative indexes are not allowed because ChunkArray cannot determine total length")
            sofar = 0
            for chunk in self._chunks:
                if sofar <= head < sofar + len(chunk):
                    return chunk[(head - sofar,) + rest]
                sofar += len(chunk)
            raise IndexError("index {0} out of bounds for length {1}".format(head, sofar))

        elif isinstance(head, slice):
            start, stop, step = head.start, head.stop, head.step
            if start < 0 or stop < 0:
                raise IndexError("negative indexes are not allowed because ChunkArray cannot determine total length")

            out = []
            sofar = 0
            for chunk in self._chunks:
                if start is None:
                    localstart = 0
                elif start < sofar:
                    localstart = 0
                elif sofar <= start < sofar + len(chunk):
                    localstart = start - sofar
                else:
                    continue

                if stop is None:
                    localstop = len(chunk)
                elif stop <= sofar:
                    break
                elif sofar <= stop < sofar + len(chunk):
                    localstop = stop - sofar
                else:
                    localstop = len(chunk)

                out.append(chunk[localstart:localstop:step])
                sofar += len(chunk)

            if len(out) == 0:
                try:
                    dtype = self.dtype
                except ValueError:
                    dtype = numpy.dtype(numpy.void)
                return numpy.empty(0, dtype)

            elif len(out) == 1:
                return out[0]

            else:
                return numpy.concatenate(out)

        else:
            head = numpy.array(head, copy=False)
            if len(head.shape) == 1 and issubclass(head.dtype.type, numpy.integer):
                THREE

            elif len(head.shape) == 1 and issubclass(head.dtype.type, (numpy.bool, numpy.bool_)):
                FOUR

            else:
                raise TypeError("cannot interpret shape {0}, dtype {1} as a fancy index or mask".format(head.shape, head.dtype))

class PartitionedArray(ChunkedArray):
    pass
