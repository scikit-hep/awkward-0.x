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

import collections
import numbers
import itertools

import numpy

import awkward.array.base
import awkward.util

class ChunkedArray(awkward.array.base.AwkwardArray):
    def __init__(self, chunks, writeable=True):
        self.chunks = chunks
        self.writeable = writeable

    @property
    def chunks(self):
        return self._chunks

    @chunks.setter
    def chunks(self, value):
        try:
            self._chunks = list(value)
        except TypeError:
            raise TypeError("chunks must be iterable")

    @property
    def writeable(self):
        return self._writeable

    @writeable.setter
    def writeable(self, value):
        self._writeable = bool(value)

    def _chunkiterator(self, minindex):
        dtype = None

        sofar = i = 0
        while i < len(self._chunks):
            if not isinstance(self._chunks[i], (numpy.ndarray, awkward.array.base.AwkwardArray)):
                self._chunks[i] = self._toarray(self._chunks[i], self.CHARTYPE, (numpy.ndarray, awkward.array.base.AwkwardArray))

            if len(self._chunks[i]) != 0:
                thisdtype = numpy.dtype((self._chunks[i].dtype, self._chunks[i].shape[1:]))
                if dtype is None:
                    dtype = thisdtype
                elif dtype != thisdtype:
                    raise ValueError("chunk starting at index {0} has dtype {1}, different from {2}".format(sofar, thisdtype, dtype))

            if sofar + len(self._chunks[i]) > minindex:
                yield sofar, self._chunks[i]

            sofar += len(self._chunks[i])
            i += 1

    @property
    def dtype(self):
        for sofar, chunk in self._chunkiterator(0):
            if len(chunk) != 0:
                return numpy.dtype((chunk.dtype, chunk.shape[1:]))
        raise ValueError("chunks are empty; cannot determine dtype")

    @property
    def dimension(self):
        try:
            return self.dtype.shape
        except ValueError:
            raise ValueError("chunks are empty; cannot determine dimension")

    def __len__(self):
        return sum(len(chunk) for chunk in self._chunks)

    @property
    def shape(self):
        return (len(self),) + self.dimension

    def topartitioned(self):
        offsets = [0]
        for chunk in self._chunks:
            offsets.append(offsets[-1] + len(chunk))
        return PartitionedArray(offsets, self._chunks, writeable=self._writeable)

    def __iter__(self):
        i = 0
        while i < len(self._chunks):
            if not isinstance(self._chunks[i], (numpy.ndarray, awkward.array.base.AwkwardArray)):
                self._chunks[i] = self._toarray(self._chunks[i], self.CHARTYPE, (numpy.ndarray, awkward.array.base.AwkwardArray))
            for x in self._chunks[i]:
                yield x
            i += 1

    def __str__(self):
        values = []
        for x in self:
            if len(values) == 7:
                return "[{0} ...]".format(" ".join(str(x) for x in values))
            values.append(x)
        return "[{0}]".format(" ".join(str(x) for x in values))

    def _slicedchunks(self, start, stop, step, tail):
        if step == 0:
            raise ValueError("slice step cannot be zero")
        elif step is None:
            step = 1

        slicedchunks = []
        localstep = 1 if step > 0 else -1

        if step is None or step > 0:
            if start is None:
                minindex = 0
            else:
                minindex = start
        else:
            if stop is None:
                minindex = 0
            else:
                minindex = stop + 1

        for sofar, chunk in self._chunkiterator(minindex):
            if len(chunk) == 0:
                continue

            if step > 0:
                if start is None:
                    localstart = None
                elif start < sofar:
                    localstart = None
                elif sofar <= start < sofar + len(chunk):
                    localstart = start - sofar
                else:
                    continue

                if stop is None:
                    localstop = None
                elif stop <= sofar:
                    break
                elif sofar < stop < sofar + len(chunk):
                    localstop = stop - sofar
                else:
                    localstop = None

            else:
                if start is None:
                    localstart = None
                elif start < sofar:
                    break
                elif sofar <= start < sofar + len(chunk):
                    localstart = start - sofar
                else:
                    localstart = None

                if stop is None:
                    localstop = None
                elif stop < sofar:
                    localstop = None
                elif sofar <= stop < sofar + len(chunk):
                    localstop = stop - sofar
                else:
                    continue

            slicedchunk = chunk[(slice(localstart, localstop, localstep),) + tail]
            if len(slicedchunk) != 0:
                slicedchunks.append(slicedchunk)

        if step > 0:
            return slicedchunks
        else:
            return list(reversed(slicedchunks))

    def _zerolen(self):
        try:
            dtype = self.dtype
        except ValueError:
            return numpy.empty(0)
        else:
            return numpy.empty(0, dtype)

    def __getitem__(self, where):
        if self._isstring(where):
            chunks = []
            offsets = [0]
            for sofar, chunk in self._chunkiterator(0):
                chunks.append(chunk[where])
                offsets.append(offsets[-1] + len(chunks[-1]))
            return PartitionedArray(offsets, chunks, writeable=self._writeable)

        if not isinstance(where, tuple):
            where = (where,)
        head, tail = where[0], where[1:]

        if isinstance(head, (numbers.Integral, numpy.integer)):
            if head < 0:
                raise IndexError("negative indexes are not allowed in ChunkedArray")

            sofar = None
            for sofar, chunk in self._chunkiterator(head):
                if sofar <= head < sofar + len(chunk):
                    return chunk[(head - sofar,) + tail]

            raise IndexError("index {0} out of bounds for length {1}".format(head, 0 if sofar is None else sofar + len(chunk)))

        elif isinstance(head, slice):
            start, stop, step = head.start, head.stop, head.step
            if (start is not None and start < 0) or (stop is not None and stop < 0):
                raise IndexError("negative indexes are not allowed in ChunkedArray")

            slicedchunks = self._slicedchunks(start, stop, step, tail)

            if len(slicedchunks) == 0:
                return self._zerolen()

            if len(slicedchunks) == 1:
                out = slicedchunks[0]
            else:
                out = numpy.concatenate(slicedchunks)

            if step is None or step == 1:
                return out
            else:
                return out[::abs(step)]

        else:
            head = numpy.array(head, copy=False)
            if len(head.shape) == 1 and issubclass(head.dtype.type, numpy.integer):
                if len(head) == 0:
                    return self._zerolen()

                if (head < 0).any():
                    raise IndexError("negative indexes are not allowed in ChunkedArray")
                minindex, maxindex = head.min(), head.max()

                sofar = None
                out = None
                for sofar, chunk in self._chunkiterator(minindex):
                    if len(chunk) == 0:
                        continue
                    if out is None:
                        out = numpy.empty(len(head), dtype=numpy.dtype((chunk.dtype, chunk.shape[1:])))

                    indexes = head - sofar
                    mask = (indexes >= 0)
                    numpy.bitwise_and(mask, (indexes < len(chunk)), mask)
                    masked = indexes[mask]
                    if len(masked) != 0:
                        out[(mask,) + tail] = chunk[(masked,) + tail]

                    if sofar + len(chunk) > maxindex:
                        break

                if maxindex >= sofar + len(chunk):
                    raise IndexError("index {0} out of bounds for length {1}".format(maxindex, 0 if sofar is None else sofar + len(chunk)))

                return out[(slice(None),) + tail]

            elif len(head.shape) == 1 and issubclass(head.dtype.type, (numpy.bool, numpy.bool_)):
                out = None
                this = next = 0
                for sofar, chunk in self._chunkiterator(0):
                    if len(chunk) == 0:
                        continue
                    if out is None:
                        out = numpy.empty(numpy.count_nonzero(head), dtype=numpy.dtype((chunk.dtype, chunk.shape[1:])))

                    submask = head[sofar : sofar + len(chunk)]

                    next += numpy.count_nonzero(submask)
                    out[(slice(this, next),) + tail] = chunk[(submask,) + tail]
                    this = next

                if len(head) != sofar + len(chunk):
                    raise IndexError("boolean index did not match indexed array along dimension 0; dimension is {0} but corresponding boolean dimension is {1}".format(sofar + len(chunk), len(head)))

                return out[(slice(None),) + tail]

            else:
                raise TypeError("cannot interpret shape {0}, dtype {1} as a fancy index or mask".format(head.shape, head.dtype))

    def __setitem__(self, where, what):
        if self._isstring(where):
            if isinstance(what, (collections.Sequence, numpy.ndarray, awkward.array.base.AwkwardArray)) and len(what) != 1:
                what = numpy.array(what, copy=False)
                if self.shape != what.shape:
                    raise ValueError("shape mismatch: value array of shape {0} could not be broadcast to indexing result of shape {1}".format(what.shape, self.shape))
                for sofar, chunk in self._chunkiterator(0):
                    chunk[where] = what[sofar : sofar + len(chunk)]
            else:
                for sofar, chunk in self._chunkiterator(0):
                    chunk[where] = what
            return

        if not self._writeable:
            raise ValueError("assignment destination is read-only")

        if not isinstance(where, tuple):
            where = (where,)
        head, tail = where[0], where[1:]

        if isinstance(head, (numbers.Integral, numpy.integer)):
            if head < 0:
                raise IndexError("negative indexes are not allowed in ChunkedArray")

            sofar = None
            for sofar, chunk in self._chunkiterator(head):
                if sofar <= head < sofar + len(chunk):
                    chunk[(head - sofar,) + tail] = what
                    return

            raise IndexError("index {0} out of bounds for length {1}".format(head, 0 if sofar is None else sofar + len(chunk)))

        elif isinstance(head, slice):
            start, stop, step = head.start, head.stop, head.step
            if (start is not None and start < 0) or (stop is not None and stop < 0):
                raise IndexError("negative indexes are not allowed in ChunkedArray")

            carry = 0
            fullysliced = []
            for slicedchunk in self._slicedchunks(start, stop, step, tail):
                if step is not None and step != 1:
                    length = len(slicedchunk)
                    slicedchunk = slicedchunk[carry::abs(step)]
                    carry = (carry - length) % step
                fullysliced.append(slicedchunk)

            if isinstance(what, (collections.Sequence, numpy.ndarray, awkward.array.base.AwkwardArray)) and len(what) == 1:
                for slicedchunk in fullysliced:
                    slicedchunk[:] = what[0]
            elif isinstance(what, (collections.Sequence, numpy.ndarray, awkward.array.base.AwkwardArray)):
                if len(what) != sum(len(x) for x in fullysliced):
                    raise ValueError("cannot copy sequence with size {0} to array with dimension {1}".format(len(what), sum(len(x) for x in fullysliced)))
                this = next = 0
                for slicedchunk in fullysliced:
                    next += len(slicedchunk)
                    slicedchunk[:] = what[this:next]
                    this = next
            else:
                for slicedchunk in fullysliced:
                    slicedchunk[:] = what

        else:
            head = numpy.array(head, copy=False)
            if len(head.shape) == 1 and issubclass(head.dtype.type, numpy.integer):
                if isinstance(what, (collections.Sequence, numpy.ndarray, awkward.array.base.AwkwardArray)) and len(what) != 1:
                    if hasattr(what, "shape"):
                        whatshape = what.shape
                    else:
                        whatshape = (len(what),)
                    if (len(head),) + tail != whatshape:
                        raise ValueError("shape mismatch: value array of shape {0} could not be broadcast to indexing result of shape {1}".format(whatshape, (len(head),) + tail))

                if len(head) == 0:
                    return

                if (head < 0).any():
                    raise IndexError("negative indexes are not allowed in ChunkArray")
                minindex, maxindex = head.min(), head.max()

                sofar = None
                chunks = []
                offsets = []
                for sofar, chunk in self._chunkiterator(minindex):
                    chunks.append(chunk)
                    if len(offsets) == 0:
                        offsets.append(sofar)
                    offsets.append(offsets[-1] + len(chunk))

                    if sofar + len(chunk) > maxindex:
                        break

                if sofar is None or maxindex >= sofar + len(chunk):
                    raise IndexError("index {0} out of bounds for length {1}".format(maxindex, 0 if sofar is None else sofar + len(chunk)))

                if isinstance(what, (collections.Sequence, numpy.ndarray, awkward.array.base.AwkwardArray)) and len(what) == 1:
                    for chunk, offset in awkward.util.izip(chunks, offsets):
                        indexes = head - offset
                        mask = (indexes >= 0)
                        numpy.bitwise_and(mask, (indexes < len(chunk)), mask)
                        chunk[indexes[mask]] = what[0]

                elif isinstance(what, (collections.Sequence, numpy.ndarray, awkward.array.base.AwkwardArray)):
                    # must fill "self[where] = what" using the same order for where and what, with locations scattered among a list of chunks
                    # thus, Pythonic iteration is necessary
                    chunkindex = numpy.searchsorted(offsets, head, side="right") - 1
                    assert (chunkindex >= 0).all()
                    i = 0
                    for headi, chunki in awkward.util.izip(head, chunkindex):
                        chunks[chunki][headi - offsets[chunki]] = what[i]
                        i += 1

                else:
                    for chunk, offset in awkward.util.izip(chunks, offsets):
                        indexes = head - offset
                        mask = (indexes >= 0)
                        numpy.bitwise_and(mask, (indexes < len(chunk)), mask)
                        chunk[indexes[mask]] = what
                    
            elif len(head.shape) == 1 and issubclass(head.dtype.type, (numpy.bool, numpy.bool_)):
                submasks = []
                for sofar, chunk in self._chunkiterator(0):
                    submask = head[sofar : sofar + len(chunk)]
                    submasks.append((submask, chunk))

                if len(head) != sofar + len(chunk):
                    raise IndexError("boolean index did not match indexed array along dimension 0; dimension is {0} but corresponding boolean dimension is {1}".format(sofar + len(chunk), len(head)))

                if isinstance(what, (collections.Sequence, numpy.ndarray, awkward.array.base.AwkwardArray)) and len(what) == 1:
                    for submask, chunk in submasks:
                        chunk[submask] = what[0]

                elif isinstance(what, (collections.Sequence, numpy.ndarray, awkward.array.base.AwkwardArray)):
                    this = next = 0
                    for submask, chunk in submasks:
                        next += numpy.count_nonzero(submask)
                        chunk[submask] = what[this:next]
                        this = next

                else:
                    for submask, chunk in submasks:
                        chunk[submask] = what

            else:
                raise TypeError("cannot interpret shape {0}, dtype {1} as a fancy index or mask".format(head.shape, head.dtype))

class PartitionedArray(ChunkedArray):
    def __init__(self, offsets, chunks, writeable=True):
        super(PartitionedArray, self).__init__(chunks, writeable=writeable)
        self.offsets = offsets
        if len(self._offsets) != len(self._chunks) + 1:
            raise ValueError("length of offsets {0} must be equal to length of chunks {1} plus one ({2})".format(len(self._offsets), len(self._chunks), len(self._chunks) + 1))

    @property
    def offsets(self):
        return self._offsets

    @offsets.setter
    def offsets(self, value):
        value = self._toarray(value, self.INDEXTYPE, (numpy.ndarray, awkward.array.base.AwkwardArray))

        if len(value) == 0:
            raise ValueError("offsets must be non-empty")
        if value[0] != 0:
            raise ValueError("offsets must begin with zero")
        if (value[1:] - value[:-1] < 0).any():
            raise ValueError("offsets must be monotonically increasing")

        self._offsets = value

    def _chunkiterator(self, minindex):
        dtype = None

        if len(self._offsets) != len(self._chunks) + 1:
            raise ValueError("length of offsets {0} must be equal to length of chunks {1} plus one ({2})".format(len(self._offsets), len(self._chunks), len(self._chunks) + 1))

        i = numpy.searchsorted(self._offsets, minindex, side="right") - 1
        assert i >= 0
        sofar = self._offsets[i]
        while i < len(self._chunks):
            if not isinstance(self._chunks[i], (numpy.ndarray, awkward.array.base.AwkwardArray)):
                self._chunks[i] = self._toarray(self._chunks[i], self.CHARTYPE, (numpy.ndarray, awkward.array.base.AwkwardArray))

            if len(self._chunks[i]) != 0:
                thisdtype = numpy.dtype((self._chunks[i].dtype, self._chunks[i].shape[1:]))
                if dtype is None:
                    dtype = thisdtype
                elif dtype != thisdtype:
                    raise ValueError("chunk starting at index {0} has dtype {1}, different from {2}".format(sofar, thisdtype, dtype))

            if sofar + len(self._chunks[i]) != self._offsets[i + 1]:
                raise ValueError("partitioning is wrong: chunk starting at index {0} has length {1}, the sum of which is not equal to the next offset at {2}".format(sofar, len(self._chunks[i]), self._offsets[i + 1]))

            if sofar + len(self._chunks[i]) > minindex:
                yield sofar, self._chunks[i]

            sofar += len(self._chunks[i])
            i += 1

    @property
    def dtype(self):
        if len(self._offsets) != len(self._chunks) + 1:
            raise ValueError("length of offsets {0} must be equal to length of chunks {1} plus one ({2})".format(len(self._offsets), len(self._chunks), len(self._chunks) + 1))

        for i, count in enumerate(self._offsets[1:] - self._offsets[:-1]):
            if count > 0:
                if not isinstance(self._chunks[i], (numpy.ndarray, awkward.array.base.AwkwardArray)):
                    self._chunks[i] = self._toarray(self._chunks[i], self.CHARTYPE, (numpy.ndarray, awkward.array.base.AwkwardArray))
                return numpy.dtype((self._chunks[i].dtype, self._chunks[i].shape[1:]))

        raise ValueError("chunks are empty; cannot determine dtype")

    def __len__(self):
        return self._offsets[-1]

    def __iter__(self):
        if len(self._offsets) != len(self._chunks) + 1:
            raise ValueError("length of offsets {0} must be equal to length of chunks {1} plus one ({2})".format(len(self._offsets), len(self._chunks), len(self._chunks) + 1))

        i = 0
        while i < len(self._chunks):
            if not isinstance(self._chunks[i], (numpy.ndarray, awkward.array.base.AwkwardArray)):
                self._chunks[i] = self._toarray(self._chunks[i], self.CHARTYPE, (numpy.ndarray, awkward.array.base.AwkwardArray))
            for x in self._chunks[i]:
                yield x
            i += 1
        
    def __str__(self):
        return super(ChunkedArray, self).__str__()

    def _normalizeindex(self, where):
        if not isinstance(where, tuple):
            where = (where,)
        head, tail = where[0], where[1:]

        if isinstance(head, (numbers.Integral, numpy.integer)):
            if head + len(self) < 0:
                raise IndexError("index {0} out of bounds for length {1}".format(head, len(self)))
            if head < 0:
                head += len(self)
            assert head >= 0

        elif isinstance(head, slice):
            start, stop, step = head.indices(len(self))

            if step < 0 and start == -1:
                start = None
            if step < 0 and stop == -1:
                stop = None
            assert start is None or start >= 0
            assert stop is None or stop >= 0

            head = slice(start, stop, step)

        else:
            head = numpy.array(head, copy=False)
            if len(head.shape) == 1 and issubclass(head.dtype.type, numpy.integer):
                mask = (head < 0)
                if mask.any():
                    head[mask] += len(self)
                    if (head < 0).any():
                        raise IndexError("index {0} out of bounds for length {1}".format(head[head < 0][0] - len(self), len(self)))

        return (head,) + tail

    def __getitem__(self, where):
        if self._isstring(where):
            return super(PartitionedArray, self).__getitem__(where)
        else:
            return super(PartitionedArray, self).__getitem__(self._normalizeindex(where))

    def __setitem__(self, where, what):
        if self._isstring(where):
            super(PartitionedArray, self).__setitem__(where, what)
        else:
            super(PartitionedArray, self).__setitem__(self._normalizeindex(where), what)
