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

import awkward.array.base
import awkward.type
import awkward.util

class ChunkedArray(awkward.array.base.AwkwardArray):
    def __init__(self, chunks, counts=[]):
        self.chunks = chunks
        self.counts = counts
        
    def copy(self, chunks=None, counts=None):
        out = self.__class__.__new__(self.__class__)
        out._chunks = list(self._chunks)
        out._counts = list(self._counts)
        out._types = list(self._types)
        out._offsets = self._offsets
        if chunks is not None:
            out.chunks = chunks
        if counts is not None:
            out.counts = counts
        return out

    def deepcopy(self, chunks=None, counts=None):
        out = self.copy(chunks=chunks, counts=counts)
        out._chunks = [awkward.util.deepcopy(out._chunks) for x in out._chunks]
        return out

    def empty_like(self, **overrides):
        mine = {}
        if len(self._chunks) == 0:
            chunks = []
        elif isinstance(self._chunks[0], awkward.util.numpy.ndarray):
            chunks = [awkward.util.numpy.empty_like(self._chunks[0])]
        else:
            chunks = [self._chunks[0].empty_like(**overrides)]
        return self.copy(chunks=chunks, counts=[len(self._chunks[0])])

    def zeros_like(self, **overrides):
        mine = {}
        if len(self._chunks) == 0:
            chunks = []
        elif isinstance(self._chunks[0], awkward.util.numpy.ndarray):
            chunks = [awkward.util.numpy.zeros_like(self._chunks[0])]
        else:
            chunks = [self._chunks[0].zeros_like(**overrides)]
        return self.copy(chunks=chunks, counts=[len(self._chunks[0])])

    def ones_like(self, **overrides):
        mine = {}
        if len(self._chunks) == 0:
            chunks = []
        elif isinstance(self._chunks[0], awkward.util.numpy.ndarray):
            chunks = [awkward.util.numpy.ones_like(self._chunks[0])]
        else:
            chunks = [self._chunks[0].ones_like(**overrides)]
        return self.copy(chunks=chunks, counts=[len(self._chunks[0])])

    @property
    def chunks(self):
        return self._chunks

    @chunks.setter
    def chunks(self, value):
        try:
            self._chunks = list(value)
        except TypeError:
            raise TypeError("chunks must be iterable")
        self._types = [None] * len(self._chunks)

    @property
    def counts(self):
        return self._counts

    @counts.setter
    def counts(self, value):
        try:
            if not all(isinstance(x, (numbers.Integral and awkward.util.numpy.integer)) and x >= 0 for x in value):
                raise ValueError("counts must contain only non-negative integers")
        except TypeError:
            raise TypeError("counts must be iterable")
        self._counts = list(value)
        self._offsets = None

    @property
    def offsets(self):
        import awkward.array.jagged
        if self._offsets is None or len(self._offsets) != len(self._counts) + 1:
            self._offsets = awkward.array.jagged.counts2offsets(self._counts)
        return self._offsets

    @property
    def countsknown(self):
        return len(self._counts) == len(self._chunks)

    @property
    def typesknown(self):
        return all(x is not None for x in self._types)

    def knowcounts(self, until=None):
        if until is None:
            until = len(self._chunks)
        if not 0 <= until <= len(self._chunks):
            raise IndexError("cannot knowcounts until chunkid {0} with {1} chunks".format(until, len(self._chunks)))
        for i in range(len(self._counts), until):
            self._counts.append(len(self._chunks[i]))

    def knowtype(self, at):
        if not 0 <= at < len(self._chunks):
            raise IndexError("cannot knowtype at chunkid {0} with {1} chunks".format(at, len(self._chunks)))
        self._types[at] = awkward.type.fromarray(self._chunks[at]).to
        return self._types[at]

    def global2chunkid(self, index):
        self._valid()

        if isinstance(index, numbers.Integral, awkward.util.numpy.integer):
            if index < 0:
                index += len(self)
            if index < 0:
                raise IndexError("index {0} out of bounds for length {1}".format(index, len(self)))

            cumulative = self.offsets[-1]
            while index >= cumulative:
                if self.countsknown:
                    raise IndexError("index {0} out of bounds for length {1}".format(index, len(self)))
                count = len(self._chunks[len(self._counts)])
                cumulative += count
                self._counts.append(count)

            return awkward.util.numpy.searchsorted(self.offsets, index, "right") - 1

        else:
            index = awkward.util.numpy.array(index, copy=False)
            if len(index.shape) == 1 and issubclass(index.dtype.type, awkward.util.numpy.integer):
                if len(index) == 0:
                    return awkward.util.numpy.empty(0, dtype=awkward.util.INDEXTYPE)

                if (index < 0).any():
                    index += len(self)
                if (index < 0).any():
                    raise IndexError("index out of bounds for length {0}".format(len(self)))

                global2chunkid(index.max())    # make sure all the counts we need are known

                return awkward.util.numpy.searchsorted(self.offsets, index, "right") - 1

            else:
                raise TypeError("global2chunkid requires an integer or an array of integers")

    def global2local(self, index):
        chunkid = self.global2chunkid(index)
        if isinstance(index, numbers.Integral, awkward.util.numpy.integer):
            return index - self.offsets[chunkid], self._chunks[chunkid]
        else:
            return index - self.offsets[chunkid], awkward.util.numpy.array(self._chunks, dtype=awkward.util.numpy.object)[chunkid]

    def local2global(self, index, chunkid):
        if isinstance(chunkid, (numbers.Integral, awkward.util.numpy.integer)):
            self.knowcounts(chunkid + 1)
            self._valid()
            original_index = index
            if index < 0:
                index += self._counts[chunkid]
            if not 0 <= index < self._counts[chunkid]:
                raise IndexError("local index {0} is out of bounds in chunk {1}, which has length {2}".format(original_index, chunkid, self._counts[chunkid]))
            return self.offsets[chunkid] + index

        else:
            index = awkward.util.numpy.array(index, copy=False)
            chunkid = awkward.util.numpy.array(chunkid, copy=False)
            if len(index.shape) == 1 and issubclass(index.dtype.type, awkward.util.numpy.integer) and len(chunkid.shape) == 1 and issubclass(chunkid.dtype.type, awkward.util.numpy.integer):
                if len(index) != len(chunkid):
                    raise ValueError("len(index) is {0} and len(chunkid) is {1}, but they should be equal".format(len(index), len(chunkid)))

                self.knowcounts(chunkid.max() + 1)
                self._valid()
                counts = numpy.array(self._counts, dtype=awkward.util.INDEXTYPE)
                mask = (index < 0)
                index[mask] += counts[mask]
                if not ((0 <= index) & (index < counts)).all():
                    raise IndexError("some local indexes are out of bounds")
                return counts[chunkid] + index

            else:
                raise TypeError("local2global requires index and chunkid to be integers or arrays of integers")

    @property
    def type(self):
        for tpe in self._types:
            if tpe is not None:
                break
        else:
            if len(self._chunks) == 0:
                tpe = awkward.util.DEFAULTTYPE
            else:
                tpe = self.knowtype(0)

        self._valid()
        return awkward.type.ArrayType(len(self), tpe)

    def __len__(self):
        self.knowcounts()
        return self.offsets[-1]

    @property
    def shape(self):
        return self.type.shape

    @property
    def dtype(self):
        return self.type.dtype

    @property
    def slices(self):
        self.knowcounts()
        offsets = self.offsets
        return [slice(start, stop) for start, stop in zip(offsets[:-1], offsets[1:])]

    @property
    def base(self):
        raise TypeError("ChunkedArray has no base")

    def _valid(self):
        for tpe in self._types:
            if tpe is not None:
                break

        if tpe is not None:
            for i in range(len(self._types)):
                if self._types[i] is None or self._types[i] is tpe:
                    pass
                elif self._types[i] == tpe:
                    self._types[i] = tpe
                else:
                    raise TypeError("chunks do not have matching types:\n\n{0}\n\nversus\n\n{1}".format(tpe.__str__(indent="    "), self._types[i].__str__(indent="    ")))

        return len(self._counts) <= len(self._chunks)

    def _argfields(self, function):
        if isinstance(function, types.FunctionType) and function.__code__.co_argcount == 1:
            return awkward.util._argfields(function)
        if len(self._chunks) == 0 or isinstance(self.type.to, awkward.util.numpy.dtype):
            return awkward.util._argfields(function)
        else:
            return self._chunks[0]._argfields(function)

    def __iter__(self):
        raise NotImplementedError

    def __getitem__(self, where):
        self._valid()

        if awkward.util.isstringslice(where):
            if isinstance(where, awkward.util.string):
                if not self.type.hascolumn(where):
                    raise ValueError("no column named {0}".format(repr(where)))
            else:
                for x in where:
                    if not self.type.hascolumn(x):
                        raise ValueError("no column named {0}".format(repr(x)))
            chunks = []
            counts = []
            for chunk in self._chunks:
                chunks.append(chunk[where])
                counts.append(len(chunks[-1]))
            return ChunkedArray(chunks, counts=counts)

        if isinstance(where, tuple) and len(where) == 0:
            return self
        if not isinstance(where, tuple):
            where = (where,)
        head, tail = where[0], where[1:]

        if isinstance(head, (numbers.Integral, awkward.util.numpy.integer)):
            i = self.global2chunkid(head)





        elif isinstance(head, slice):
            raise NotImplementedError

        else:
            head = numpy.array(head, copy=False)
            if len(head.shape) == 1 and issubclass(head.dtype.type, awkward.util.numpy.integer):
                raise NotImplementedError

            elif len(head.shape) == 1 and issubclass(head.dtype.type, (awkward.util.numpy.bool, awkward.util.numpy.bool_)):
                raise NotImplementedError

            else:
                raise TypeError("cannot interpret shape {0}, dtype {1} as a fancy index or mask".format(head.shape, head.dtype))

    def _aligned(self, what):
        self.knowcounts()
        what.knowcounts()
        return self._counts == what._counts

    def __setitem__(self, where, what):
        if isinstance(what, ChunkedArray) and self._aligned(what):
            for mine, theirs in zip(self._chunks, what._chunks):
                mine[where] = theirs
        else:
            raise ValueError("only ChunkedArrays with the same chunk sizes can be assigned to columns of a ChunkedArray")
                    
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        self._valid()

        if method != "__call__":
            return NotImplemented

        first = None
        rest = []
        for x in inputs:
            if isinstance(x, ChunkedArray):
                if first is None:
                    first = x
                else:
                    rest.append(x)

        assert first is not None
        if not all(first._aligned(x) for x in rest):
            raise ValueError("ChunkedArrays can only be combined if they have the same chunk sizes")

        batches = []
        for i, slc in enumerate(first.slices):
            batch = []
            for x in inputs:
                if isinstance(x, ChunkedArray):
                    batch.append(x._chunks[i])
                elif isinstance(x, awkward.util.numpy.ndarray, awkward.array.base.AwkwardArray):
                    batch.append(x[slc])
                else:
                    batch.append(x)
            batches.append(batch)
        
        out = None
        chunks = {}
        for batch in batches:
            result = getattr(ufunc, method)(*batch, **kwargs)

            if isinstance(result, tuple):
                if out is None:
                    out = list(result)
                for i, x in enumerate(result):
                    if isinstance(x, (awkward.util.numpy.ndarray, awkward.array.base.AwkwardBase)):
                        if i not in chunks:
                            chunks[i] = []
                        chunks[i].append(x)

            elif method == "at":
                pass

            else:
                if isinstance(result, (awkward.util.numpy.ndarray, awkward.array.base.AwkwardBase)):
                    if None not in chunks:
                        chunks[None] = []
                    chunks[None].append(result)

            if out is None:
                if None in chunks:
                    return ChunkedArray(chunks[None])
                else:
                    return None
            else:
                for i in range(len(out)):
                    if i in chunks:
                        out[i] = ChunkedArray(chunks[i])
                return tuple(out)

    @classmethod
    def concat(cls, first, *rest):
        raise NotImplementedError

    @property
    def columns(self):
        




        raise NotImplementedError

    @property
    def allcolumns(self):
        raise NotImplementedError

    def pandas(self):
        raise NotImplementedError

# class ChunkedArray(awkward.array.base.AwkwardArray):
#     def __init__(self, chunks):
#         self.chunks = chunks

#     @property
#     def chunks(self):
#         return self._chunks

#     @chunks.setter
#     def chunks(self, value):
#         try:
#             self._chunks = list(value)
#         except TypeError:
#             raise TypeError("chunks must be iterable")

#     def _chunkiterator(self, minindex):
#         dtype = None

#         sofar = i = 0
#         while i < len(self._chunks):
#             if not isinstance(self._chunks[i], (numpy.ndarray, awkward.array.base.AwkwardArray)):
#                 self._chunks[i] = self._toarray(self._chunks[i], self.CHARTYPE, (numpy.ndarray, awkward.array.base.AwkwardArray))

#             if len(self._chunks[i]) != 0:
#                 thisdtype = numpy.dtype((self._chunks[i].dtype, self._chunks[i].shape[1:]))
#                 if dtype is None:
#                     dtype = thisdtype
#                 elif dtype != thisdtype:
#                     raise ValueError("chunk starting at index {0} has dtype {1}, different from {2}".format(sofar, thisdtype, dtype))

#             if sofar + len(self._chunks[i]) > minindex:
#                 yield sofar, self._chunks[i]

#             sofar += len(self._chunks[i])
#             i += 1

#     @property
#     def dtype(self):
#         for sofar, chunk in self._chunkiterator(0):
#             if len(chunk) != 0:
#                 return numpy.dtype((chunk.dtype, chunk.shape[1:]))
#         raise ValueError("chunks are empty; cannot determine dtype")

#     @property
#     def dimension(self):
#         try:
#             return self.dtype.shape
#         except ValueError:
#             raise ValueError("chunks are empty; cannot determine dimension")

#     def __len__(self):
#         return sum(len(chunk) for chunk in self._chunks)

#     @property
#     def shape(self):
#         return (len(self),) + self.dimension

#     def topartitioned(self):
#         offsets = [0]
#         for chunk in self._chunks:
#             offsets.append(offsets[-1] + len(chunk))
#         return PartitionedArray(offsets, self._chunks)

#     def __iter__(self):
#         i = 0
#         while i < len(self._chunks):
#             if not isinstance(self._chunks[i], (numpy.ndarray, awkward.array.base.AwkwardArray)):
#                 self._chunks[i] = self._toarray(self._chunks[i], self.CHARTYPE, (numpy.ndarray, awkward.array.base.AwkwardArray))
#             for x in self._chunks[i]:
#                 yield x
#             i += 1

#     def __str__(self):
#         values = []
#         for x in self:
#             if len(values) == 7:
#                 return "[{0} ...]".format(" ".join(str(x) if isinstance(x, (numpy.ndarray, awkward.array.base.AwkwardArray)) else repr(x) for x in values))
#             values.append(x)
#         return "[{0}]".format(" ".join(str(x) if isinstance(x, (numpy.ndarray, awkward.array.base.AwkwardArray)) else repr(x) for x in values))

#     def _slicedchunks(self, start, stop, step, tail):
#         if step == 0:
#             raise ValueError("slice step cannot be zero")
#         elif step is None:
#             step = 1

#         slicedchunks = []
#         localstep = 1 if step > 0 else -1

#         if step is None or step > 0:
#             if start is None:
#                 minindex = 0
#             else:
#                 minindex = start
#         else:
#             if stop is None:
#                 minindex = 0
#             else:
#                 minindex = stop + 1

#         for sofar, chunk in self._chunkiterator(minindex):
#             if len(chunk) == 0:
#                 continue

#             if step > 0:
#                 if start is None:
#                     localstart = None
#                 elif start < sofar:
#                     localstart = None
#                 elif sofar <= start < sofar + len(chunk):
#                     localstart = start - sofar
#                 else:
#                     continue

#                 if stop is None:
#                     localstop = None
#                 elif stop <= sofar:
#                     break
#                 elif sofar < stop < sofar + len(chunk):
#                     localstop = stop - sofar
#                 else:
#                     localstop = None

#             else:
#                 if start is None:
#                     localstart = None
#                 elif start < sofar:
#                     break
#                 elif sofar <= start < sofar + len(chunk):
#                     localstart = start - sofar
#                 else:
#                     localstart = None

#                 if stop is None:
#                     localstop = None
#                 elif stop < sofar:
#                     localstop = None
#                 elif sofar <= stop < sofar + len(chunk):
#                     localstop = stop - sofar
#                 else:
#                     continue

#             slicedchunk = chunk[self._singleton((slice(localstart, localstop, localstep),) + tail)]
#             if len(slicedchunk) != 0:
#                 slicedchunks.append(slicedchunk)

#         if step > 0:
#             return slicedchunks
#         else:
#             return list(reversed(slicedchunks))

#     def _zerolen(self):
#         try:
#             dtype = self.dtype
#         except ValueError:
#             return numpy.empty(0)
#         else:
#             return numpy.empty(0, dtype)

#     def __getitem__(self, where):
#         if self._isstring(where):
#             chunks = []
#             offsets = [0]
#             for sofar, chunk in self._chunkiterator(0):
#                 chunks.append(chunk[where])
#                 offsets.append(offsets[-1] + len(chunks[-1]))
#             return PartitionedArray(offsets, chunks)

#         if not isinstance(where, tuple):
#             where = (where,)
#         head, tail = where[0], where[1:]

#         if isinstance(head, (numbers.Integral, numpy.integer)):
#             if head < 0:
#                 raise IndexError("negative indexes are not allowed in ChunkedArray")

#             sofar = None
#             for sofar, chunk in self._chunkiterator(head):
#                 if sofar <= head < sofar + len(chunk):
#                     return chunk[self._singleton((head - sofar,) + tail)]

#             raise IndexError("index {0} out of bounds for length {1}".format(head, 0 if sofar is None else sofar + len(chunk)))

#         elif isinstance(head, slice):
#             start, stop, step = head.start, head.stop, head.step
#             if (start is not None and start < 0) or (stop is not None and stop < 0):
#                 raise IndexError("negative indexes are not allowed in ChunkedArray")

#             slicedchunks = self._slicedchunks(start, stop, step, tail)

#             if len(slicedchunks) == 0:
#                 return self._zerolen()

#             if len(slicedchunks) == 1:
#                 out = slicedchunks[0]
#             else:
#                 out = numpy.concatenate(slicedchunks)

#             if step is None or step == 1:
#                 return out
#             else:
#                 return out[::abs(step)]

#         else:
#             head = numpy.array(head, copy=False)
#             if len(head.shape) == 1 and issubclass(head.dtype.type, numpy.integer):
#                 if len(head) == 0:
#                     return self._zerolen()

#                 if (head < 0).any():
#                     raise IndexError("negative indexes are not allowed in ChunkedArray")
#                 minindex, maxindex = head.min(), head.max()

#                 sofar = None
#                 out = None
#                 for sofar, chunk in self._chunkiterator(minindex):
#                     if len(chunk) == 0:
#                         continue
#                     if out is None:
#                         out = numpy.empty(len(head), dtype=numpy.dtype((chunk.dtype, chunk.shape[1:])))

#                     indexes = head - sofar
#                     mask = (indexes >= 0)
#                     numpy.bitwise_and(mask, (indexes < len(chunk)), mask)
#                     masked = indexes[mask]
#                     if len(masked) != 0:
#                         out[self._singleton((mask,) + tail)] = chunk[self._singleton((masked,) + tail)]

#                     if sofar + len(chunk) > maxindex:
#                         break

#                 if maxindex >= sofar + len(chunk):
#                     raise IndexError("index {0} out of bounds for length {1}".format(maxindex, 0 if sofar is None else sofar + len(chunk)))

#                 return out[self._singleton((slice(None),) + tail)]

#             elif len(head.shape) == 1 and issubclass(head.dtype.type, (numpy.bool, numpy.bool_)):
#                 out = None
#                 this = next = 0
#                 for sofar, chunk in self._chunkiterator(0):
#                     if len(chunk) == 0:
#                         continue
#                     if out is None:
#                         out = numpy.empty(numpy.count_nonzero(head), dtype=numpy.dtype((chunk.dtype, chunk.shape[1:])))

#                     submask = head[sofar : sofar + len(chunk)]

#                     next += numpy.count_nonzero(submask)
#                     out[self._singleton((slice(this, next),) + tail)] = chunk[self._singleton((submask,) + tail)]
#                     this = next

#                 if len(head) != sofar + len(chunk):
#                     raise IndexError("boolean index did not match indexed array along dimension 0; dimension is {0} but corresponding boolean dimension is {1}".format(sofar + len(chunk), len(head)))

#                 return out[self._singleton((slice(None),) + tail)]

#             else:
#                 raise TypeError("cannot interpret shape {0}, dtype {1} as a fancy index or mask".format(head.shape, head.dtype))

# class PartitionedArray(ChunkedArray):
#     def __init__(self, offsets, chunks):
#         super(PartitionedArray, self).__init__(chunks)
#         self.offsets = offsets
#         if len(self._offsets) != len(self._chunks) + 1:
#             raise ValueError("length of offsets {0} must be equal to length of chunks {1} plus one ({2})".format(len(self._offsets), len(self._chunks), len(self._chunks) + 1))

#     @property
#     def offsets(self):
#         return self._offsets

#     @offsets.setter
#     def offsets(self, value):
#         value = self._toarray(value, self.INDEXTYPE, (numpy.ndarray, awkward.array.base.AwkwardArray))

#         if len(value) == 0:
#             raise ValueError("offsets must be non-empty")
#         if value[0] != 0:
#             raise ValueError("offsets must begin with zero")
#         if (value[1:] - value[:-1] < 0).any():
#             raise ValueError("offsets must be monotonically increasing")

#         self._offsets = value

#     def _chunkiterator(self, minindex):
#         dtype = None

#         if len(self._offsets) != len(self._chunks) + 1:
#             raise ValueError("length of offsets {0} must be equal to length of chunks {1} plus one ({2})".format(len(self._offsets), len(self._chunks), len(self._chunks) + 1))

#         i = numpy.searchsorted(self._offsets, minindex, side="right") - 1
#         assert i >= 0
#         sofar = self._offsets[i]
#         while i < len(self._chunks):
#             if not isinstance(self._chunks[i], (numpy.ndarray, awkward.array.base.AwkwardArray)):
#                 self._chunks[i] = self._toarray(self._chunks[i], self.CHARTYPE, (numpy.ndarray, awkward.array.base.AwkwardArray))

#             if len(self._chunks[i]) != 0:
#                 thisdtype = numpy.dtype((self._chunks[i].dtype, self._chunks[i].shape[1:]))
#                 if dtype is None:
#                     dtype = thisdtype
#                 elif dtype != thisdtype:
#                     raise ValueError("chunk starting at index {0} has dtype {1}, different from {2}".format(sofar, thisdtype, dtype))

#             if sofar + len(self._chunks[i]) < self._offsets[i + 1]:
#                 raise ValueError("partitioning is wrong: chunk starting at index {0} has length {1}, which is too short to reach the next offset at {2}".format(sofar, len(self._chunks[i]), self._offsets[i + 1]))

#             if sofar + len(self._chunks[i]) > minindex:
#                 yield sofar, self._chunks[i]

#             sofar += len(self._chunks[i])
#             i += 1

#     @property
#     def dtype(self):
#         if len(self._offsets) != len(self._chunks) + 1:
#             raise ValueError("length of offsets {0} must be equal to length of chunks {1} plus one ({2})".format(len(self._offsets), len(self._chunks), len(self._chunks) + 1))

#         for i in range(len(self._offsets) - 1):
#             if self._offsets[i + 1] - self._offsets[i] > 0:
#                 if not isinstance(self._chunks[i], (numpy.ndarray, awkward.array.base.AwkwardArray)):
#                     self._chunks[i] = self._toarray(self._chunks[i], self.CHARTYPE, (numpy.ndarray, awkward.array.base.AwkwardArray))
#                 return numpy.dtype((self._chunks[i].dtype, self._chunks[i].shape[1:]))

#         raise ValueError("chunks are empty; cannot determine dtype")

#     def __len__(self):
#         return self._offsets[-1]

#     def __iter__(self):
#         if len(self._offsets) != len(self._chunks) + 1:
#             raise ValueError("length of offsets {0} must be equal to length of chunks {1} plus one ({2})".format(len(self._offsets), len(self._chunks), len(self._chunks) + 1))

#         i = 0
#         while i < len(self._chunks):
#             if not isinstance(self._chunks[i], (numpy.ndarray, awkward.array.base.AwkwardArray)):
#                 self._chunks[i] = self._toarray(self._chunks[i], self.CHARTYPE, (numpy.ndarray, awkward.array.base.AwkwardArray))
#             for x in self._chunks[i][: self._offsets[i + 1] - self._offsets[i]]:
#                 yield x
#             i += 1
        
#     def __str__(self):
#         return super(ChunkedArray, self).__str__()

#     def _normalizeindex(self, where):
#         if not isinstance(where, tuple):
#             where = (where,)
#         head, tail = where[0], where[1:]

#         if isinstance(head, (numbers.Integral, numpy.integer)):
#             if head + len(self) < 0:
#                 raise IndexError("index {0} out of bounds for length {1}".format(head, len(self)))
#             if head < 0:
#                 head += len(self)
#             assert head >= 0

#         elif isinstance(head, slice):
#             start, stop, step = head.indices(len(self))

#             if step < 0 and start == -1:
#                 start = None
#             if step < 0 and stop == -1:
#                 stop = None
#             assert start is None or start >= 0
#             assert stop is None or stop >= 0

#             head = slice(start, stop, step)

#         else:
#             head = numpy.array(head, copy=False)
#             if len(head.shape) == 1 and issubclass(head.dtype.type, numpy.integer):
#                 mask = (head < 0)
#                 if mask.any():
#                     head[mask] += len(self)
#                     if (head < 0).any():
#                         raise IndexError("index {0} out of bounds for length {1}".format(head[head < 0][0] - len(self), len(self)))

#         return (head,) + tail

#     def __getitem__(self, where):
#         if self._isstring(where):
#             return super(PartitionedArray, self).__getitem__(where)
#         else:
#             return super(PartitionedArray, self).__getitem__(self._normalizeindex(where))

class AppendableArray(ChunkedArray):
    pass

# class AppendableArray(PartitionedArray):
#     @classmethod
#     def empty(cls, generator):
#         return AppendableArray([0], [], generator)

#     def __init__(self, offsets, chunks, generator):
#         super(AppendableArray, self).__init__(offsets, chunks)
#         self.generator = generator

#     @property
#     def offsets(self):
#         return self._offsets

#     @offsets.setter
#     def offsets(self, value):
#         self._offsets = list(value)

#     @property
#     def generator(self):
#         return self._generator

#     @generator.setter
#     def generator(self, value):
#         if not callable(value):
#             raise TypeError("generator must be a callable (of zero arguments)")
#         self._generator = value

#     def append(self, value):
#         if len(self._offsets) != len(self._chunks) + 1:
#             raise ValueError("length of offsets {0} must be equal to length of chunks {1} plus one ({2})".format(len(self._offsets), len(self._chunks), len(self._chunks) + 1))

#         if len(self._chunks) == 0 or self._offsets[-1] - self._offsets[-2] == len(self._chunks[-1]):
#             self._chunks.append(self._generator())
#             self._offsets.append(self._offsets[-1])

#         laststart = self._offsets[-1] - self._offsets[-2]
#         self._chunks[-1][laststart] = value
#         self._offsets[-1] += 1

#     def extend(self, values):
#         if len(self._offsets) != len(self._chunks) + 1:
#             raise ValueError("length of offsets {0} must be equal to length of chunks {1} plus one ({2})".format(len(self._offsets), len(self._chunks), len(self._chunks) + 1))

#         while len(values) > 0:
#             if len(self._chunks) == 0 or self._offsets[-1] - self._offsets[-2] >= len(self._chunks[-1]):
#                 self._chunks.append(self._generator())
#                 self._offsets.append(self._offsets[-1])

#             laststart = self._offsets[-1] - self._offsets[-2]
#             available = len(self._chunks[-1]) - laststart
#             if len(values) < available:
#                 self._chunks[-1][laststart : laststart + len(values)] = values
#                 self._offsets[-1] += len(values)
#                 values = []
#             else:
#                 self._chunks[-1][laststart:] = values[:available]
#                 self._offsets[-1] += available
#                 values = values[available:]
