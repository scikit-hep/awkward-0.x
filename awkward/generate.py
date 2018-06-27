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

import awkward.array.base
import awkward.util
from awkward.array.chunked import ChunkedArray, PartitionedArray, AppendableArray
from awkward.array.indexed import IndexedArray, ByteIndexedArray, UnionArray
from awkward.array.jagged import JaggedArray, ByteJaggedArray
from awkward.array.masked import MaskedArray, BitMaskedArray
from awkward.array.sparse import SparseArray
from awkward.array.table import Table
from awkward.array.virtual import VirtualArray, VirtualObjectArray, PersistentArray

def fromiterable(iterable, chunksize=1024, references=False, writeable=True):
    if references:
        raise NotImplementedError    # keep all ids in a hashtable to create pointers

    def add(chunks, offsets, newchunk, ismine, fillobj):
        if len(chunks) == 0 or offsets[-1] - offsets[-2] == len(chunks[-1]):
            chunks.append(newchunk())
            offsets.append(offsets[-1])

        if ismine(chunks[-1]):
            fillobj(chunks[-1], offsets[-1] - offsets[-2])
            offsets[-1] += 1

        else:
            if not isinstance(chunks[-1], UnionArray):
                chunks[-1] = UnionArray(numpy.empty(chunksize, dtype=awkward.array.base.AwkwardArray.INDEXTYPE),
                                        numpy.empty(chunksize, dtype=awkward.array.base.AwkwardArray.INDEXTYPE),
                                        [chunks[-1]],
                                        writeable=writeable)
                chunks[-1]._nextindex = [offsets[-1] - offsets[-2]]
                chunks[-1]._tags[: offsets[-1] - offsets[-2]] = 0
                chunks[-1]._index[: offsets[-1] - offsets[-2]] = numpy.arange(offsets[-1] - offsets[-2], dtype=awkward.array.base.AwkwardArray.INDEXTYPE)

            if not any(ismine(content) for content in chunks[-1]._contents):
                chunks[-1]._nextindex.append(0)
                chunks[-1]._contents = chunks[-1]._contents + (newchunk(),)

            for tag, content in enumerate(chunks[-1]._contents):
                if ismine(content):
                    nextindex = chunks[-1]._nextindex[tag]
                    chunks[-1]._nextindex[tag] += 1
                    fillobj(content, nextindex)
                    chunks[-1]._tags[offsets[-1] - offsets[-2]] = tag
                    chunks[-1]._index[offsets[-1] - offsets[-2]] = nextindex
                    offsets[-1] += 1
                    break

    def recurse(obj, chunks, offsets):
        if obj is None:
            HERE

        elif isinstance(obj, (bool, numpy.bool, numpy.bool_)):
            def newchunk():
                return numpy.empty(chunksize, dtype=numpy.bool_)

            def ismine(x):
                return isinstance(x, numpy.ndarray) and x.dtype == numpy.dtype(numpy.bool_)

            def fillobj(array, where):
                array[where] = obj

            add(chunks, offsets, newchunk, ismine, fillobj)

        elif isinstance(obj, (numbers.Integral, numpy.integer)):
            def newchunk():
                return numpy.empty(chunksize, dtype=numpy.int64)

            def ismine(x):
                return isinstance(x, numpy.ndarray) and x.dtype == numpy.dtype(numpy.int64)

            def fillobj(array, where):
                array[where] = obj

            add(chunks, offsets, newchunk, ismine, fillobj)

        elif isinstance(obj, (numbers.Real, numpy.floating)):
            HERE

        elif isinstance(obj, (numbers.Complex, numpy.complex, numpy.complexfloating)):
            raise NotImplementedError

        elif isinstance(obj, bytes):
            raise NotImplementedError

        elif isinstance(obj, awkward.util.string):
            raise NotImplementedError

        elif isinstance(obj, dict):
            HERE

        elif isinstance(obj, tuple):
            raise NotImplementedError

        else:
            try:
                it = iter(obj)

            except TypeError:
                HERE

            else:
                def newchunk():
                    out = JaggedArray.fromoffsets(numpy.zeros(chunksize + 1, dtype=awkward.array.base.AwkwardArray.INDEXTYPE),
                                                  PartitionedArray([0], [], writeable=writeable),
                                                  writeable=writeable)
                    out._starts[0] = 0
                    out._content._offsets = [0]  # as an appendable list, not a Numpy array
                    return out

                def ismine(x):
                    return isinstance(x, JaggedArray)

                def fillobj(array, where):
                    array._stops[where] = array._starts[where]
                    for x in it:
                        recurse(x, array._content._chunks, array._content._offsets)
                        array._stops[where] += 1

                add(chunks, offsets, newchunk, ismine, fillobj)

    chunks = []
    offsets = [0]
    for x in iterable:
        recurse(x, chunks, offsets)

    return PartitionedArray(offsets, chunks, writeable=writeable)
