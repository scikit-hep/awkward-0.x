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

import codecs
import numbers

import numpy

import awkward.array.base
import awkward.util
from awkward.array.chunked import PartitionedArray, AppendableArray
from awkward.array.indexed import IndexedMaskedArray, UnionArray
from awkward.array.jagged import JaggedArray
from awkward.array.masked import BitMaskedArray
from awkward.array.table import Table
from awkward.array.virtual import VirtualObjectArray

def fromiter(iterable, chunksize=1024, references=False):
    if references:
        raise NotImplementedError    # keep all ids in a hashtable to create pointers

    tobytes = lambda x: x.tobytes()
    tostring = lambda x: codecs.utf_8_decode(x.tobytes())[0]

    def insert(obj, chunks, offsets, newchunk, ismine, promote, fillobj):
        if len(chunks) == 0 or offsets[-1] - offsets[-2] == len(chunks[-1]):
            chunks.append(newchunk())
            offsets.append(offsets[-1])

        if ismine(chunks[-1]):
            chunks[-1] = promote(chunks[-1])
            fillobj(obj, chunks[-1], offsets[-1] - offsets[-2])
            offsets[-1] += 1

        elif isinstance(chunks[-1], IndexedMaskedArray) and len(chunks[-1]._content) == 0:
            chunks[-1]._content = newchunk()

            nextindex = chunks[-1]._nextindex
            chunks[-1]._nextindex += 1
            chunks[-1]._index[offsets[-1] - offsets[-2]] = nextindex

            chunks[-1]._content = promote(chunks[-1]._content)
            fillobj(obj, chunks[-1]._content, nextindex)
            offsets[-1] += 1

        elif isinstance(chunks[-1], IndexedMaskedArray) and ismine(chunks[-1]._content):
            nextindex = chunks[-1]._nextindex
            chunks[-1]._nextindex += 1
            chunks[-1]._index[offsets[-1] - offsets[-2]] = nextindex

            chunks[-1]._content = promote(chunks[-1]._content)
            fillobj(obj, chunks[-1]._content, nextindex)
            offsets[-1] += 1

        else:
            if not isinstance(chunks[-1], UnionArray):
                chunks[-1] = UnionArray(numpy.empty(chunksize, dtype=awkward.array.base.AwkwardArray.INDEXTYPE),
                                        numpy.empty(chunksize, dtype=awkward.array.base.AwkwardArray.INDEXTYPE),
                                        [chunks[-1]])
                chunks[-1]._nextindex = [offsets[-1] - offsets[-2]]
                chunks[-1]._tags[: offsets[-1] - offsets[-2]] = 0
                chunks[-1]._index[: offsets[-1] - offsets[-2]] = numpy.arange(offsets[-1] - offsets[-2], dtype=awkward.array.base.AwkwardArray.INDEXTYPE)
                chunks[-1]._contents = list(chunks[-1]._contents)

            if not any(ismine(content) for content in chunks[-1]._contents):
                chunks[-1]._nextindex.append(0)
                chunks[-1]._contents.append(newchunk())

            for tag in range(len(chunks[-1]._contents)):
                if ismine(chunks[-1]._contents[tag]):
                    nextindex = chunks[-1]._nextindex[tag]
                    chunks[-1]._nextindex[tag] += 1

                    chunks[-1]._contents[tag] = promote(chunks[-1]._contents[tag])
                    fillobj(obj, chunks[-1]._contents[tag], nextindex)

                    chunks[-1]._tags[offsets[-1] - offsets[-2]] = tag
                    chunks[-1]._index[offsets[-1] - offsets[-2]] = nextindex

                    offsets[-1] += 1
                    break

    def fill(obj, chunks, offsets):
        if obj is None:
            # anything with None -> IndexedMaskedArray

            if len(chunks) == 0 or offsets[-1] - offsets[-2] == len(chunks[-1]):
                chunks.append(IndexedMaskedArray(numpy.empty(chunksize, dtype=awkward.array.base.AwkwardArray.INDEXTYPE), []))
                chunks[-1]._nextindex = 0
                offsets.append(offsets[-1])

            if not isinstance(chunks[-1], IndexedMaskedArray):
                chunks[-1] = IndexedMaskedArray(numpy.empty(chunksize, dtype=awkward.array.base.AwkwardArray.INDEXTYPE), chunks[-1])
                chunks[-1]._index[: offsets[-1] - offsets[-2]] = numpy.arange(offsets[-1] - offsets[-2], dtype=awkward.array.base.AwkwardArray.INDEXTYPE)
                chunks[-1]._nextindex = offsets[-1] - offsets[-2]

            chunks[-1]._index[offsets[-1] - offsets[-2]] = chunks[-1]._maskedwhen
            offsets[-1] += 1

        elif isinstance(obj, (bool, numpy.bool, numpy.bool_)):
            # bool -> Numpy bool_

            def newchunk():
                return numpy.empty(chunksize, dtype=numpy.bool_)

            def ismine(x):
                return isinstance(x, numpy.ndarray) and x.dtype == numpy.dtype(numpy.bool_)

            def promote(x):
                return x

            def fillobj(obj, array, where):
                array[where] = obj

            insert(obj, chunks, offsets, newchunk, ismine, promote, fillobj)

        elif isinstance(obj, (numbers.Integral, numpy.integer)):
            # int -> Numpy int64, float64, or complex128 (promotes to largest)

            def newchunk():
                return numpy.empty(chunksize, dtype=numpy.int64)

            def ismine(x):
                return isinstance(x, numpy.ndarray) and issubclass(x.dtype.type, numpy.number)

            def promote(x):
                return x

            def fillobj(obj, array, where):
                array[where] = obj

            insert(obj, chunks, offsets, newchunk, ismine, promote, fillobj)

        elif isinstance(obj, (numbers.Real, numpy.floating)):
            # float -> Numpy int64, float64, or complex128 (promotes to largest)

            def newchunk():
                return numpy.empty(chunksize, dtype=numpy.int64)

            def ismine(x):
                return isinstance(x, numpy.ndarray) and issubclass(x.dtype.type, numpy.number)

            def promote(x):
                if issubclass(x.dtype.type, numpy.floating):
                    return x
                else:
                    return x.astype(numpy.float64)

            def fillobj(obj, array, where):
                array[where] = obj

            insert(obj, chunks, offsets, newchunk, ismine, promote, fillobj)

        elif isinstance(obj, (numbers.Complex, numpy.complex, numpy.complexfloating)):
            # complex -> Numpy int64, float64, or complex128 (promotes to largest)

            def newchunk():
                return numpy.empty(chunksize, dtype=numpy.complex128)

            def ismine(x):
                return isinstance(x, numpy.ndarray) and issubclass(x.dtype.type, numpy.number)

            def promote(x):
                if issubclass(x.dtype.type, numpy.complexfloating):
                    return x
                else:
                    return x.astype(numpy.complex128)

            def fillobj(obj, array, where):
                array[where] = obj

            insert(obj, chunks, offsets, newchunk, ismine, promote, fillobj)

        elif isinstance(obj, bytes):
            # bytes -> VirtualObjectArray of JaggedArray

            def newchunk():
                out = VirtualObjectArray(tobytes, JaggedArray.fromoffsets(
                    numpy.zeros(chunksize + 1, dtype=awkward.array.base.AwkwardArray.INDEXTYPE),
                    AppendableArray.empty(lambda: numpy.empty(chunksize, dtype=awkward.array.base.AwkwardArray.CHARTYPE))))
                out._content._starts[0] = 0
                return out

            def ismine(x):
                return isinstance(x, VirtualObjectArray) and (x._generator is tobytes or x._generator is tostring)

            def promote(x):
                return x

            def fillobj(obj, array, where):
                array._content._stops[where] = array._content._starts[where] + len(obj)
                array._content._content.extend(numpy.fromstring(obj, dtype=awkward.array.base.AwkwardArray.CHARTYPE))
                
            insert(obj, chunks, offsets, newchunk, ismine, promote, fillobj)

        elif isinstance(obj, awkward.util.string):
            # str -> VirtualObjectArray of JaggedArray

            def newchunk():
                out = VirtualObjectArray(tostring, JaggedArray.fromoffsets(
                    numpy.zeros(chunksize + 1, dtype=awkward.array.base.AwkwardArray.INDEXTYPE),
                    AppendableArray.empty(lambda: numpy.empty(chunksize, dtype=awkward.array.base.AwkwardArray.CHARTYPE))))
                out._content._starts[0] = 0
                return out

            def ismine(x):
                return isinstance(x, VirtualObjectArray) and (x._generator is tobytes or x._generator is tostring)

            def promote(x):
                if x._generator is tostring:
                    return x
                else:
                    return VirtualObjectArray(tostring, x._content)

            def fillobj(obj, array, where):
                bytes = codecs.utf_8_encode(obj)[0]
                array._content._stops[where] = array._content._starts[where] + len(bytes)
                array._content._content.extend(numpy.fromstring(bytes, dtype=awkward.array.base.AwkwardArray.CHARTYPE))
                
            insert(obj, chunks, offsets, newchunk, ismine, promote, fillobj)

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
                # iterable -> JaggedArray (and recurse)

                def newchunk():
                    out = JaggedArray.fromoffsets(numpy.zeros(chunksize + 1, dtype=awkward.array.base.AwkwardArray.INDEXTYPE), PartitionedArray([0], []))
                    out._starts[0] = 0
                    out._content._offsets = [0]  # as an appendable list, not a Numpy array
                    return out

                def ismine(x):
                    return isinstance(x, JaggedArray)

                def promote(x):
                    return x

                def fillobj(obj, array, where):
                    array._stops[where] = array._starts[where]
                    for x in it:
                        fill(x, array._content._chunks, array._content._offsets)
                        array._stops[where] += 1

                insert(obj, chunks, offsets, newchunk, ismine, promote, fillobj)

    def trim(length, array):
        if isinstance(array, numpy.ndarray):
            if len(array) == length:
                return array                          # the length is right: don't copy it
            else:
                return numpy.array(array[:length])    # copy so that the base can be deleted

        elif isinstance(array, PartitionedArray):
            for i in range(len(array._chunks)):
                array._chunks[i] = trim(array._offsets[i + 1] - array._offsets[i], array._chunks[i])
            return array

        elif isinstance(array, IndexedMaskedArray):
            index = trim(length, array._index)
            selection = (index != array._maskedwhen)
            content = trim(index[selection][-1] + 1, array._content)

            if isinstance(content, numpy.ndarray):
                # for simple types, IndexedMaskedArray wastes space; convert to an Arrow-like BitMaskedArray
                mask = numpy.zeros(length, dtype=awkward.array.base.AwkwardArray.MASKTYPE)
                mask[selection] = True

                newcontent = numpy.empty(length, dtype=content.dtype)
                newcontent[selection] = content

                return BitMaskedArray.fromboolmask(mask, newcontent, maskedwhen=False, lsb=True)

            else:
                # for complex types, IndexedMaskedArray saves space; keep it
                return IndexedMaskedArray(index, content)

        elif isinstance(array, UnionArray):
            tags = trim(length, array._tags)
            index = trim(length, array._index)

            contents = []
            for tag, content in enumerate(array._contents):
                length = index[tags == tag][-1] + 1
                contents.append(trim(length, content))

            return UnionArray(tags, index, contents)

        elif isinstance(array, JaggedArray):
            offsets = array.offsets                   # fill creates aliased starts/stops
            if len(offsets) != length + 1:
                offsets = numpy.array(offsets[: length + 1])

            return JaggedArray.fromoffsets(offsets, trim(offsets[-1], array._content))

        elif isinstance(array, Table):
            return array   # FIXME

        elif isinstance(array, VirtualObjectArray):
            return VirtualObjectArray(array._generator, trim(length, array._content))

        else:
            raise AssertionError(array)

    chunks = []
    offsets = [0]
    length = 0
    for x in iterable:
        fill(x, chunks, offsets)
        length += 1
        
    return trim(length, PartitionedArray(offsets, chunks))
