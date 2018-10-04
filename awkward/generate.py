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
import collections
import numbers

import awkward.array.base
import awkward.util
from awkward.array.chunked import ChunkedArray, AppendableArray
from awkward.array.jagged import JaggedArray
from awkward.array.masked import BitMaskedArray, IndexedMaskedArray
from awkward.array.objects import ObjectArray
from awkward.array.table import Table
from awkward.array.union import UnionArray

# FIXME: the following must be totally broken from upstream changes

def fromiter(iterable, chunksize=1024, maskmissing=True, references=False):
    if references:
        raise NotImplementedError    # keep all ids in a hashtable to create pointers (IndexedArray)

    tobytes = lambda x: x.tobytes()
    tostring = lambda x: codecs.utf_8_decode(x.tobytes())[0]

    def insert(obj, chunks, offsets, newchunk, ismine, promote, fillobj):
        if len(chunks) == 0 or offsets[-1] - offsets[-2] == len(chunks[-1]):
            chunks.append(newchunk(obj))
            offsets.append(offsets[-1])

        if ismine(obj, chunks[-1]):
            chunks[-1] = promote(obj, chunks[-1])
            fillobj(obj, chunks[-1], offsets[-1] - offsets[-2])
            offsets[-1] += 1

        elif isinstance(chunks[-1], IndexedMaskedArray) and len(chunks[-1]._content) == 0:
            chunks[-1]._content = newchunk(obj)

            nextindex = chunks[-1]._nextindex
            chunks[-1]._nextindex += 1
            chunks[-1]._index[offsets[-1] - offsets[-2]] = nextindex

            chunks[-1]._content = promote(obj, chunks[-1]._content)
            fillobj(obj, chunks[-1]._content, nextindex)
            offsets[-1] += 1

        elif isinstance(chunks[-1], IndexedMaskedArray) and ismine(obj, chunks[-1]._content):
            nextindex = chunks[-1]._nextindex
            chunks[-1]._nextindex += 1
            chunks[-1]._index[offsets[-1] - offsets[-2]] = nextindex

            chunks[-1]._content = promote(obj, chunks[-1]._content)
            fillobj(obj, chunks[-1]._content, nextindex)
            offsets[-1] += 1

        elif isinstance(chunks[-1], UnionArray) and any(isinstance(content, IndexedMaskedArray) and ismine(obj, content._content) for content in chunks[-1]._contents):
            for tag in range(len(chunks[-1]._contents)):
                if isinstance(chunks[-1]._contents[tag], IndexedMaskedArray) and ismine(obj, chunks[-1]._contents[tag]._content):
                    nextindex_union = chunks[-1]._nextindex[tag]
                    chunks[-1]._nextindex[tag] += 1

                    nextindex_mask = chunks[-1]._contents[tag]._nextindex
                    chunks[-1]._contents[tag]._nextindex += 1
                    chunks[-1]._contents[tag]._index[nextindex_union] = nextindex_mask

                    chunks[-1]._contents[tag]._content = promote(obj, chunks[-1]._contents[tag]._content)
                    fillobj(obj, chunks[-1]._contents[tag]._content, nextindex_mask)

                    chunks[-1]._tags[offsets[-1] - offsets[-2]] = tag
                    chunks[-1]._index[offsets[-1] - offsets[-2]] = nextindex_union

                    offsets[-1] += 1
                    break

        else:
            if not isinstance(chunks[-1], UnionArray):
                chunks[-1] = UnionArray(numpy.empty(chunksize, dtype=awkward.array.base.AwkwardArray.INDEXTYPE),
                                        numpy.empty(chunksize, dtype=awkward.array.base.AwkwardArray.INDEXTYPE),
                                        [chunks[-1]])
                chunks[-1]._nextindex = [offsets[-1] - offsets[-2]]
                chunks[-1]._tags[: offsets[-1] - offsets[-2]] = 0
                chunks[-1]._index[: offsets[-1] - offsets[-2]] = numpy.arange(offsets[-1] - offsets[-2], dtype=awkward.array.base.AwkwardArray.INDEXTYPE)
                chunks[-1]._contents = list(chunks[-1]._contents)

            if not any(ismine(obj, content) for content in chunks[-1]._contents):
                chunks[-1]._nextindex.append(0)
                chunks[-1]._contents.append(newchunk(obj))

            for tag in range(len(chunks[-1]._contents)):
                if ismine(obj, chunks[-1]._contents[tag]):
                    nextindex = chunks[-1]._nextindex[tag]
                    chunks[-1]._nextindex[tag] += 1

                    chunks[-1]._contents[tag] = promote(obj, chunks[-1]._contents[tag])
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

            if isinstance(chunks[-1], UnionArray) and any(isinstance(content, IndexedMaskedArray) for content in chunks[-1]._contents):
                for tag in range(len(chunks[-1]._contents)):
                    if isinstance(chunks[-1]._contents[tag], IndexedMaskedArray):
                        nextindex = chunks[-1]._nextindex[tag]
                        chunks[-1]._nextindex[tag] += 1

                        chunks[-1]._contents[tag]._index[nextindex] = chunks[-1]._contents[tag]._maskedwhen

                        chunks[-1]._tags[offsets[-1] - offsets[-2]] = tag
                        chunks[-1]._index[offsets[-1] - offsets[-2]] = nextindex

                        offsets[-1] += 1
                        break

            else:
                if not isinstance(chunks[-1], IndexedMaskedArray):
                    chunks[-1] = IndexedMaskedArray(numpy.empty(chunksize, dtype=awkward.array.base.AwkwardArray.INDEXTYPE), chunks[-1])
                    chunks[-1]._index[: offsets[-1] - offsets[-2]] = numpy.arange(offsets[-1] - offsets[-2], dtype=awkward.array.base.AwkwardArray.INDEXTYPE)
                    chunks[-1]._nextindex = offsets[-1] - offsets[-2]

                chunks[-1]._index[offsets[-1] - offsets[-2]] = chunks[-1]._maskedwhen
                offsets[-1] += 1

        elif isinstance(obj, (bool, numpy.bool, numpy.bool_)):
            # bool -> Numpy bool_

            def newchunk(obj):
                return numpy.empty(chunksize, dtype=numpy.bool_)

            def ismine(obj, x):
                return isinstance(x, numpy.ndarray) and x.dtype == numpy.dtype(numpy.bool_)

            def promote(obj, x):
                return x

            def fillobj(obj, array, where):
                array[where] = obj

            insert(obj, chunks, offsets, newchunk, ismine, promote, fillobj)

        elif isinstance(obj, (numbers.Integral, numpy.integer)):
            # int -> Numpy int64, float64, or complex128 (promotes to largest)

            def newchunk(obj):
                return numpy.empty(chunksize, dtype=numpy.int64)

            def ismine(obj, x):
                return isinstance(x, numpy.ndarray) and issubclass(x.dtype.type, numpy.number)

            def promote(obj, x):
                return x

            def fillobj(obj, array, where):
                array[where] = obj

            insert(obj, chunks, offsets, newchunk, ismine, promote, fillobj)

        elif isinstance(obj, (numbers.Real, numpy.floating)):
            # float -> Numpy int64, float64, or complex128 (promotes to largest)

            def newchunk(obj):
                return numpy.empty(chunksize, dtype=numpy.int64)

            def ismine(obj, x):
                return isinstance(x, numpy.ndarray) and issubclass(x.dtype.type, numpy.number)

            def promote(obj, x):
                if issubclass(x.dtype.type, numpy.floating):
                    return x
                else:
                    return x.astype(numpy.float64)

            def fillobj(obj, array, where):
                array[where] = obj

            insert(obj, chunks, offsets, newchunk, ismine, promote, fillobj)

        elif isinstance(obj, (numbers.Complex, numpy.complex, numpy.complexfloating)):
            # complex -> Numpy int64, float64, or complex128 (promotes to largest)

            def newchunk(obj):
                return numpy.empty(chunksize, dtype=numpy.complex128)

            def ismine(obj, x):
                return isinstance(x, numpy.ndarray) and issubclass(x.dtype.type, numpy.number)

            def promote(obj, x):
                if issubclass(x.dtype.type, numpy.complexfloating):
                    return x
                else:
                    return x.astype(numpy.complex128)

            def fillobj(obj, array, where):
                array[where] = obj

            insert(obj, chunks, offsets, newchunk, ismine, promote, fillobj)

        elif isinstance(obj, bytes):
            # bytes -> ObjectArray of JaggedArray

            def newchunk(obj):
                out = ObjectArray(tobytes, JaggedArray.fromoffsets(
                    numpy.zeros(chunksize + 1, dtype=awkward.array.base.AwkwardArray.INDEXTYPE),
                    AppendableArray.empty(lambda: numpy.empty(chunksize, dtype=awkward.array.base.AwkwardArray.CHARTYPE))))
                out._content._starts[0] = 0
                return out

            def ismine(obj, x):
                return isinstance(x, ObjectArray) and (x._generator is tobytes or x._generator is tostring)

            def promote(obj, x):
                return x

            def fillobj(obj, array, where):
                array._content._stops[where] = array._content._starts[where] + len(obj)
                array._content._content.extend(numpy.fromstring(obj, dtype=awkward.array.base.AwkwardArray.CHARTYPE))
                
            insert(obj, chunks, offsets, newchunk, ismine, promote, fillobj)

        elif isinstance(obj, awkward.util.string):
            # str -> ObjectArray of JaggedArray

            def newchunk(obj):
                out = ObjectArray(tostring, JaggedArray.fromoffsets(
                    numpy.zeros(chunksize + 1, dtype=awkward.array.base.AwkwardArray.INDEXTYPE),
                    AppendableArray.empty(lambda: numpy.empty(chunksize, dtype=awkward.array.base.AwkwardArray.CHARTYPE))))
                out._content._starts[0] = 0
                return out

            def ismine(obj, x):
                return isinstance(x, ObjectArray) and (x._generator is tobytes or x._generator is tostring)

            def promote(obj, x):
                if x._generator is tostring:
                    return x
                else:
                    return ObjectArray(tostring, x._content)

            def fillobj(obj, array, where):
                bytes = codecs.utf_8_encode(obj)[0]
                array._content._stops[where] = array._content._starts[where] + len(bytes)
                array._content._content.extend(numpy.fromstring(bytes, dtype=awkward.array.base.AwkwardArray.CHARTYPE))
                
            insert(obj, chunks, offsets, newchunk, ismine, promote, fillobj)

        elif isinstance(obj, dict):
            # dict keys -> Table columns

            def newchunk(obj):
                return Table(chunksize, collections.OrderedDict((n, []) for n in obj))

            if maskmissing:
                def ismine(obj, x):
                    return isinstance(x, Table)

                def promote(obj, x):
                    for n in obj:
                        if not n in x._content:
                            x._content[n] = IndexedMaskedArray(numpy.empty(chunksize, dtype=awkward.array.base.AwkwardArray.INDEXTYPE), [])
                            x._content[n]._index[: offsets[-1] - offsets[-2]] = x._content[n]._maskedwhen
                            x._content[n]._nextindex = 0
                    return x

            else:
                def ismine(obj, x):
                    return isinstance(x, Table) and all(n in x._content for n in obj)

                def promote(obj, x):
                    return x

            def fillobj(obj, array, where):
                for n in obj:
                    if len(array._content[n]) == 0:
                        subchunks = []
                        suboffsets = [offsets[-2]]
                    else:
                        subchunks = [array._content[n]]
                        suboffsets = [offsets[-2], offsets[-1]]

                    fill(obj[n], subchunks, suboffsets)
                    array._content[n] = subchunks[-1]

            insert(obj, chunks, offsets, newchunk, ismine, promote, fillobj)

        elif isinstance(obj, tuple):
            # tuple items -> Table columns

            def newchunk(obj):
                return Table(chunksize, collections.OrderedDict(("_" + str(i), []) for i in range(len(obj))))

            def ismine(obj, x):
                return isinstance(x, Table) and list(x._content) == ["_" + str(i) for i in range(len(obj))]

            def promote(obj, x):
                return x

            def fillobj(obj, array, where):
                for i, x in enumerate(obj):
                    n = "_" + str(i)
                    if len(array._content[n]) == 0:
                        subchunks = []
                        suboffsets = [offsets[-2]]
                    else:
                        subchunks = [array._content[n]]
                        suboffsets = [offsets[-2], offsets[-1]]

                    fill(x, subchunks, suboffsets)
                    array._content[n] = subchunks[-1]

            insert(obj, chunks, offsets, newchunk, ismine, promote, fillobj)

        else:
            try:
                it = iter(obj)

            except TypeError:
                # object attributes -> Table columns

                def newchunk(obj):
                    return NamedTable(chunksize, obj.__class__.__name__, collections.OrderedDict((n, []) for n in dir(obj) if not n.startswith("_")))

                if maskmissing:
                    def ismine(obj, x):
                        return isinstance(x, NamedTable) and obj.__class__.__name__ == x._name

                    def promote(obj, x):
                        for n in dir(obj):
                            if not n.startswith("_") and not n in x._content:
                                x._content[n] = IndexedMaskedArray(numpy.empty(chunksize, dtype=awkward.array.base.AwkwardArray.INDEXTYPE), [])
                                x._content[n]._index[: offsets[-1] - offsets[-2]] = x._content[n]._maskedwhen
                                x._content[n]._nextindex = 0
                        return x

                else:
                    def ismine(obj, x):
                        return isinstance(x, NamedTable) and obj.__class__.__name__ == x._name and all(n in x._content for n in dir(obj) if not n.startswith("_"))

                    def promote(obj, x):
                        return x

                def fillobj(obj, array, where):
                    for n in dir(obj):
                        if not n.startswith("_"):
                            if len(array._content[n]) == 0:
                                subchunks = []
                                suboffsets = [offsets[-2]]
                            else:
                                subchunks = [array._content[n]]
                                suboffsets = [offsets[-2], offsets[-1]]

                            fill(getattr(obj, n), subchunks, suboffsets)
                            array._content[n] = subchunks[-1]

                insert(obj, chunks, offsets, newchunk, ismine, promote, fillobj)

            else:
                # iterable -> JaggedArray (and recurse)

                def newchunk(obj):
                    out = JaggedArray.fromoffsets(numpy.zeros(chunksize + 1, dtype=awkward.array.base.AwkwardArray.INDEXTYPE), PartitionedArray([0], []))
                    out._starts[0] = 0
                    out._content._offsets = [0]  # as an appendable list, not a Numpy array
                    return out

                def ismine(obj, x):
                    return isinstance(x, JaggedArray)

                def promote(obj, x):
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

        elif isinstance(array, NamedTable):
            return NamedTable(length, array._name, collections.OrderedDict((n, trim(length, x)) for n, x in array._content.items()))

        elif isinstance(array, Table):
            return Table(length, collections.OrderedDict((n, trim(length, x)) for n, x in array._content.items()))

        elif isinstance(array, ObjectArray):
            return ObjectArray(array._generator, trim(length, array._content))

        else:
            raise AssertionError(array)

    chunks = []
    offsets = [0]
    length = 0
    for x in iterable:
        fill(x, chunks, offsets)
        length += 1
        
    return trim(length, PartitionedArray(offsets, chunks))
