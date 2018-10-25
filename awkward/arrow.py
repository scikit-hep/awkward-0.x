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

import awkward.array.chunked
import awkward.array.indexed
import awkward.array.jagged
import awkward.array.masked
import awkward.array.table
import awkward.derived.strings
import awkward.util

ARROW_BITMASKTYPE = awkward.util.numpy.uint8
ARROW_INDEXTYPE = awkward.util.numpy.int32
ARROW_TAGTYPE = awkward.util.numpy.uint8
ARROW_CHARTYPE = awkward.util.numpy.uint8

def view(obj):
    import pyarrow

    def recurse(tpe, buffers):
        if isinstance(tpe, pyarrow.lib.DictionaryType):
            content = view(tpe.dictionary)
            index = recurse(tpe.index_type, buffers)
            assert isinstance(index, awkward.array.masked.BitMaskedArray)
            return awkward.array.masked.BitMaskedArray(index.mask, awkward.array.indexed.IndexedArray(index.content, content), maskedwhen=index.maskedwhen, lsborder=index.lsborder)

        elif isinstance(tpe, pyarrow.lib.StructType):
            pairs = []
            for i in range(tpe.num_children - 1, -1, -1):
                pairs.insert(0, (tpe[i].name, recurse(tpe[i].type, buffers)))
            mask = awkward.util.numpy.frombuffer(buffers.pop(), dtype=ARROW_BITMASKTYPE)
            return awkward.array.masked.BitMaskedArray(mask, awkward.array.table.Table.frompairs(pairs), maskedwhen=False, lsborder=True)

        elif isinstance(tpe, pyarrow.lib.ListType):
            content = recurse(tpe.value_type, buffers)
            offsets = awkward.util.numpy.frombuffer(buffers.pop(), dtype=ARROW_INDEXTYPE)
            mask = awkward.util.numpy.frombuffer(buffers.pop(), dtype=ARROW_BITMASKTYPE)
            return awkward.array.masked.BitMaskedArray(mask, awkward.array.jagged.JaggedArray.fromoffsets(offsets, content), maskedwhen=False, lsborder=True)

        elif isinstance(tpe, pyarrow.lib.UnionType) and tpe.mode == "sparse":
            contents = []
            for i in range(tpe.num_children - 1, -1, -1):
                contents.insert(0, recurse(tpe[i].type, buffers))
            assert buffers.pop() is None
            tags = awkward.util.numpy.frombuffer(buffers.pop(), dtype=ARROW_TAGTYPE)
            index = awkward.util.numpy.arange(len(tags), dtype=ARROW_INDEXTYPE)
            mask = awkward.util.numpy.frombuffer(buffers.pop(), dtype=ARROW_BITMASKTYPE)
            return awkward.array.masked.BitMaskedArray(mask, awkward.array.union.UnionArray(tags, index, contents), maskedwhen=False, lsborder=True)

        elif isinstance(tpe, pyarrow.lib.UnionType) and tpe.mode == "dense":
            contents = []
            for i in range(tpe.num_children - 1, -1, -1):
                contents.insert(0, recurse(tpe[i].type, buffers))
            index = awkward.util.numpy.frombuffer(buffers.pop(), dtype=ARROW_INDEXTYPE)
            tags = awkward.util.numpy.frombuffer(buffers.pop(), dtype=ARROW_TAGTYPE)
            mask = awkward.util.numpy.frombuffer(buffers.pop(), dtype=ARROW_BITMASKTYPE)
            return awkward.array.masked.BitMaskedArray(mask, awkward.array.union.UnionArray(tags, index, contents), maskedwhen=False, lsborder=True)

        elif tpe == pyarrow.string():
            content = awkward.util.numpy.frombuffer(buffers.pop(), dtype=ARROW_CHARTYPE)
            offsets = awkward.util.numpy.frombuffer(buffers.pop(), dtype=ARROW_INDEXTYPE)
            mask = awkward.util.numpy.frombuffer(buffers.pop(), dtype=ARROW_BITMASKTYPE)
            return awkward.array.masked.BitMaskedArray(mask, awkward.derived.strings.StringArray.fromoffsets(offsets, content, encoding="utf-8"), maskedwhen=False, lsborder=True)

        elif tpe == pyarrow.binary():
            content = awkward.util.numpy.frombuffer(buffers.pop(), dtype=ARROW_CHARTYPE)
            offsets = awkward.util.numpy.frombuffer(buffers.pop(), dtype=ARROW_INDEXTYPE)
            mask = awkward.util.numpy.frombuffer(buffers.pop(), dtype=ARROW_BITMASKTYPE)
            return awkward.array.masked.BitMaskedArray(mask, awkward.derived.strings.StringArray.fromoffsets(offsets, content, encoding=None), maskedwhen=False, lsborder=True)

        elif tpe == pyarrow.bool_():
            content = awkward.util.numpy.unpackbits(awkward.util.numpy.frombuffer(buffers.pop(), dtype=ARROW_CHARTYPE)).view(awkward.util.BOOLTYPE)
            content = content.reshape(-1, 8)[:,::-1].reshape(-1)    # lsborder=True
            mask = awkward.util.numpy.frombuffer(buffers.pop(), dtype=ARROW_BITMASKTYPE)
            return awkward.array.masked.BitMaskedArray(mask, content, maskedwhen=False, lsborder=True)

        elif isinstance(tpe, pyarrow.lib.DataType):
            content = awkward.util.numpy.frombuffer(buffers.pop(), dtype=tpe.to_pandas_dtype())
            mask = awkward.util.numpy.frombuffer(buffers.pop(), dtype=ARROW_BITMASKTYPE)
            return awkward.array.masked.BitMaskedArray(mask, content, maskedwhen=False, lsborder=True)

        else:
            raise NotImplementedError(repr(tpe))

    if isinstance(obj, pyarrow.lib.Array):
        buffers = obj.buffers()
        out = recurse(obj.type, buffers)[:len(obj)]
        assert len(buffers) == 0
        return out

    elif isinstance(obj, pyarrow.lib.ChunkedArray):
        chunks = [x for x in obj.chunks if len(x) > 0]
        return awkward.array.chunked.ChunkedArray([view(x) for x in chunks], counts=[len(x) for x in chunks])

    elif isinstance(obj, pyarrow.lib.RecordBatch):
        out = awkward.array.table.Table()
        for n, x in zip(obj.schema.names, obj.columns):
            out[n] = view(x)
        return out

    elif isinstance(obj, pyarrow.lib.Table):
        chunks = []
        counts = []
        for batch in obj.to_batches():
            chunk = view(batch)
            if len(chunk) > 0:
                chunks.append(chunk)
                counts.append(len(chunk))
        return awkward.array.chunked.ChunkedArray(chunks, counts=counts)

    else:
        raise NotImplementedError(type(obj))
