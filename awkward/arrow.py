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
import awkward.type
import awkward.util

ARROW_BITMASKTYPE = awkward.util.numpy.uint8
ARROW_INDEXTYPE = awkward.util.numpy.int32
ARROW_TAGTYPE = awkward.util.numpy.uint8
ARROW_CHARTYPE = awkward.util.numpy.uint8

def schema2type(schema):


    def recurse(tpe):
        if isinstance(tpe, pyarrow.lib.DictionaryType):
            return recurse(tpe.dictionary.type)

        elif isinstance(tpe, pyarrow.lib.StructType):
            out = None
            for i in range(tpe.num_children):
                x = awkward.type.ArrayType(tpe[i].name, recurse(tpe[i].type))
                if out is None:
                    out = x
                else:
                    out = out & x
            return out

        elif isinstance(tpe, pyarrow.lib.ListType):
            return awkward.type.ArrayType(float("inf"), recurse(tpe.value_type))

        elif isinstance(tpe, pyarrow.lib.UnionType):
            out = None
            for i in range(tpe.num_children):
                x = recurse(tpe[i].type)
                if out is None:
                    out = x
                else:
                    out = out | x
            return out

        elif tpe == pyarrow.string():
            raise NotImplementedError

        elif tpe == pyarrow.binary():
            raise NotImplementedError

        elif tpe == pyarrow.bool_():
            raise NotImplementedError

        elif isinstance(tpe, pyarrow.lib.DataType):
            tpe.to_pandas_dtype()
            raise NotImplementedError

        else:
            raise NotImplementedError(repr(tpe))





def view(obj):
    import pyarrow

    def popbuffers(tpe, buffers):
        if isinstance(tpe, pyarrow.lib.DictionaryType):
            content = view(tpe.dictionary)
            index = popbuffers(tpe.index_type, buffers)
            if isinstance(index, awkward.array.masked.BitMaskedArray):
                return awkward.array.masked.BitMaskedArray(index.mask, awkward.array.indexed.IndexedArray(index.content, content), maskedwhen=index.maskedwhen, lsborder=index.lsborder)
            else:
                return awkward.array.indexed.IndexedArray(index, content)

        elif isinstance(tpe, pyarrow.lib.StructType):
            pairs = []
            for i in range(tpe.num_children - 1, -1, -1):
                pairs.insert(0, (tpe[i].name, popbuffers(tpe[i].type, buffers)))
            out = awkward.array.table.Table.frompairs(pairs)
            mask = buffers.pop()
            if mask is not None:
                mask = awkward.util.numpy.frombuffer(mask, dtype=ARROW_BITMASKTYPE)
                return awkward.array.masked.BitMaskedArray(mask, out, maskedwhen=False, lsborder=True)
            else:
                return out

        elif isinstance(tpe, pyarrow.lib.ListType):
            content = popbuffers(tpe.value_type, buffers)
            offsets = awkward.util.numpy.frombuffer(buffers.pop(), dtype=ARROW_INDEXTYPE)
            out = awkward.array.jagged.JaggedArray.fromoffsets(offsets, content)
            mask = buffers.pop()
            if mask is not None:
                mask = awkward.util.numpy.frombuffer(mask, dtype=ARROW_BITMASKTYPE)
                return awkward.array.masked.BitMaskedArray(mask, out, maskedwhen=False, lsborder=True)
            else:
                return out

        elif isinstance(tpe, pyarrow.lib.UnionType) and tpe.mode == "sparse":
            contents = []
            for i in range(tpe.num_children - 1, -1, -1):
                contents.insert(0, popbuffers(tpe[i].type, buffers))
            assert buffers.pop() is None
            tags = awkward.util.numpy.frombuffer(buffers.pop(), dtype=ARROW_TAGTYPE)
            index = awkward.util.numpy.arange(len(tags), dtype=ARROW_INDEXTYPE)
            out = awkward.array.union.UnionArray(tags, index, contents)
            mask = buffers.pop()
            if mask is not None:
                mask = awkward.util.numpy.frombuffer(mask, dtype=ARROW_BITMASKTYPE)
                return awkward.array.masked.BitMaskedArray(mask, out, maskedwhen=False, lsborder=True)
            else:
                return out

        elif isinstance(tpe, pyarrow.lib.UnionType) and tpe.mode == "dense":
            contents = []
            for i in range(tpe.num_children - 1, -1, -1):
                contents.insert(0, popbuffers(tpe[i].type, buffers))
            index = awkward.util.numpy.frombuffer(buffers.pop(), dtype=ARROW_INDEXTYPE)
            tags = awkward.util.numpy.frombuffer(buffers.pop(), dtype=ARROW_TAGTYPE)
            out = awkward.array.union.UnionArray(tags, index, contents)
            mask = buffers.pop()
            if mask is not None:
                mask = awkward.util.numpy.frombuffer(mask, dtype=ARROW_BITMASKTYPE)
                return awkward.array.masked.BitMaskedArray(mask, out, maskedwhen=False, lsborder=True)
            else:
                return out

        elif tpe == pyarrow.string():
            content = awkward.util.numpy.frombuffer(buffers.pop(), dtype=ARROW_CHARTYPE)
            offsets = awkward.util.numpy.frombuffer(buffers.pop(), dtype=ARROW_INDEXTYPE)
            out = awkward.derived.strings.StringArray.fromoffsets(offsets, content, encoding="utf-8")
            mask = buffers.pop()
            if mask is not None:
                mask = awkward.util.numpy.frombuffer(mask, dtype=ARROW_BITMASKTYPE)
                return awkward.array.masked.BitMaskedArray(mask, out, maskedwhen=False, lsborder=True)
            else:
                return out

        elif tpe == pyarrow.binary():
            content = awkward.util.numpy.frombuffer(buffers.pop(), dtype=ARROW_CHARTYPE)
            offsets = awkward.util.numpy.frombuffer(buffers.pop(), dtype=ARROW_INDEXTYPE)
            out = awkward.derived.strings.StringArray.fromoffsets(offsets, content, encoding=None)
            mask = buffers.pop()
            if mask is not None:
                mask = awkward.util.numpy.frombuffer(mask, dtype=ARROW_BITMASKTYPE)
                return awkward.array.masked.BitMaskedArray(mask, out, maskedwhen=False, lsborder=True)
            else:
                return out

        elif tpe == pyarrow.bool_():
            out = awkward.util.numpy.unpackbits(awkward.util.numpy.frombuffer(buffers.pop(), dtype=ARROW_CHARTYPE)).view(awkward.util.BOOLTYPE)
            out = out.reshape(-1, 8)[:,::-1].reshape(-1)    # lsborder=True
            mask = buffers.pop()
            if mask is not None:
                mask = awkward.util.numpy.frombuffer(mask, dtype=ARROW_BITMASKTYPE)
                return awkward.array.masked.BitMaskedArray(mask, out, maskedwhen=False, lsborder=True)
            else:
                return out

        elif isinstance(tpe, pyarrow.lib.DataType):
            out = awkward.util.numpy.frombuffer(buffers.pop(), dtype=tpe.to_pandas_dtype())
            mask = buffers.pop()
            if mask is not None:
                mask = awkward.util.numpy.frombuffer(mask, dtype=ARROW_BITMASKTYPE)
                return awkward.array.masked.BitMaskedArray(mask, out, maskedwhen=False, lsborder=True)
            else:
                return out

        else:
            raise NotImplementedError(repr(tpe))

    if isinstance(obj, pyarrow.lib.Array):
        buffers = obj.buffers()
        out = popbuffers(obj.type, buffers)[:len(obj)]
        assert len(buffers) == 0
        return out

    elif isinstance(obj, pyarrow.lib.ChunkedArray):
        chunks = [x for x in obj.chunks if len(x) > 0]
        if len(chunks) == 1:
            return chunks[0]
        else:
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
        if len(chunks) == 1:
            return chunks[0]
        else:
            return awkward.array.chunked.ChunkedArray(chunks, counts=counts)

    else:
        raise NotImplementedError(type(obj))
