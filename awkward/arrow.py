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

import json

import numpy

import awkward.array.base
import awkward.array.chunked
import awkward.array.indexed
import awkward.array.jagged
import awkward.array.masked
import awkward.array.objects
import awkward.array.table
import awkward.array.virtual
import awkward.type
import awkward.util

################################################################################ type conversions

def schema2type(schema):
    import pyarrow

    def recurse(tpe, nullable):
        if isinstance(tpe, pyarrow.lib.DictionaryType):
            out = recurse(tpe.dictionary.type, nullable)
            if nullable:
                return awkward.type.OptionType(out)
            else:
                return out

        elif isinstance(tpe, pyarrow.lib.StructType):
            out = None
            for i in range(tpe.num_children):
                x = awkward.type.ArrayType(tpe[i].name, recurse(tpe[i].type, tpe[i].nullable))
                if out is None:
                    out = x
                else:
                    out = out & x
            if nullable:
                return awkward.type.OptionType(out)
            else:
                return out

        elif isinstance(tpe, pyarrow.lib.ListType):
            out = awkward.type.ArrayType(float("inf"), recurse(tpe.value_type, nullable))
            if nullable:
                return awkward.type.OptionType(out)
            else:
                return out

        elif isinstance(tpe, pyarrow.lib.UnionType):
            out = None
            for i in range(tpe.num_children):
                x = recurse(tpe[i].type, nullable)
                if out is None:
                    out = x
                else:
                    out = out | x
            if nullable:
                return awkward.type.OptionType(out)
            else:
                return out

        elif tpe == pyarrow.string():
            if nullable:
                return awkward.type.OptionType(str)
            else:
                return str

        elif tpe == pyarrow.binary():
            if nullable:
                return awkward.type.OptionType(bytes)
            else:
                return bytes

        elif tpe == pyarrow.bool_():
            out = awkward.numpy.dtype(bool)
            if nullable:
                return awkward.type.OptionType(out)
            else:
                return out
            
        elif isinstance(tpe, pyarrow.lib.DataType):
            if nullable:
                return awkward.type.OptionType(tpe.to_pandas_dtype())
            else:
                return tpe.to_pandas_dtype()

        else:
            raise NotImplementedError(repr(tpe))

    out = None
    for name in schema.names:
        field = schema.field_by_name(name)
        mytype = awkward.type.ArrayType(name, recurse(field.type, field.nullable))
        if out is None:
            out = mytype
        else:
            out = out & mytype

    return out

################################################################################ value conversions

def toarrow(obj):
    import pyarrow

    def recurse(obj, mask):
        if isinstance(obj, numpy.ndarray):
            return pyarrow.array(obj, mask=mask)

        elif isinstance(obj, awkward.array.chunked.ChunkedArray):   # includes AppendableArray
            raise TypeError("only top-level ChunkedArrays can be converted to Arrow (as RecordBatches)")

        elif isinstance(obj, awkward.array.indexed.IndexedArray):
            if mask is None:
                return pyarrow.DictionaryArray.from_arrays(obj.index, recurse(obj.content, mask))
            else:
                return recurse(obj.content[obj.index], mask)

        elif isinstance(obj, awkward.array.indexed.SparseArray):
            return recurse(obj.dense, mask)

        elif isinstance(obj, awkward.array.jagged.JaggedArray):
            obj = obj.compact()
            if mask is not None:
                mask = obj.tojagged(mask).flatten()
            return pyarrow.ListArray.from_arrays(obj.offsets, recurse(obj.content, mask))

        elif isinstance(obj, awkward.array.masked.IndexedMaskedArray):
            thismask = obj.boolmask(maskedwhen=True)
            if mask is not None:
                thismask = mask & thismask
            if len(obj.content) == 0:
                content = obj.numpy.empty(len(obj.mask), dtype=obj.DEFAULTTYPE)
            else:
                content = obj.content[obj.mask]
            return recurse(content, thismask)

        elif isinstance(obj, awkward.array.masked.MaskedArray):   # includes BitMaskedArray
            thismask = obj.boolmask(maskedwhen=True)
            if mask is not None:
                thismask = mask & thismask
            return recurse(obj.content, thismask)

        elif isinstance(obj, awkward.array.objects.ObjectArray):
            # throw away Python object interpretation, which Arrow can't handle while being multilingual
            return recurse(obj.content, mask)

        elif isinstance(obj, awkward.array.objects.StringArray):
            # obj = obj.compact()
            raise NotImplementedError("I don't know how to make an Arrow StringArray")

            # I don't understand this
            # pyarrow.StringArray.from_buffers(2, pyarrow.py_buffer(numpy.array([0, 5, 10])), pyarrow.py_buffer(b"helloHELLO"), offset=0)
            # returns ["", "hello"]
            # ???

            # Also, be sure to check for the difference between strings and bytes!

        elif isinstance(obj, awkward.array.table.Table):
            return pyarrow.StructArray.from_arrays([recurse(x, mask) for x in obj.contents.values()], list(obj.contents))

        elif isinstance(obj, awkward.array.union.UnionArray):
            contents = []
            for i, x in enumerate(obj.contents):
                if mask is None:
                    thismask = None
                else:
                    thistags = (obj.tags == i)
                    thismask = obj.numpy.empty(len(x), dtype=obj.MASKTYPE)
                    thismask[obj.index[thistags]] = mask[thistags]    # hmm... obj.index could have repeats; the Arrow mask in that case would not be well-defined...
                contents.append(recurse(x, thismask))

            return pyarrow.UnionArray.from_dense(pyarrow.array(obj.tags.astype(numpy.int8)), pyarrow.array(obj.index.astype(numpy.int32)), contents)

        elif isinstance(obj, awkward.array.virtual.VirtualArray):
            return recurse(obj.array, mask)

        else:
            raise TypeError("cannot convert type {0} to Arrow".format(type(obj)))

    if isinstance(obj, awkward.array.chunked.ChunkedArray):   # includes AppendableArray
        batches = []
        for chunk in obj.chunks:
            arr = toarrow(chunk)
            if isinstance(arr, pyarrow.Table):
                batches.extend(arr.to_batches())
            else:
                batches.append(pyarrow.RecordBatch.from_arrays([arr], [""]))
        return pyarrow.Table.from_batches(batches)

    elif isinstance(obj, awkward.array.masked.IndexedMaskedArray) and isinstance(obj.content, awkward.array.table.Table):
        mask = obj.boolmask(maskedwhen=True)
        if len(obj.content) == 0:
            content = obj.numpy.empty(len(obj.mask), dtype=obj.DEFAULTTYPE)
        else:
            content = obj.content[obj.mask]
        return pyarrow.Table.from_batches([pyarrow.RecordBatch.from_arrays([recurse(x, mask) for x in obj.content.contents.values()], list(obj.content.contents))])

    elif isinstance(obj, awkward.array.masked.MaskedArray) and isinstance(obj.content, awkward.array.table.Table):   # includes BitMaskedArray
        mask = obj.boolmask(maskedwhen=True)
        return pyarrow.Table.from_batches([pyarrow.RecordBatch.from_arrays([recurse(x, mask) for x in obj.content.contents.values()], list(obj.content.contents))])

    elif isinstance(obj, awkward.array.table.Table):
        return pyarrow.Table.from_batches([pyarrow.RecordBatch.from_arrays([recurse(x, None) for x in obj.contents.values()], list(obj.contents))])

    else:
        return recurse(obj, None)

def fromarrow(obj, awkwardlib=None):
    import pyarrow
    awkwardlib = awkward.util.awkwardlib(awkwardlib)
    ARROW_BITMASKTYPE = awkwardlib.numpy.uint8
    ARROW_INDEXTYPE = awkwardlib.numpy.int32
    ARROW_TAGTYPE = awkwardlib.numpy.uint8
    ARROW_CHARTYPE = awkwardlib.numpy.uint8

    def popbuffers(tpe, buffers):
        if isinstance(tpe, pyarrow.lib.DictionaryType):
            content = fromarrow(tpe.dictionary)
            index = popbuffers(tpe.index_type, buffers)
            if isinstance(index, awkwardlib.BitMaskedArray):
                return awkwardlib.BitMaskedArray(index.mask, awkwardlib.IndexedArray(index.content, content), maskedwhen=index.maskedwhen, lsborder=index.lsborder)
            else:
                return awkwardlib.IndexedArray(index, content)

        elif isinstance(tpe, pyarrow.lib.StructType):
            pairs = []
            for i in range(tpe.num_children - 1, -1, -1):
                pairs.insert(0, (tpe[i].name, popbuffers(tpe[i].type, buffers)))
            out = awkwardlib.Table.frompairs(pairs)
            mask = buffers.pop()
            if mask is not None:
                mask = awkwardlib.numpy.frombuffer(mask, dtype=ARROW_BITMASKTYPE)
                return awkwardlib.BitMaskedArray(mask, out, maskedwhen=False, lsborder=True)
            else:
                return out

        elif isinstance(tpe, pyarrow.lib.ListType):
            content = popbuffers(tpe.value_type, buffers)
            offsets = awkwardlib.numpy.frombuffer(buffers.pop(), dtype=ARROW_INDEXTYPE)
            out = awkwardlib.JaggedArray.fromoffsets(offsets, content[:offsets[-1]])
            mask = buffers.pop()
            if mask is not None:
                mask = awkwardlib.numpy.frombuffer(mask, dtype=ARROW_BITMASKTYPE)
                return awkwardlib.BitMaskedArray(mask, out, maskedwhen=False, lsborder=True)
            else:
                return out

        elif isinstance(tpe, pyarrow.lib.UnionType) and tpe.mode == "sparse":
            contents = []
            for i in range(tpe.num_children - 1, -1, -1):
                contents.insert(0, popbuffers(tpe[i].type, buffers))
            assert buffers.pop() is None
            tags = awkwardlib.numpy.frombuffer(buffers.pop(), dtype=ARROW_TAGTYPE)
            index = awkwardlib.numpy.arange(len(tags), dtype=ARROW_INDEXTYPE)
            for i in range(len(contents)):
                these = index[tags == i]
                if len(these) == 0:
                    contents[i] = contents[i][0:0]
                else:
                    contents[i] = contents[i][: these[-1] + 1]
            out = awkwardlib.UnionArray(tags, index, contents)
            mask = buffers.pop()
            if mask is not None:
                mask = awkwardlib.numpy.frombuffer(mask, dtype=ARROW_BITMASKTYPE)
                return awkwardlib.BitMaskedArray(mask, out, maskedwhen=False, lsborder=True)
            else:
                return out

        elif isinstance(tpe, pyarrow.lib.UnionType) and tpe.mode == "dense":
            contents = []
            for i in range(tpe.num_children - 1, -1, -1):
                contents.insert(0, popbuffers(tpe[i].type, buffers))
            index = awkwardlib.numpy.frombuffer(buffers.pop(), dtype=ARROW_INDEXTYPE)
            tags = awkwardlib.numpy.frombuffer(buffers.pop(), dtype=ARROW_TAGTYPE)
            for i in range(len(contents)):
                these = index[tags == i]
                if len(these) == 0:
                    contents[i] = contents[i][0:0]
                else:
                    contents[i] = contents[i][: these.max() + 1]
            out = awkwardlib.UnionArray(tags, index, contents)
            mask = buffers.pop()
            if mask is not None:
                mask = awkwardlib.numpy.frombuffer(mask, dtype=ARROW_BITMASKTYPE)
                return awkwardlib.BitMaskedArray(mask, out, maskedwhen=False, lsborder=True)
            else:
                return out

        elif tpe == pyarrow.string():
            content = awkwardlib.numpy.frombuffer(buffers.pop(), dtype=ARROW_CHARTYPE)
            offsets = awkwardlib.numpy.frombuffer(buffers.pop(), dtype=ARROW_INDEXTYPE)
            out = awkwardlib.StringArray.fromoffsets(offsets, content[:offsets[-1]], encoding="utf-8")
            mask = buffers.pop()
            if mask is not None:
                mask = awkwardlib.numpy.frombuffer(mask, dtype=ARROW_BITMASKTYPE)
                return awkwardlib.BitMaskedArray(mask, out, maskedwhen=False, lsborder=True)
            else:
                return out

        elif tpe == pyarrow.binary():
            content = awkwardlib.numpy.frombuffer(buffers.pop(), dtype=ARROW_CHARTYPE)
            offsets = awkwardlib.numpy.frombuffer(buffers.pop(), dtype=ARROW_INDEXTYPE)
            out = awkwardlib.StringArray.fromoffsets(offsets, content[:offsets[-1]], encoding=None)
            mask = buffers.pop()
            if mask is not None:
                mask = awkwardlib.numpy.frombuffer(mask, dtype=ARROW_BITMASKTYPE)
                return awkwardlib.BitMaskedArray(mask, out, maskedwhen=False, lsborder=True)
            else:
                return out

        elif tpe == pyarrow.bool_():
            out = awkwardlib.numpy.unpackbits(awkwardlib.numpy.frombuffer(buffers.pop(), dtype=ARROW_CHARTYPE)).view(awkwardlib.MaskedArray.BOOLTYPE)
            out = out.reshape(-1, 8)[:,::-1].reshape(-1)    # lsborder=True
            mask = buffers.pop()
            if mask is not None:
                mask = awkwardlib.numpy.frombuffer(mask, dtype=ARROW_BITMASKTYPE)
                return awkwardlib.BitMaskedArray(mask, out, maskedwhen=False, lsborder=True)
            else:
                return out

        elif isinstance(tpe, pyarrow.lib.DataType):
            out = awkwardlib.numpy.frombuffer(buffers.pop(), dtype=tpe.to_pandas_dtype())
            mask = buffers.pop()
            if mask is not None:
                mask = awkwardlib.numpy.frombuffer(mask, dtype=ARROW_BITMASKTYPE)
                return awkwardlib.BitMaskedArray(mask, out, maskedwhen=False, lsborder=True)
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
            return awkwardlib.ChunkedArray([fromarrow(x) for x in chunks], counts=[len(x) for x in chunks])

    elif isinstance(obj, pyarrow.lib.RecordBatch):
        out = awkwardlib.Table()
        for n, x in zip(obj.schema.names, obj.columns):
            out[n] = fromarrow(x)
        return out

    elif isinstance(obj, pyarrow.lib.Table):
        chunks = []
        counts = []
        for batch in obj.to_batches():
            chunk = fromarrow(batch)
            if len(chunk) > 0:
                chunks.append(chunk)
                counts.append(len(chunk))
        if len(chunks) == 1:
            return chunks[0]
        else:
            return awkwardlib.ChunkedArray(chunks, counts=counts)

    else:
        raise NotImplementedError(type(obj))

################################################################################ Parquet file handling

def toparquet(obj, where, **options):
    import pyarrow.parquet

    options["where"] = where

    def convert(obj, message):
        if isinstance(obj, (awkward.array.base.AwkwardArray, numpy.ndarray)):
            out = toarrow(obj)
            if isinstance(out, pyarrow.Table):
                return out
            else:
                return pyarrow.Table.from_batches([pyarrow.RecordBatch.from_arrays([out], [""])])
        else:
            raise TypeError(message)

    if isinstance(obj, awkward.array.chunked.ChunkedArray):
        obj = iter(obj.chunks)
        try:
            awkitem = next(obj)
        except StopIteration:
            raise ValueError("iterable is empty")

        arritem = convert(awkitem, None)
        if "schema" not in options:
            options["schema"] = arritem.schema
        writer = pyarrow.parquet.ParquetWriter(**options)
        writer.write_table(arritem)

        try:
            while True:
                try:
                    awkitem = next(obj)
                except StopIteration:
                    break
                else:
                    writer.write_table(convert(awkitem, None))
        finally:
            writer.close()

    elif isinstance(obj, (awkward.array.base.AwkwardArray, numpy.ndarray)):
        arritem = convert(obj, None)
        options["schema"] = arritem.schema
        writer = pyarrow.parquet.ParquetWriter(**options)
        writer.write_table(arritem)
        writer.close()

    else:
        try:
            obj = iter(obj)
        except TypeError:
            raise TypeError("cannot write {0} to Parquet file".format(type(obj)))
        try:
            awkitem = next(obj)
        except StopIteration:
            raise ValueError("iterable is empty")

        arritem = convert(awkitem, "cannot write iterator of {0} to Parquet file".format(type(awkitem)))
        if "schema" not in options:
            options["schema"] = arritem.schema
        writer = pyarrow.parquet.ParquetWriter(**options)
        writer.write_table(arritem)

        try:
            while True:
                try:
                    awkitem = next(obj)
                except StopIteration:
                    break
                else:
                    writer.write_table(convert(awkitem, "cannot write iterator of {0} to Parquet file".format(type(awkitem))))
        finally:
            writer.close()

class _ParquetFile(object):
    def __init__(self, file, cache=None, metadata=None, common_metadata=None):
        self.file = file
        self.cache = cache
        self.metadata = metadata
        self.common_metadata = common_metadata
        self._init()

    def _init(self):
        import pyarrow.parquet
        self.parquetfile = pyarrow.parquet.ParquetFile(self.file, metadata=self.metadata, common_metadata=self.common_metadata)
        self.type = schema2type(self.parquetfile.schema.to_arrow_schema())
        
    def __getstate__(self):
        return {"file": self.file, "metadata": self.metadata, "common_metadata": self.common_metadata}

    def __setstate__(self, state):
        self.file = state["file"]
        self.cache = None
        self.metadata = state["metadata"]
        self.common_metadata = state["common_metadata"]
        self._init()

    def __call__(self, rowgroup, column):
        return fromarrow(self.parquetfile.read_row_group(rowgroup, columns=[column]))[column]

    def tojson(self):
        json.dumps([self.file, self.metadata, self.common_metadata])
        return {"file": self.file, "metadata": self.metadata, "common_metadata": self.common_metadata}

    @classmethod
    def fromjson(cls, state):
        return cls(state["file"], cache=None, metadata=state["metadata"], common_metadata=state["common_metadata"])

def fromparquet(file, awkwardlib=None, cache=None, persistvirtual=False, metadata=None, common_metadata=None):
    awkwardlib = awkward.util.awkwardlib(awkwardlib)
    parquetfile = _ParquetFile(file, cache=cache, metadata=metadata, common_metadata=common_metadata)
    columns = parquetfile.type.columns

    chunks = []
    counts = []
    for i in range(parquetfile.parquetfile.num_row_groups):
        numrows = parquetfile.parquetfile.metadata.row_group(i).num_rows
        if numrows > 0:
            if columns == [""]:
                chunk = awkwardlib.VirtualArray(parquetfile, (i, ""), cache=cache, type=awkwardlib.type.ArrayType(numrows, parquetfile.type[""]), persistvirtual=persistvirtual)
            else:
                chunk = awkwardlib.Table()
                for n in columns:
                    q = awkwardlib.VirtualArray(parquetfile, (i, n), cache=cache, type=awkwardlib.type.ArrayType(numrows, parquetfile.type[n]), persistvirtual=persistvirtual)
                    chunk.contents[n] = q

            chunks.append(chunk)
            counts.append(numrows)

    return awkwardlib.ChunkedArray(chunks, counts)
