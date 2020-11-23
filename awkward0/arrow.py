#!/usr/bin/env python

# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-0.x/blob/master/LICENSE

import codecs
import json

import numpy

import awkward0.array.base
import awkward0.array.chunked
import awkward0.array.indexed
import awkward0.array.jagged
import awkward0.array.masked
import awkward0.array.objects
import awkward0.array.table
import awkward0.array.virtual
import awkward0.type
import awkward0.util

################################################################################ type conversions

def schema2type(schema):
    import pyarrow

    def recurse(tpe, nullable):
        if isinstance(tpe, pyarrow.lib.DictionaryType):
            out = recurse(tpe.dictionary.type, nullable)
            if nullable:
                return awkward0.type.OptionType(out)
            else:
                return out

        elif isinstance(tpe, pyarrow.lib.StructType):
            out = None
            for i in range(tpe.num_children):
                x = awkward0.type.ArrayType(tpe[i].name, recurse(tpe[i].type, tpe[i].nullable))
                if out is None:
                    out = x
                else:
                    out = out & x
            if nullable:
                return awkward0.type.OptionType(out)
            else:
                return out

        elif isinstance(tpe, pyarrow.lib.ListType):
            out = awkward0.type.ArrayType(float("inf"), recurse(tpe.value_type, nullable))
            if nullable:
                return awkward0.type.OptionType(out)
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
                return awkward0.type.OptionType(out)
            else:
                return out

        elif tpe == pyarrow.string():
            if nullable:
                return awkward0.type.OptionType(str)
            else:
                return str

        elif tpe == pyarrow.binary():
            if nullable:
                return awkward0.type.OptionType(bytes)
            else:
                return bytes

        elif tpe == pyarrow.bool_():
            out = awkward0.numpy.dtype(bool)
            if nullable:
                return awkward0.type.OptionType(out)
            else:
                return out

        elif isinstance(tpe, pyarrow.lib.DataType):
            if nullable:
                return awkward0.type.OptionType(tpe.to_pandas_dtype())
            else:
                return tpe.to_pandas_dtype()

        else:
            raise NotImplementedError(repr(tpe))

    out = None
    for name in schema.names:
        field = schema.field(name)
        mytype = awkward0.type.ArrayType(name, recurse(field.type, field.nullable))
        if out is None:
            out = mytype
        else:
            out = out & mytype

    return out

################################################################################ value conversions

# we need an opt-out of the large indices in certain cases, otherwise use by default
def toarrow(obj):
    import pyarrow

    def recurse(obj, mask):
        if isinstance(obj, numpy.ndarray):
            return pyarrow.array(obj, mask=mask)

        elif isinstance(obj, awkward0.array.chunked.ChunkedArray):   # includes AppendableArray
            raise TypeError("only top-level ChunkedArrays can be converted to Arrow (as RecordBatches)")

        elif isinstance(obj, awkward0.array.indexed.IndexedArray):
            if mask is None:
                return pyarrow.DictionaryArray.from_arrays(obj.index, recurse(obj.content, mask))
            else:
                return recurse(obj.content[obj.index], mask)

        elif isinstance(obj, awkward0.array.indexed.SparseArray):
            return recurse(obj.dense, mask)

        elif isinstance(obj, awkward0.array.jagged.JaggedArray):
            obj = obj.compact()
            if mask is not None:
                mask = obj.tojagged(mask).flatten()
            arrow_type = pyarrow.ListArray
            # 64bit offsets not yet completely golden in arrow
            # if hasattr(pyarrow, 'LargeListArray') and obj.starts.itemsize > 4:
            #     arrow_type = pyarrow.LargeListArray
            return arrow_type.from_arrays(obj.offsets, recurse(obj.content, mask))

        elif isinstance(obj, awkward0.array.masked.IndexedMaskedArray):
            thismask = obj.boolmask(maskedwhen=True)
            if mask is not None:
                thismask = mask | thismask
            if len(obj.content) == 0:
                content = obj.numpy.empty(len(obj.mask), dtype=obj.DEFAULTTYPE)
            else:
                content = obj.content[obj.mask]
            return recurse(content, thismask)

        elif isinstance(obj, awkward0.array.masked.MaskedArray):   # includes BitMaskedArray
            thismask = obj.boolmask(maskedwhen=True)
            if mask is not None:
                thismask = mask | thismask
            return recurse(obj.content, thismask)

        elif isinstance(obj, awkward0.array.objects.StringArray):
            if obj.encoding is None and hasattr(pyarrow.BinaryArray, 'from_buffers'):
                arrow_type = pyarrow.BinaryArray
                arrow_offset_type = pyarrow.binary()
                # 64bit offsets not yet completely golden in arrow
                # if hasattr(pyarrow, 'LargeBinaryArray') and obj.starts.itemsize > 4:
                #     arrow_type = pyarrow.LargeBinaryArray
                #     arrow_offset_type = pyarrow.large_binary()
                convert = lambda length, offsets, content: arrow_type.from_buffers(arrow_offset_type, length, [None, offsets, content])
            elif codecs.lookup(obj.encoding) is codecs.lookup("utf-8") or obj.encoding is None:
                arrow_type = pyarrow.StringArray
                # if hasattr(pyarrow, 'LargeStringArray') and obj.starts.itemsize > 4:
                #     arrow_type = pyarrow.LargeStringArray
                convert = lambda length, offsets, content: arrow_type.from_buffers(length, offsets, content)
            else:
                raise ValueError("only encoding=None or encoding='utf-8' can be converted to Arrow")

            obj = obj.compact()
            offsets = obj.offsets
            if offsets.dtype != numpy.dtype(numpy.int32):
                offsets = offsets.astype(numpy.int32)

            return convert(len(offsets) - 1, pyarrow.py_buffer(offsets), pyarrow.py_buffer(obj.content))

        elif isinstance(obj, awkward0.array.objects.ObjectArray):
            # throw away Python object interpretation, which Arrow can't handle while being multilingual
            return recurse(obj.content, mask)

        elif isinstance(obj, awkward0.array.table.Table):
            return pyarrow.StructArray.from_arrays([recurse(x, mask) for x in obj.contents.values()], list(obj.contents))

        elif isinstance(obj, awkward0.array.union.UnionArray):
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

        elif isinstance(obj, awkward0.array.virtual.VirtualArray):
            return recurse(obj.array, mask)

        else:
            raise TypeError("cannot convert type {0} to Arrow".format(type(obj)))

    if isinstance(obj, awkward0.array.chunked.ChunkedArray):   # includes AppendableArray
        batches = []
        for chunk in obj.chunks:
            arr = toarrow(chunk)
            if isinstance(arr, pyarrow.Table):
                batches.extend(arr.to_batches())
            else:
                batches.append(pyarrow.RecordBatch.from_arrays([arr], [""]))
        return pyarrow.Table.from_batches(batches)

    elif isinstance(obj, awkward0.array.masked.IndexedMaskedArray) and isinstance(obj.content, awkward0.array.table.Table):
        mask = obj.boolmask(maskedwhen=True)
        if len(obj.content) == 0:
            content = obj.numpy.empty(len(obj.mask), dtype=obj.DEFAULTTYPE)
        else:
            content = obj.content[obj.mask]
        return pyarrow.Table.from_batches([pyarrow.RecordBatch.from_arrays([recurse(x, mask) for x in obj.content.contents.values()], list(obj.content.contents))])

    elif isinstance(obj, awkward0.array.masked.MaskedArray) and isinstance(obj.content, awkward0.array.table.Table):   # includes BitMaskedArray
        mask = obj.boolmask(maskedwhen=True)
        return pyarrow.Table.from_batches([pyarrow.RecordBatch.from_arrays([recurse(x, mask) for x in obj.content.contents.values()], list(obj.content.contents))])

    elif isinstance(obj, awkward0.array.table.Table):
        return pyarrow.Table.from_batches([pyarrow.RecordBatch.from_arrays([recurse(x, None) for x in obj.contents.values()], list(obj.contents))])

    else:
        return recurse(obj, None)

def fromarrow(obj):
    import pyarrow
    awkwardlib = awkward0
    ARROW_BITMASKTYPE = awkwardlib.numpy.uint8
    ARROW_INDEXTYPE = awkwardlib.numpy.int32
    ARROW_LARGEINDEXTYPE = awkwardlib.numpy.int64
    ARROW_TAGTYPE = awkwardlib.numpy.uint8
    ARROW_CHARTYPE = awkwardlib.numpy.uint8

    def popbuffers(array, tpe, buffers, length):
        if isinstance(tpe, pyarrow.lib.DictionaryType):
            index = popbuffers(None if array is None else array.indices, tpe.index_type, buffers, length)
            if hasattr(tpe, "dictionary"):
                content = fromarrow(tpe.dictionary)
            elif array is not None:
                content = fromarrow(array.dictionary)
            else:
                raise NotImplementedError("no way to access Arrow dictionary inside of UnionArray")
            if isinstance(index, awkwardlib.BitMaskedArray):
                return awkwardlib.BitMaskedArray(index.mask, awkwardlib.IndexedArray(index.content, content), maskedwhen=index.maskedwhen, lsborder=index.lsborder)
            else:
                return awkwardlib.IndexedArray(index, content)

        elif isinstance(tpe, pyarrow.lib.StructType):
            assert getattr(tpe, "num_buffers", 1) == 1
            mask = buffers.pop(0)
            pairs = []
            for i in range(tpe.num_children):
                pairs.append((tpe[i].name, popbuffers(None if array is None else array.field(tpe[i].name), tpe[i].type, buffers, length)))
            out = awkwardlib.Table.frompairs(pairs, 0)   # FIXME: better rowstart
            if mask is not None:
                mask = awkwardlib.numpy.frombuffer(mask, dtype=ARROW_BITMASKTYPE)
                return awkwardlib.BitMaskedArray(mask, out, maskedwhen=False, lsborder=True)
            else:
                return out

        elif isinstance(tpe, pyarrow.lib.ListType):
            assert getattr(tpe, "num_buffers", 2) == 2
            mask = buffers.pop(0)
            offsets = awkwardlib.numpy.frombuffer(buffers.pop(0), dtype=ARROW_INDEXTYPE)[:length + 1]
            content = popbuffers(None if array is None else array.flatten(), tpe.value_type, buffers, offsets[-1])
            out = awkwardlib.JaggedArray.fromoffsets(offsets, content)
            if mask is not None:
                mask = awkwardlib.numpy.frombuffer(mask, dtype=ARROW_BITMASKTYPE)
                return awkwardlib.BitMaskedArray(mask, out, maskedwhen=False, lsborder=True)
            else:
                return out

        elif hasattr(pyarrow.lib, 'LargeListType') and isinstance(tpe, pyarrow.lib.LargeListType):
            assert getattr(tpe, "num_buffers", 2) == 2
            mask = buffers.pop(0)
            offsets = awkwardlib.numpy.frombuffer(buffers.pop(0), dtype=ARROW_LARGEINDEXTYPE)[:length + 1]
            content = popbuffers(None if array is None else array.flatten(), tpe.value_type, buffers, offsets[-1])
            out = awkwardlib.JaggedArray.fromoffsets(offsets, content)
            if mask is not None:
                mask = awkwardlib.numpy.frombuffer(mask, dtype=ARROW_BITMASKTYPE)
                return awkwardlib.BitMaskedArray(mask, out, maskedwhen=False, lsborder=True)
            else:
                return out

        elif isinstance(tpe, pyarrow.lib.UnionType) and tpe.mode == "sparse":
            assert getattr(tpe, "num_buffers", 3) == 3
            mask = buffers.pop(0)
            tags = awkwardlib.numpy.frombuffer(buffers.pop(0), dtype=ARROW_TAGTYPE)[:length]
            assert buffers.pop(0) is None
            index = awkwardlib.numpy.arange(len(tags), dtype=ARROW_INDEXTYPE)
            contents = []
            for i in range(tpe.num_children):
                try:
                    sublength = index[tags == i][-1] + 1
                except IndexError:
                    sublength = 0
                contents.append(popbuffers(None, tpe[i].type, buffers, sublength))
            for i in range(len(contents)):
                these = index[tags == i]
                if len(these) == 0:
                    contents[i] = contents[i][0:0]
                else:
                    contents[i] = contents[i][: these[-1] + 1]
            out = awkwardlib.UnionArray(tags, index, contents)
            if mask is not None:
                mask = awkwardlib.numpy.frombuffer(mask, dtype=ARROW_BITMASKTYPE)
                return awkwardlib.BitMaskedArray(mask, out, maskedwhen=False, lsborder=True)
            else:
                return out

        elif isinstance(tpe, pyarrow.lib.UnionType) and tpe.mode == "dense":
            assert getattr(tpe, "num_buffers", 3) == 3
            mask = buffers.pop(0)
            tags = awkwardlib.numpy.frombuffer(buffers.pop(0), dtype=ARROW_TAGTYPE)[:length]
            index = awkwardlib.numpy.frombuffer(buffers.pop(0), dtype=ARROW_INDEXTYPE)[:length]
            contents = []
            for i in range(tpe.num_children):
                try:
                    sublength = index[tags == i].max() + 1
                except ValueError:
                    sublength = 0
                contents.append(popbuffers(None, tpe[i].type, buffers, sublength))
            for i in range(len(contents)):
                these = index[tags == i]
                if len(these) == 0:
                    contents[i] = contents[i][0:0]
                else:
                    contents[i] = contents[i][: these.max() + 1]
            out = awkwardlib.UnionArray(tags, index, contents)
            if mask is not None:
                mask = awkwardlib.numpy.frombuffer(mask, dtype=ARROW_BITMASKTYPE)
                return awkwardlib.BitMaskedArray(mask, out, maskedwhen=False, lsborder=True)
            else:
                return out

        elif tpe == pyarrow.string():
            assert getattr(tpe, "num_buffers", 3) == 3
            mask = buffers.pop(0)
            offsets = awkwardlib.numpy.frombuffer(buffers.pop(0), dtype=ARROW_INDEXTYPE)[:length + 1]
            content = awkwardlib.numpy.frombuffer(buffers.pop(0), dtype=ARROW_CHARTYPE)[:offsets[-1]]
            out = awkwardlib.StringArray.fromoffsets(offsets, content[:offsets[-1]], encoding="utf-8")
            if mask is not None:
                mask = awkwardlib.numpy.frombuffer(mask, dtype=ARROW_BITMASKTYPE)
                return awkwardlib.BitMaskedArray(mask, out, maskedwhen=False, lsborder=True)
            else:
                return out

        elif tpe == pyarrow.large_string():
            assert getattr(tpe, "num_buffers", 3) == 3
            mask = buffers.pop(0)
            offsets = awkwardlib.numpy.frombuffer(buffers.pop(0), dtype=ARROW_LARGEINDEXTYPE)[:length + 1]
            content = awkwardlib.numpy.frombuffer(buffers.pop(0), dtype=ARROW_CHARTYPE)[:offsets[-1]]
            out = awkwardlib.StringArray.fromoffsets(offsets, content[:offsets[-1]], encoding="utf-8")
            if mask is not None:
                mask = awkwardlib.numpy.frombuffer(mask, dtype=ARROW_BITMASKTYPE)
                return awkwardlib.BitMaskedArray(mask, out, maskedwhen=False, lsborder=True)
            else:
                return out

        elif tpe == pyarrow.binary():
            assert getattr(tpe, "num_buffers", 3) == 3
            mask = buffers.pop(0)
            offsets = awkwardlib.numpy.frombuffer(buffers.pop(0), dtype=ARROW_INDEXTYPE)[:length + 1]
            content = awkwardlib.numpy.frombuffer(buffers.pop(0), dtype=ARROW_CHARTYPE)[:offsets[-1]]
            out = awkwardlib.StringArray.fromoffsets(offsets, content[:offsets[-1]], encoding=None)
            if mask is not None:
                mask = awkwardlib.numpy.frombuffer(mask, dtype=ARROW_BITMASKTYPE)
                return awkwardlib.BitMaskedArray(mask, out, maskedwhen=False, lsborder=True)
            else:
                return out

        elif tpe == pyarrow.large_binary():
            assert getattr(tpe, "num_buffers", 3) == 3
            mask = buffers.pop(0)
            offsets = awkwardlib.numpy.frombuffer(buffers.pop(0), dtype=ARROW_LARGEINDEXTYPE)[:length + 1]
            content = awkwardlib.numpy.frombuffer(buffers.pop(0), dtype=ARROW_CHARTYPE)[:offsets[-1]]
            out = awkwardlib.StringArray.fromoffsets(offsets, content[:offsets[-1]], encoding=None)
            if mask is not None:
                mask = awkwardlib.numpy.frombuffer(mask, dtype=ARROW_BITMASKTYPE)
                return awkwardlib.BitMaskedArray(mask, out, maskedwhen=False, lsborder=True)
            else:
                return out

        elif tpe == pyarrow.bool_():
            assert getattr(tpe, "num_buffers", 2) == 2
            mask = buffers.pop(0)
            out = awkwardlib.numpy.unpackbits(awkwardlib.numpy.frombuffer(buffers.pop(0), dtype=ARROW_CHARTYPE)).view(awkwardlib.MaskedArray.BOOLTYPE)
            out = out.reshape(-1, 8)[:,::-1].reshape(-1)[:length]    # lsborder=True
            if mask is not None:
                mask = awkwardlib.numpy.frombuffer(mask, dtype=ARROW_BITMASKTYPE)
                return awkwardlib.BitMaskedArray(mask, out, maskedwhen=False, lsborder=True)
            else:
                return out

        elif isinstance(tpe, pyarrow.lib.DataType):
            assert getattr(tpe, "num_buffers", 2) == 2
            mask = buffers.pop(0)
            out = awkwardlib.numpy.frombuffer(buffers.pop(0), dtype=tpe.to_pandas_dtype())[:length]
            if mask is not None:
                mask = awkwardlib.numpy.frombuffer(mask, dtype=ARROW_BITMASKTYPE)
                return awkwardlib.BitMaskedArray(mask, out, maskedwhen=False, lsborder=True)
            else:
                return out

        else:
            raise NotImplementedError(repr(tpe))

    if isinstance(obj, pyarrow.lib.Array):
        buffers = obj.buffers()
        out = popbuffers(obj, obj.type, buffers, len(obj))
        assert len(buffers) == 0
        return out

    elif isinstance(obj, pyarrow.lib.ChunkedArray):
        chunks = [x for x in obj.chunks if len(x) > 0]
        if len(chunks) == 1:
            return fromarrow(chunks[0])
        else:
            return awkwardlib.ChunkedArray([fromarrow(x) for x in chunks], chunksizes=[len(x) for x in chunks])

    elif isinstance(obj, pyarrow.lib.RecordBatch):
        out = awkwardlib.Table()
        for n, x in zip(obj.schema.names, obj.columns):
            out[n] = fromarrow(x)
        return out

    elif isinstance(obj, pyarrow.lib.Table):
        chunks = []
        chunksizes = []
        for batch in obj.to_batches():
            chunk = fromarrow(batch)
            if len(chunk) > 0:
                chunks.append(chunk)
                chunksizes.append(len(chunk))
        if len(chunks) == 1:
            return chunks[0]
        else:
            return awkwardlib.ChunkedArray(chunks, chunksizes=chunksizes)

    else:
        raise NotImplementedError(type(obj))

################################################################################ Parquet file handling

def toparquet(where, obj, **options):
    import pyarrow.parquet

    options["where"] = where

    def convert(obj, message):
        if isinstance(obj, (awkward0.array.base.AwkwardArray, numpy.ndarray)):
            out = toarrow(obj)
            if isinstance(out, pyarrow.Table):
                return out
            else:
                return pyarrow.Table.from_batches([pyarrow.RecordBatch.from_arrays([out], [""])])
        else:
            raise TypeError(message)

    if isinstance(obj, awkward0.array.chunked.ChunkedArray):
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

    elif isinstance(obj, (awkward0.array.base.AwkwardArray, numpy.ndarray)):
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
    def __init__(self, file, metadata=None, common_metadata=None):
        self.file = file
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
        return cls(state["file"], metadata=state["metadata"], common_metadata=state["common_metadata"])

def fromparquet(file, cache=None, persistvirtual=False, metadata=None, common_metadata=None):
    awkwardlib = awkward0
    parquetfile = _ParquetFile(file, metadata=metadata, common_metadata=common_metadata)
    columns = parquetfile.type.columns

    chunks = []
    chunksizes = []
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
            chunksizes.append(numrows)

    return awkwardlib.ChunkedArray(chunks, chunksizes)
