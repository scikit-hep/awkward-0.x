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

import codecs
import collections
import numbers
try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

import numpy

import awkward.type
import awkward.util

def typeof(obj):
    if obj is None:
        return None

    elif isinstance(obj, (bool, numpy.bool_, numpy.bool)):
        return BoolFillable
    elif isinstance(obj, (numbers.Number, awkward.numpy.number)):
        return NumberFillable
    elif isinstance(obj, bytes):
        return BytesFillable
    elif isinstance(obj, awkward.util.string):
        return StringFillable

    elif isinstance(obj, dict):
        if any(not isinstance(x, str) for x in obj):
            raise TypeError("only dicts with str-typed keys may be converted")
        if len(obj) == 0:
            return None
        else:
            return set(obj)

    elif isinstance(obj, tuple) and hasattr(obj, "_fields") and obj._fields is type(obj)._fields:
        return obj._fields, type(obj)

    elif isinstance(obj, Iterable):
        return JaggedFillable

    else:
        return set(n for n in obj.__dict__ if not n.startswith("_")), type(obj)

class Fillable(object):
    def __init__(self, awkwardlib):
        self.awkwardlib = awkwardlib

    @staticmethod
    def make(tpe, awkwardlib):
        if tpe is None:
            return MaskedFillable(UnknownFillable(awkwardlib), 0, awkwardlib)

        elif isinstance(tpe, type):
            return tpe(awkwardlib)

        elif isinstance(tpe, set):
            return TableFillable(tpe, awkwardlib)

        elif isinstance(tpe, tuple) and len(tpe) == 2 and isinstance(tpe[0], set):
            if len(tpe[0]) == 0:
                return ObjectFillable(JaggedFillable(awkwardlib), tpe[1], awkwardlib)
            else:
                return ObjectFillable(TableFillable(tpe[0], awkwardlib), tpe[1], awkwardlib)

        elif isinstance(tpe, tuple) and len(tpe) == 2 and isinstance(tpe[0], tuple):
            if len(tpe[0]) == 0:
                return NamedTupleFillable(JaggedFillable(awkwardlib), tpe[1], awkwardlib)
            else:
                return NamedTupleFillable(TableFillable(tpe[0], awkwardlib), tpe[1], awkwardlib)

        else:
            raise AssertionError(tpe)

    def matches(self, tpe):
        return type(self) is tpe

class UnknownFillable(Fillable):
    __slots__ = ["count", "awkwardlib"]

    def __init__(self, awkwardlib):
        self.count = 0
        self.awkwardlib = awkwardlib

    def __len__(self):
        return self.count

    def clear(self):
        self.count = 0

    def append(self, obj, tpe):
        if tpe is None:
            self.count += 1
            return self

        else:
            fillable = Fillable.make(tpe, self.awkwardlib)
            if self.count == 0:
                return fillable.append(obj, tpe)
            else:
                return MaskedFillable(fillable.append(obj, tpe), self.count, self.awkwardlib)

    def finalize(self, **options):
        if self.count == 0:
            return self.awkwardlib.numpy.empty(0, dtype=self.awkwardlib.JaggedArray.DEFAULTTYPE)
        else:
            mask = self.awkwardlib.numpy.zeros(self.count, dtype=self.awkwardlib.MaskedArray.MASKTYPE)
            return self.awkwardlib.MaskedArray(mask, mask, maskedwhen=False)

class SimpleFillable(Fillable):
    __slots__ = ["data", "awkwardlib"]

    def __init__(self, awkwardlib):
        self.data = []
        self.awkwardlib = awkwardlib

    def __len__(self):
        return len(self.data)

    def clear(self):
        self.data = []

    def append(self, obj, tpe):
        if tpe is None:
            return MaskedFillable(self, 0, self.awkwardlib).append(obj, tpe)

        if self.matches(tpe):
            self.data.append(obj)
            return self

        else:
            return UnionFillable(self, self.awkwardlib).append(obj, tpe)

class BoolFillable(SimpleFillable):
    def finalize(self, **options):
        return self.awkwardlib.numpy.array(self.data, dtype=self.awkwardlib.JaggedArray.BOOLTYPE)

class NumberFillable(SimpleFillable):
    def finalize(self, **options):
        return self.awkwardlib.numpy.array(self.data)

class BytesFillable(SimpleFillable):
    def finalize(self, **options):
        dictencoding = options.get("dictencoding", False)
        if (callable(dictencoding) and dictencoding(self.data)) or (not callable(dictencoding) and dictencoding):
            dictionary, index = self.awkwardlib.numpy.unique(self.data, return_inverse=True)
            return self.awkwardlib.IndexedArray(index, self.awkwardlib.StringArray.fromiter(dictionary, encoding=None))
        else:
            return self.awkwardlib.StringArray.fromiter(self.data, encoding=None)

class StringFillable(SimpleFillable):
    def finalize(self, **options):
        dictencoding = options.get("dictencoding", False)
        if (callable(dictencoding) and dictencoding(self.data)) or (not callable(dictencoding) and dictencoding):
            dictionary, index = self.awkwardlib.numpy.unique(self.data, return_inverse=True)
            return self.awkwardlib.IndexedArray(index, self.awkwardlib.StringArray.fromiter(dictionary, encoding="utf-8"))
        else:
            return self.awkwardlib.StringArray.fromiter(self.data, encoding="utf-8")

class JaggedFillable(Fillable):
    __slots__ = ["content", "offsets", "awkwardlib"]

    def __init__(self, awkwardlib):
        self.content = UnknownFillable(awkwardlib)
        self.offsets = [0]
        self.awkwardlib = awkwardlib

    def __len__(self):
        return len(self.offsets) - 1

    def clear(self):
        self.content.clear()
        self.offsets = [0]

    def append(self, obj, tpe):
        if tpe is None:
            return MaskedFillable(self, 0, self.awkwardlib).append(obj, tpe)

        if self.matches(tpe):
            for x in obj:
                self.content = self.content.append(x, typeof(x))
            self.offsets.append(len(self.content))
            return self

        else:
            return UnionFillable(self, self.awkwardlib).append(obj, tpe)

    def finalize(self, **options):
        return self.awkwardlib.JaggedArray.fromoffsets(self.offsets, self.content.finalize(**options))

class TableFillable(Fillable):
    __slots__ = ["fields", "contents", "count", "awkwardlib"]

    def __init__(self, fields, awkwardlib):
        assert len(fields) > 0
        self.fields = fields
        self.contents = {n: UnknownFillable(awkwardlib) for n in fields}
        self.count = 0
        self.awkwardlib = awkwardlib

    def __len__(self):
        return self.count

    def clear(self):
        for content in self.contents.values():
            content.clear()
        self.count = 0

    def matches(self, tpe):
        return self.fields == tpe

    def append(self, obj, tpe):
        if tpe is None:
            return MaskedFillable(self, 0, self.awkwardlib).append(obj, tpe)

        if self.matches(tpe):
            for n in self.fields:
                x = obj[n]
                self.contents[n] = self.contents[n].append(x, typeof(x))
            self.count += 1
            return self

        else:
            return UnionFillable(self, self.awkwardlib).append(obj, tpe)

    def finalize(self, **options):
        return self.awkwardlib.Table.frompairs((n, self.contents[n].finalize(**options)) for n in sorted(self.fields))

class ObjectFillable(Fillable):
    __slots__ = ["content", "cls", "awkwardlib"]

    def __init__(self, content, cls, awkwardlib):
        self.content = content
        self.cls = cls
        self.awkwardlib = awkwardlib

    def __len__(self):
        return len(self.content)

    def clear(self):
        self.content.clear()

    def matches(self, tpe):
        return isinstance(tpe, tuple) and len(tpe) == 2 and tpe[1] is self.cls and (len(tpe[0]) == 0 or self.content.matches(tpe[0]))

    def append(self, obj, tpe):
        if tpe is None:
            return MaskedFillable(self, 0, self.awkwardlib).append(obj, tpe)

        if self.matches(tpe):
            if len(tpe[0]) == 0:
                self.content.append([], JaggedFillable)
            else:
                self.content.append(obj.__dict__, tpe[0])
            return self

        else:
            return UnionFillable(self, self.awkwardlib).append(obj, tpe)

    def finalize(self, **options):
        def make(x):
            out = self.cls.__new__(self.cls)
            out.__dict__.update(x.tolist())
            return out

        return self.awkwardlib.ObjectArray(self.content.finalize(**options), make)

class NamedTupleFillable(ObjectFillable):
    def append(self, obj, tpe):
        if tpe is None:
            return MaskedFillable(self, 0, self.awkwardlib).append(obj, tpe)

        if self.matches(tpe):
            if len(tpe[0]) == 0:
                self.content.append([], JaggedFillable)
            else:
                self.content.append({n: x for n, x in zip(obj._fields, obj)}, tpe[0])
            return self

        else:
            return UnionFillable(self, self.awkwardlib).append(obj, tpe)

    def finalize(self, **options):
        def make(x):
            asdict = x.tolist()
            return self.cls(*[asdict[n] for n in self.cls._fields])

        return self.awkwardlib.ObjectArray(self.content.finalize(**options), make)

class MaskedFillable(Fillable):
    __slots__ = ["content", "nullpos", "awkwardlib"]

    def __init__(self, content, count, awkwardlib):
        self.content = content
        self.nullpos = list(range(count))
        self.awkwardlib = awkwardlib

    def matches(self, tpe):
        return tpe is None

    def __len__(self):
        return len(self.content) + len(self.nullpos)

    def clear(self):
        self.content.clear()
        self.nullpos = []

    def append(self, obj, tpe):
        if tpe is None:
            self.nullpos.append(len(self))
        else:
            self.content = self.content.append(obj, tpe)
        return self

    def finalize(self, **options):
        if isinstance(self.content, (TableFillable, ObjectFillable, UnionFillable)):
            index = self.awkwardlib.numpy.zeros(len(self), dtype=self.awkwardlib.IndexedMaskedArray.INDEXTYPE)
            index[self.nullpos] = -1
            index[index == 0] = awkward.numpy.arange(len(self.content))

            return self.awkwardlib.IndexedMaskedArray(index, self.content.finalize(**options))

        valid = self.awkwardlib.numpy.ones(len(self), dtype=self.awkwardlib.MaskedArray.MASKTYPE)
        valid[self.nullpos] = False

        if isinstance(self.content, (BoolFillable, NumberFillable)):
            compact = self.content.finalize(**options)
            expanded = self.awkwardlib.numpy.empty(len(self), dtype=compact.dtype)
            expanded[valid] = compact

            return self.awkwardlib.MaskedArray(valid, expanded, maskedwhen=False)

        elif isinstance(self.content, (BytesFillable, StringFillable)):
            compact = self.content.finalize(**options)

            if isinstance(compact, self.awkwardlib.IndexedArray):
                index = self.awkwardlib.numpy.zeros(len(self), dtype=compact.index.dtype)
                index[valid] = compact.index
                expanded = self.awkwardlib.IndexedArray(index, compact.content)
            else:
                counts = self.awkwardlib.numpy.zeros(len(self), dtype=compact.counts.dtype)
                counts[valid] = compact.counts
                expanded = self.awkwardlib.StringArray.fromcounts(counts, compact.content, encoding=compact.encoding)

            return self.awkwardlib.MaskedArray(valid, expanded, maskedwhen=False)

        elif isinstance(self.content, JaggedFillable):
            compact = self.content.finalize(**options)
            counts = self.awkwardlib.numpy.zeros(len(self), dtype=compact.counts.dtype)
            counts[valid] = compact.counts
            expanded = self.awkwardlib.JaggedArray.fromcounts(counts, compact.content)

            return self.awkwardlib.MaskedArray(valid, expanded, maskedwhen=False)

        else:
            raise AssertionError(self.content)

class UnionFillable(Fillable):
    __slots__ = ["contents", "tags", "index", "awkwardlib"]

    def __init__(self, content, awkwardlib):
        self.contents = [content]
        self.tags = [0] * len(content)
        self.index = list(range(len(content)))
        self.awkwardlib = awkwardlib

    def __len__(self):
        return len(self.tags)

    def clear(self):
        for content in self.contents:
            content.clear()
        self.tags = []
        self.index = []

    def append(self, obj, tpe):
        if tpe is None:
            return MaskedFillable(self, 0, self.awkwardlib).append(obj, tpe)

        else:
            for tag, content in enumerate(self.contents):
                if content.matches(tpe):
                    self.tags.append(tag)
                    self.index.append(len(content))
                    content.append(obj, tpe)
                    break

            else:
                fillable = Fillable.make(tpe, self.awkwardlib)
                self.tags.append(len(self.contents))
                self.index.append(len(fillable))
                self.contents.append(fillable.append(obj, tpe))

            return self

    def finalize(self, **options):
        return self.awkwardlib.UnionArray(self.tags, self.index, [x.finalize(**options) for x in self.contents])

def _checkoptions(options):
    unrecognized = set(options).difference(["dictencoding"])
    if len(unrecognized) != 0:
        raise TypeError("unrecognized options: {0}".format(", ".join(sorted(unrecognized))))

def fromiter(iterable, awkwardlib=None, **options):
    _checkoptions(options)

    awkwardlib = awkward.util.awkwardlib(awkwardlib)
    fillable = UnknownFillable(awkwardlib)

    for obj in iterable:
        fillable = fillable.append(obj, typeof(obj))

    return fillable.finalize(**options)

def fromiterchunks(iterable, chunksize, awkwardlib=None, **options):
    if not isinstance(chunksize, (numbers.Integral, numpy.integer)) or chunksize <= 0:
        raise TypeError("chunksize must be a positive integer")

    _checkoptions(options)

    awkwardlib = awkward.util.awkwardlib(awkwardlib)
    fillable = UnknownFillable(awkwardlib)
    count = 0
    tpe = None

    for obj in iterable:
        fillable = fillable.append(obj, typeof(obj))
        count += 1

        if count == chunksize:
            out = fillable.finalize(**options)
            outtpe = awkward.type.fromarray(out).to
            if tpe is None:
                tpe = outtpe
            elif tpe != outtpe:
                raise TypeError("data type has changed after the first chunk (first chunk is not large enough to see the full generality of the data):\n\n{0}\n\nversus\n\n{1}".format(awkward.type._str(tpe, indent="    "), awkward.type._str(outtpe, indent="    ")))
            yield out

            fillable.clear()
            count = 0

    if count != 0:
        out = fillable.finalize(**options)
        outtpe = awkward.type.fromarray(out).to
        if tpe is None:
            tpe = outtpe
        elif tpe != outtpe:
            raise TypeError("data type has changed after the first chunk (first chunk is not large enough to see the full generality of the data):\n\n{0}\n\nversus\n\n{1}".format(awkward.type._str(tpe, indent="    "), awkward.type._str(outtpe, indent="    ")))
        yield out
