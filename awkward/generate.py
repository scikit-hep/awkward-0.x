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
try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

import awkward.array.base
import awkward.array.indexed
import awkward.array.jagged
import awkward.array.masked
import awkward.array.objects
import awkward.array.table
import awkward.array.union
import awkward.type
import awkward.util

def typeof(obj):
    if obj is None:
        return None

    elif isinstance(obj, (bool, awkward.util.numpy.bool_, awkward.util.numpy.bool)):
        return BoolFillable
    elif isinstance(obj, (numbers.Number, awkward.util.numpy.number)):
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
    @staticmethod
    def make(tpe):
        if tpe is None:
            return MaskedFillable(UnknownFillable(), 0)

        elif isinstance(tpe, type):
            return tpe()

        elif isinstance(tpe, set):
            return TableFillable(tpe)

        elif isinstance(tpe, tuple) and len(tpe) == 2 and isinstance(tpe[0], set):
            if len(tpe[0]) == 0:
                return ObjectFillable(JaggedFillable(), tpe[1])
            else:
                return ObjectFillable(TableFillable(tpe[0]), tpe[1])

        elif isinstance(tpe, tuple) and len(tpe) == 2 and isinstance(tpe[0], tuple):
            if len(tpe[0]) == 0:
                return NamedTupleFillable(JaggedFillable(), tpe[1])
            else:
                return NamedTupleFillable(TableFillable(tpe[0]), tpe[1])

        else:
            raise AssertionError(tpe)

    def matches(self, tpe):
        return type(self) is tpe

class UnknownFillable(Fillable):
    __slots__ = ["count"]

    def __init__(self):
        self.count = 0

    def __len__(self):
        return self.count

    def clear(self):
        self.count = 0

    def append(self, obj, tpe):
        if tpe is None:
            self.count += 1
            return self

        else:
            fillable = Fillable.make(tpe)
            if self.count == 0:
                return fillable.append(obj, tpe)
            else:
                return MaskedFillable(fillable.append(obj, tpe), self.count)

    def finalize(self, **options):
        if self.count == 0:
            return awkward.util.numpy.empty(0, dtype=awkward.array.base.AwkwardArray.DEFAULTTYPE)
        else:
            mask = awkward.util.numpy.zeros(self.count, dtype=awkward.array.masked.MaskedArray.MASKTYPE)
            return awkward.array.masked.MaskedArray(mask, mask, maskedwhen=False)

class SimpleFillable(Fillable):
    __slots__ = ["data"]

    def __init__(self):
        self.data = []

    def __len__(self):
        return len(self.data)

    def clear(self):
        self.data = []

    def append(self, obj, tpe):
        if tpe is None:
            return MaskedFillable(self, 0).append(obj, tpe)

        if self.matches(tpe):
            self.data.append(obj)
            return self

        else:
            return UnionFillable(self).append(obj, tpe)

class BoolFillable(SimpleFillable):
    def finalize(self, **options):
        return awkward.util.numpy.array(self.data, dtype=awkward.array.base.AwkwardArray.BOOLTYPE)

class NumberFillable(SimpleFillable):
    def finalize(self, **options):
        return awkward.util.numpy.array(self.data)

class BytesFillable(SimpleFillable):
    def finalize(self, **options):
        dictencoding = options.get("dictencoding", False)
        if (callable(dictencoding) and dictencoding(self.data)) or (not callable(dictencoding) and dictencoding):
            dictionary, index = awkward.util.numpy.unique(self.data, return_inverse=True)
            return awkward.array.indexed.IndexedArray(index, awkward.array.objects.StringArray.fromiter(dictionary, encoding=None))
        else:
            return awkward.array.objects.StringArray.fromiter(self.data, encoding=None)

class StringFillable(SimpleFillable):
    def finalize(self, **options):
        dictencoding = options.get("dictencoding", False)
        if (callable(dictencoding) and dictencoding(self.data)) or (not callable(dictencoding) and dictencoding):
            dictionary, index = awkward.util.numpy.unique(self.data, return_inverse=True)
            return awkward.array.indexed.IndexedArray(index, awkward.array.objects.StringArray.fromiter(dictionary, encoding="utf-8"))
        else:
            return awkward.array.objects.StringArray.fromiter(self.data, encoding="utf-8")

class JaggedFillable(Fillable):
    __slots__ = ["content", "offsets"]

    def __init__(self):
        self.content = UnknownFillable()
        self.offsets = [0]

    def __len__(self):
        return len(self.offsets) - 1

    def clear(self):
        self.content.clear()
        self.offsets = [0]

    def append(self, obj, tpe):
        if tpe is None:
            return MaskedFillable(self, 0).append(obj, tpe)

        if self.matches(tpe):
            for x in obj:
                self.content = self.content.append(x, typeof(x))
            self.offsets.append(len(self.content))
            return self

        else:
            return UnionFillable(self).append(obj, tpe)

    def finalize(self, **options):
        return awkward.array.jagged.JaggedArray.fromoffsets(self.offsets, self.content.finalize(**options))

class TableFillable(Fillable):
    __slots__ = ["fields", "contents", "count"]

    def __init__(self, fields):
        assert len(fields) > 0
        self.fields = fields
        self.contents = {n: UnknownFillable() for n in fields}
        self.count = 0

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
            return MaskedFillable(self, 0).append(obj, tpe)

        if self.matches(tpe):
            for n in self.fields:
                x = obj[n]
                self.contents[n] = self.contents[n].append(x, typeof(x))
            self.count += 1
            return self

        else:
            return UnionFillable(self).append(obj, tpe)

    def finalize(self, **options):
        return awkward.array.table.Table.frompairs((n, self.contents[n].finalize(**options)) for n in sorted(self.fields))

class ObjectFillable(Fillable):
    __slots__ = ["content", "cls"]

    def __init__(self, content, cls):
        self.content = content
        self.cls = cls

    def __len__(self):
        return len(self.content)

    def clear(self):
        self.content.clear()

    def matches(self, tpe):
        return isinstance(tpe, tuple) and len(tpe) == 2 and tpe[1] is self.cls and (len(tpe[0]) == 0 or self.content.matches(tpe[0]))

    def append(self, obj, tpe):
        if tpe is None:
            return MaskedFillable(self, 0).append(obj, tpe)

        if self.matches(tpe):
            if len(tpe[0]) == 0:
                self.content.append([], JaggedFillable)
            else:
                self.content.append(obj.__dict__, tpe[0])
            return self

        else:
            return UnionFillable(self).append(obj, tpe)

    def finalize(self, **options):
        def make(x):
            out = self.cls.__new__(self.cls)
            out.__dict__.update(x.tolist())
            return out

        return awkward.array.objects.ObjectArray(self.content.finalize(**options), make)

class NamedTupleFillable(ObjectFillable):
    def append(self, obj, tpe):
        if tpe is None:
            return MaskedFillable(self, 0).append(obj, tpe)

        if self.matches(tpe):
            if len(tpe[0]) == 0:
                self.content.append([], JaggedFillable)
            else:
                self.content.append({n: x for n, x in zip(obj._fields, obj)}, tpe[0])
            return self

        else:
            return UnionFillable(self).append(obj, tpe)

    def finalize(self, **options):
        def make(x):
            asdict = x.tolist()
            return self.cls(*[asdict[n] for n in self.cls._fields])

        return awkward.array.objects.ObjectArray(self.content.finalize(**options), make)

class MaskedFillable(Fillable):
    __slots__ = ["content", "nullpos"]

    def __init__(self, content, count):
        self.content = content
        self.nullpos = list(range(count))

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
            index = awkward.util.numpy.zeros(len(self), dtype=awkward.array.masked.IndexedMaskedArray.INDEXTYPE)
            index[self.nullpos] = -1
            index[index == 0] = awkward.util.numpy.arange(len(self.content))

            return awkward.array.masked.IndexedMaskedArray(index, self.content.finalize(**options))

        valid = awkward.util.numpy.ones(len(self), dtype=awkward.array.masked.MaskedArray.MASKTYPE)
        valid[self.nullpos] = False

        if isinstance(self.content, (BoolFillable, NumberFillable)):
            compact = self.content.finalize(**options)
            expanded = awkward.util.numpy.empty(len(self), dtype=compact.dtype)
            expanded[valid] = compact

            return awkward.array.masked.MaskedArray(valid, expanded, maskedwhen=False)

        elif isinstance(self.content, (BytesFillable, StringFillable)):
            compact = self.content.finalize(**options)

            if isinstance(compact, awkward.array.indexed.IndexedArray):
                index = awkward.util.numpy.zeros(len(self), dtype=compact.index.dtype)
                index[valid] = compact.index
                expanded = awkward.array.indexed.IndexedArray(index, compact.content)
            else:
                counts = awkward.util.numpy.zeros(len(self), dtype=compact.counts.dtype)
                counts[valid] = compact.counts
                expanded = awkward.array.objects.StringArray.fromcounts(counts, compact.content, encoding=compact.encoding)

            return awkward.array.masked.MaskedArray(valid, expanded, maskedwhen=False)

        elif isinstance(self.content, JaggedFillable):
            compact = self.content.finalize(**options)
            counts = awkward.util.numpy.zeros(len(self), dtype=compact.counts.dtype)
            counts[valid] = compact.counts
            expanded = awkward.array.jagged.JaggedArray.fromcounts(counts, compact.content)

            return awkward.array.masked.MaskedArray(valid, expanded, maskedwhen=False)

        else:
            raise AssertionError(self.content)

class UnionFillable(Fillable):
    __slots__ = ["contents", "tags", "index"]

    def __init__(self, content):
        self.contents = [content]
        self.tags = [0] * len(content)
        self.index = list(range(len(content)))

    def __len__(self):
        return len(self.tags)

    def clear(self):
        for content in self.contents:
            content.clear()
        self.tags = []
        self.index = []

    def append(self, obj, tpe):
        if tpe is None:
            return MaskedFillable(self, 0).append(obj, tpe)

        else:
            for tag, content in enumerate(self.contents):
                if content.matches(tpe):
                    self.tags.append(tag)
                    self.index.append(len(content))
                    content.append(obj, tpe)
                    break

            else:
                fillable = Fillable.make(tpe)
                self.tags.append(len(self.contents))
                self.index.append(len(fillable))
                self.contents.append(fillable.append(obj, tpe))

            return self

    def finalize(self, **options):
        return awkward.array.union.UnionArray(self.tags, self.index, [x.finalize(**options) for x in self.contents])

def _checkoptions(options):
    unrecognized = set(options).difference(["dictencoding"])
    if len(unrecognized) != 0:
        raise TypeError("unrecognized options: {0}".format(", ".join(sorted(unrecognized))))

def fromiter(iterable, **options):
    _checkoptions(options)

    fillable = UnknownFillable()

    for obj in iterable:
        fillable = fillable.append(obj, typeof(obj))

    return fillable.finalize(**options)

def fromiterchunks(iterable, chunksize, **options):
    if not isinstance(chunksize, (numbers.Integral, awkward.util.numpy.integer)) or chunksize <= 0:
        raise TypeError("chunksize must be a positive integer")

    _checkoptions(options)

    fillable = UnknownFillable()
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
