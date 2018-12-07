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

import awkward.util
import awkward.array.base
import awkward.array.jagged
import awkward.array.masked
import awkward.array.objects
import awkward.array.table
import awkward.array.union

def fillableof(obj):
    if obj is None:
        return None
    elif isinstance(obj, (bool, awkward.util.numpy.bool_, awkward.util.numpy.bool)):
        return BoolFillable()
    elif isinstance(obj, (numbers.Number, awkward.util.numpy.number)):
        return NumberFillable()
    elif isinstance(obj, bytes):
        return BytesFillable()
    elif isinstance(obj, awkward.util.string):
        return StringFillable()
    elif isinstance(obj, dict):
        return TableFillable(set(obj))
    elif isinstance(obj, tuple) and hasattr(obj, "_fields"):
        return TableFillable(set(self._fields))
    elif isinstance(obj, Iterable):
        return JaggedFillable()
    else:
        return TableFillable(set(n for n in obj.__dict__ if not n.startswith("_")))

class Fillable(object):
    def matches(self, fillable):
        return type(self) is type(fillable)

class UnknownFillable(Fillable):
    __slots__ = ["count"]

    def __init__(self):
        self.count = 0

    def __len__(self):
        return self.count

    def append(self, obj):
        if obj is None:
            self.count += 1
            return self
        else:
            fillable = fillableof(obj)
            if isinstance(fillable, (SimpleFillable, JaggedFillable)):
                if self.count == 0:
                    return fillable.append(obj)
                else:
                    return MaskedFillable(fillable.append(obj), self.count)

            else:
                return fillable.append(obj)

    def finalize(self):
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

    def append(self, obj):
        if obj is None:
            return MaskedFillable(self, 0).append(obj)

        fillable = fillableof(obj)
        if self.matches(fillable):
            self.data.append(obj)
            return self

        else:
            return UnionFillable(self).append(obj)

class BoolFillable(SimpleFillable):
    def finalize(self):
        return awkward.util.numpy.array(self.data, dtype=awkward.array.base.AwkwardArray.BOOLTYPE)

class NumberFillable(SimpleFillable):
    def finalize(self):
        return awkward.util.numpy.array(self.data)

class BytesFillable(SimpleFillable):
    def finalize(self):
        return awkward.array.objects.StringArray.fromiter(self.data, encoding=None)

class StringFillable(SimpleFillable):
    def finalize(self):
        return awkward.array.objects.StringArray.fromiter(self.data, encoding="utf-8")

class JaggedFillable(Fillable):
    __slots__ = ["content", "offsets"]

    def __init__(self):
        self.content = UnknownFillable()
        self.offsets = [0]

    def __len__(self):
        return len(self.offsets) - 1

    def append(self, obj):
        if obj is None:
            return MaskedFillable(self, 0).append(obj)

        fillable = fillableof(obj)
        if self.matches(fillable):
            for x in obj:
                self.content = self.content.append(x)
            self.offsets.append(len(self.content))
            return self

        else:
            return UnionFillable(self).append(obj)

    def finalize(self):
        return awkward.array.jagged.JaggedArray.fromoffsets(self.offsets, self.content.finalize())

class MaskedFillable(Fillable):
    __slots__ = ["content", "nullpos"]

    def matches(self, fillable):
        return fillable is None

    def __init__(self, content, count):
        self.content = content
        self.nullpos = list(range(count))

    def __len__(self):
        return len(self.content) + len(self.nullpos)

    def append(self, obj):
        if obj is None:
            self.nullpos.append(len(self))
        else:
            self.content = self.content.append(obj)
        return self

    def finalize(self):
        if isinstance(self.content, UnionFillable):
            index = awkward.util.numpy.zeros(len(self), dtype=awkward.array.masked.IndexedMaskedArray.INDEXTYPE)
            index[self.nullpos] = -1
            index[index == 0] = awkward.util.numpy.arange(len(self.content))

            return awkward.array.masked.IndexedMaskedArray(index, self.content.finalize())

        valid = awkward.util.numpy.ones(len(self), dtype=awkward.array.masked.MaskedArray.MASKTYPE)
        valid[self.nullpos] = False

        if isinstance(self.content, (BoolFillable, NumberFillable)):
            compact = self.content.finalize()
            expanded = awkward.util.numpy.empty(len(self), dtype=compact.dtype)
            expanded[valid] = compact

            return awkward.array.masked.MaskedArray(valid, expanded, maskedwhen=False)

        elif isinstance(self.content, (BytesFillable, StringFillable)):
            compact = self.content.finalize()
            counts = awkward.util.numpy.zeros(len(self), dtype=compact.counts.dtype)
            counts[valid] = compact.counts
            expanded = awkward.array.objects.StringArray.fromcounts(counts, compact.content)

            return awkward.array.masked.MaskedArray(valid, expanded, maskedwhen=False)

        elif isinstance(self.content, JaggedFillable):
            compact = self.content.finalize()
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

    def append(self, obj):
        fillable = fillableof(obj)
        if fillable is None:
            return MaskedFillable(self, 0).append(obj)

        else:
            for tag, content in enumerate(self.contents):
                if content.matches(fillable):
                    self.tags.append(tag)
                    self.index.append(len(content))
                    content.append(obj)
                    break

            else:
                self.tags.append(len(self.contents))
                self.index.append(len(fillable))
                self.contents.append(fillable.append(obj))

            return self

    def finalize(self):
        return awkward.array.union.UnionArray(self.tags, self.index, [x.finalize() for x in self.contents])
    
def fromiter(iterable):
    fillable = UnknownFillable()

    for obj in iterable:
        fillable = fillable.append(obj)

    return fillable.finalize()
