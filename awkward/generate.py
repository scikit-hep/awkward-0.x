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
        return MaskedFillable
    elif isinstance(obj, (bool, awkward.util.numpy.bool_, awkward.util.numpy.bool)):
        return BoolFillable
    elif isinstance(obj, (numbers.Number, awkward.util.numpy.number)):
        return NumberFillable
    elif isinstance(obj, bytes):
        return BytesFillable
    elif isinstance(obj, awkward.util.string):
        return StringFillable
    elif isinstance(obj, dict):
        return tuple(obj)
    elif isinstance(obj, tuple) and hasattr(obj, "_fields"):
        return self._fields
    elif isinstance(obj, Iterable):
        return JaggedFillable
    else:
        return tuple(n for n in obj.__dict__ if not n.startswith("_"))

class Fillable(object): pass

class UnknownFillable(Fillable):
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
            if issubclass(fillable, SimpleFillable):
                if self.count == 0:
                    return fillable().append(obj)
                else:
                    return MaskedFillable.fromcount(fillable().append(obj), self.count)

            else:
                return fillable(UnknownFillable()).append(obj)

class SimpleFillable(Fillable):
    def __init__(self):
        self.data = []

    def __len__(self):
        return len(self.data)

    def append(self, obj):
        fillable = fillableof(obj)
        if fillable is self.__class__:
            self.data.append(obj)
            return self
        else:
            return fillable(self).append(obj)

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

class MaskedFillable(Fillable):
    def __init__(self, content):
        self.content = content
        self.nullpos = []

    @classmethod
    def fromcount(cls, content, count):
        self = MaskedFillable(content)
        self.nullpos = list(range(count))
        return self

    def __len__(self):
        return len(self.content) + len(self.nullpos)

    def append(self, obj):
        if obj is None:
            self.nullpos.append(len(self))
        else:
            self.content = self.content.append(obj)
        return self

    def finalize(self):
        if isinstance(self.content, (BoolFillable, NumberFillable)):
            valid = awkward.util.numpy.ones(len(self), dtype=bool)
            valid[self.nullpos] = False

            compact = self.content.finalize()
            expanded = awkward.util.numpy.empty(len(self), dtype=compact.dtype)
            expanded[valid] = compact

            return awkward.array.masked.MaskedArray(valid, expanded, maskedwhen=False)
            
        else:
            raise NotImplementedError

def fromiter(iterable):
    fillable = UnknownFillable()

    for obj in iterable:
        fillable = fillable.append(obj)

    return fillable.finalize()
