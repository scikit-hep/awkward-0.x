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

import numpy

import awkward.util

class AwkwardArray(awkward.util.NDArrayOperatorsMixin):
    CHARTYPE = numpy.dtype(numpy.uint8)
    INDEXTYPE = numpy.dtype(numpy.int64)
    MASKTYPE = numpy.dtype(numpy.bool_)
    BITMASKTYPE = numpy.dtype(numpy.uint8)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __str__(self):
        if len(self) <= 6:
            return "[{0}]".format(" ".join(str(x) for x in self))
        else:
            return "[{0} ... {1}]".format(" ".join(str(x) for x in self[:3]), ", ".join(str(x) for x in self[-3:]))

    def __repr__(self):
        return "<{0} {1} at {2:012x}>".format(self.__class__.__name__, str(self), id(self))

    def tolist(self):
        import awkward.array.table
        out = []
        for x in self:
            if isinstance(x, awkward.array.table.Table.Row):
                out.append(dict((n, x[n]) for n in x._table._content))
            elif isinstance(x, numpy.ma.core.MaskedConstant):
                out.append(None)
            else:
                try:
                    out.append(x.tolist())
                except AttributeError:
                    out.append(x)
        return out

    @staticmethod
    def _toarray(value, defaultdtype, passthrough):
        if isinstance(value, passthrough):
            return value
        else:
            try:
                return numpy.frombuffer(value, dtype=getattr(value, "dtype", defaultdtype)).reshape(getattr(value, "shape", -1))
            except AttributeError:
                return numpy.array(value, copy=False)

    @staticmethod
    def _isstring(where):
        if isinstance(where, awkward.util.string):
            return True
        try:
            assert all(isinstance(x, awkward.util.string) for x in where)
        except (TypeError, AssertionError):
            return False
        else:
            return True
