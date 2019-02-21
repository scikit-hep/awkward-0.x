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

import operator

import numpy
import numba

class NumbaMethods(object):
    @property
    def ChunkedArray(self):
        import awkward.numba.array.chunked
        return awkward.numba.array.chunked.ChunkedArrayNumba

    @property
    def AppendableArray(self):
        import awkward.numba.array.chunked
        return awkward.numba.array.chunked.AppendableArrayNumba

    @property
    def IndexedArray(self):
        import awkward.numba.array.indexed
        return awkward.numba.array.indexed.IndexedArrayNumba

    @property
    def SparseArray(self):
        import awkward.numba.array.indexed
        return awkward.numba.array.indexed.SparseArrayNumba

    @property
    def JaggedArray(self):
        import awkward.numba.array.jagged
        return awkward.numba.array.jagged.JaggedArrayNumba

    @property
    def MaskedArray(self):
        import awkward.numba.array.masked
        return awkward.numba.array.masked.MaskedArrayNumba

    @property
    def BitMaskedArray(self):
        import awkward.numba.array.masked
        return awkward.numba.array.masked.BitMaskedArrayNumba

    @property
    def IndexedMaskedArray(self):
        import awkward.numba.array.masked
        return awkward.numba.array.masked.IndexedMaskedArrayNumba

    @property
    def Methods(self):
        import awkward.numba.array.objects
        return awkward.numba.array.objects.MethodsNumba

    @property
    def ObjectArray(self):
        import awkward.numba.array.objects
        return awkward.numba.array.objects.ObjectArrayNumba

    @property
    def StringArray(self):
        import awkward.numba.array.objects
        return awkward.numba.array.objects.StringArrayNumba

    @property
    def Table(self):
        import awkward.numba.array.table
        return awkward.numba.array.table.TableNumba

    @property
    def UnionArray(self):
        import awkward.numba.array.union
        return awkward.numba.array.union.UnionArrayNumba

    @property
    def VirtualArray(self):
        import awkward.numba.array.virtual
        return awkward.numba.array.virtual.VirtualArrayNumba

class AwkwardArrayType(numba.types.Type):
    pass

@numba.typing.templates.infer_global(len)
class _AwkwardArrayType_type_len(numba.typing.templates.AbstractTemplate):
    def generic(self, args, kwargs):
        if len(args) == 1 and len(kwargs) == 0:
            array, = args
            if isinstance(array, AwkwardArrayType):
                return numba.typing.templates.signature(numba.types.intp, array)

@numba.typing.templates.infer_global(operator.getitem)
class _AwkwardArrayType_type_getitem(numba.typing.templates.AbstractTemplate):
    def generic(self, args, kwargs):
        if len(args) == 2 and len(kwargs) == 0:
            array, where = args
            if isinstance(array, AwkwardArrayType):
                if not isinstance(where, numba.types.BaseTuple):
                    where = numba.types.Tuple((where,))
                if len(where.types) == 0:
                    return array

                if any(isinstace(x, numba.types.Array) and x.ndim == 1 for x in where):
                    where = numba.types.Tuple(tuple(numba.types.Array(x, 1, "C") if isinstance(x, numba.types.Integer) else x for x in where))

                return numba.typing.templates.signature(arraytype.getitem(where))

def clsrepr(cls):
    import awkward.array
    if any(cls is x for x in (awkward.array.base.AwkwardArray,
                              awkward.array.chunked.ChunkedArray,
                              awkward.array.chunked.AppendableArray,
                              awkward.array.indexed.IndexedArray,
                              awkward.array.indexed.SparseArray,
                              awkward.array.jagged.JaggedArray,
                              awkward.array.masked.MaskedArray,
                              awkward.array.masked.BitMaskedArray,
                              awkward.array.masked.IndexedMaskedArray,
                              awkward.array.objects.Methods,
                              awkward.array.objects.ObjectArray,
                              awkward.array.objects.StringArray,
                              awkward.array.table.Table,
                              awkward.array.union.UnionArray,
                              awkward.array.virtual.VirtualArray)):
        return cls.__name__
    else:
        bases = ", ".join(clsrepr(x) for x in cls.__bases__ if x is not object)
        if len(bases) != 0:
            bases = "<" + bases + ">"
        return "{0}.{1}{2}".format(x.__module__, x.__name__, bases)

def sliceval2(context, builder, start, stop):
    out = context.make_helper(builder, numba.types.slice2_type)
    out.start = start
    out.stop = stop
    out.step = context.get_constant(numba.types.intp, 1)
    return out._getvalue()

def sliceval3(context, builder, start, stop, step):
    out = context.make_helper(builder, numba.types.slice3_type)
    out.start = start
    out.stop = stop
    out.step = step
    return out._getvalue()

ISADVANCED = numba.types.Array(numba.types.int64, 1, "C")
NOTADVANCED = numba.types.none
