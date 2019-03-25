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

class NumbaMethods(object):
    @property
    def awkward(self):
        import awkward.numba
        return awkward.numba

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
