#!/usr/bin/env python

# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/master/LICENSE

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
