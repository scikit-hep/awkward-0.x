#!/usr/bin/env python

# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-array/blob/master/LICENSE

class CppMethods(object):
    @property
    def awkward(self):
        import awkward.cpp
        return awkward.cpp

    @property
    def JaggedArray(self):
        import awkward.cpp.array.jagged
        return awkward.cpp.array.jagged.JaggedArrayCpp
