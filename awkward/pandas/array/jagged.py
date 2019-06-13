#!/usr/bin/env python

# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-array/blob/master/LICENSE

import numpy

import awkward.array.jagged
from pandas.api.extensions import ExtensionArray
from awkward.pandas.accessor import AwkwardType

class JaggedArrayPandas(awkward.array.jagged.JaggedArray, ExtensionArray):

    @property
    def dtype(self):
        return AwkwardType()

    def __array__(self, dtype=None):
        return numpy.array(self.tolist(), dtype='object')
