#!/usr/bin/env python

# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-array/blob/master/LICENSE

import awkward.array.union
from .base import NumbaMethods

class UnionArrayNumba(NumbaMethods, awkward.array.union.UnionArray):
    pass
