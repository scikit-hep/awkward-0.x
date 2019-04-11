#!/usr/bin/env python

# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-array/blob/master/LICENSE

import awkward.array.masked
from .base import NumbaMethods

class MaskedArrayNumba(NumbaMethods, awkward.array.masked.MaskedArray):
    pass

class BitMaskedArrayNumba(NumbaMethods, awkward.array.masked.BitMaskedArray):
    pass

class IndexedMaskedArrayNumba(NumbaMethods, awkward.array.masked.IndexedMaskedArray):
    pass
