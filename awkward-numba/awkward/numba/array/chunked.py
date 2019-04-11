#!/usr/bin/env python

# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-array/blob/master/LICENSE

import awkward.array.chunked
from .base import NumbaMethods

class ChunkedArrayNumba(NumbaMethods, awkward.array.chunked.ChunkedArray):
    pass

class AppendableArrayNumba(NumbaMethods, awkward.array.chunked.AppendableArray):
    pass
