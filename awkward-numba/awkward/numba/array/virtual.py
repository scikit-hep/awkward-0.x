#!/usr/bin/env python

# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-array/blob/master/LICENSE

import awkward.array.virtual
from .base import NumbaMethods

class VirtualArrayNumba(NumbaMethods, awkward.array.virtual.VirtualArray):
    pass
