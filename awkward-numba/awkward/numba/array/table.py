#!/usr/bin/env python

# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/master/LICENSE

import awkward.array.table
from .base import NumbaMethods

class TableNumba(NumbaMethods, awkward.array.table.Table):
    pass
