#!/usr/bin/env python

# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-array/blob/master/LICENSE

######################################################################## Numba-accelerated interface

import awkward.array.base
import awkward.array.table
import sys
from .base import CppMethods
from .table import Table

class TableCpp(CppMethods, Table, awkward.array.table.Table):
    