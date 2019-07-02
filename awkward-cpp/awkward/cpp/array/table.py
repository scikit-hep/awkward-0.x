#!/usr/bin/env python

# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-array/blob/master/LICENSE

import awkward.array.base
import awkward.array.table
from .base import CppMethods
from ._table import Table


class TableCpp(CppMethods, Table, awkward.array.table.Table):
    pass