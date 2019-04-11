#!/usr/bin/env python

# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-array/blob/master/LICENSE

import re
from collections import OrderedDict

import numpy
import numba

import awkward.array.table
from .base import NumbaMethods
from .base import AwkwardArrayType
from .base import clsrepr
from .base import sliceval2
from .base import sliceval3

class TableNumba(NumbaMethods, awkward.array.table.Table):
    class Row(awkward.array.table.Table.Row):
        pass

        # def __init__(self, table, index):

        # def __repr__(self):

        # def __contains__(self, name):

        # def tolist(self):

        # def __len__(self):

        # def __iter__(self, checkiter=True):

        # def __getitem__(self, where):

        # def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):

        # def __eq__(self, other):

        # def __ne__(self, other):

        # @property
        # def i0(self):

        # @property
        # def i1(self):

        # @property
        # def i2(self):

        # @property
        # def i3(self):

        # @property
        # def i4(self):

        # @property
        # def i5(self):

        # @property
        # def i6(self):

        # @property
        # def i7(self):

        # @property
        # def i8(self):

        # @property
        # def i9(self):

    # def __init__(self, columns1={}, *columns2, **columns3):

    # def tolist(self):

    # @classmethod
    # def named(cls, rowname, columns1={}, *columns2, **columns3):

    # @property
    # def rowname(self):

    # @rowname.setter
    # def rowname(self, value):

    # @classmethod
    # def fromrec(cls, recarray):

    # @classmethod
    # def frompairs(cls, pairs):

    # @classmethod
    # def fromview(cls, view, base):

    # def copy(self, contents=None):

    # def deepcopy(self, contents=None):

    # def empty_like(self, **overrides):

    # def zeros_like(self, **overrides):

    # def ones_like(self, **overrides):

    # def __awkward_persist__(self, ident, fill, prefix, suffix, schemasuffix, storage, compression, **kwargs):

    # @property
    # def base(self):

    # @property
    # def contents(self):

    # @contents.setter
    # def contents(self, value):

    # def __len__(self):

    # def _gettype(self, seen):

    # def _length(self):

    # def _index(self):

    # def _newslice(self, head):

    # def _valid(self):

    # def __iter__(self, checkiter=True):

    # def __getitem__(self, where):

    # def __setitem__(self, where, what):

    # def __delitem__(self, where):

    # def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):

    # @property
    # def istuple(self):

    # def flattentuple(self):

    # def _hasjagged(self):

    # def _reduce(self, ufunc, identity, dtype, regularaxis):

    # @property
    # def columns(self):

    # def astype(self, dtype):

######################################################################## register types in Numba

@numba.extending.typeof_impl.register(awkward.array.table.Table)
def _Table_typeof(val, c):
    return TableType(val.rowname, OrderedDict((n, numba.typeof(x)) for n, x in val.contents.items()), special=type(val), specialrow=type(val).Row)

class TableType(AwkwardArrayType):
    class RowType(numba.types.Type):
        def __init__(self, rowname, types, special=awkward.array.table.Table.Row):
            super(RowType, self).__init__(name="RowType({0}, {{{1}}}{2})".format(repr(rowname), ", ".join("{0}: {1}".format(repr(n), t) for n, t in types.items()), "" if special is awkward.array.table.Table.Row else clsrepr(special)))
            self.rowname = rowname
            self.types = types
            self.special = special

    @property
    def rowtype(self):
        return TableType.RowType(self.rowname, self.types, special=self.specialrow)

    def __init__(self, rowname, types, special=awkward.array.table.Table, specialrow=awkward.array.table.Table.Row):
        super(TableType, self).__init__(name="TableType({0}, {{{1}}}{2}{3})".format(repr(rowname), ", ".join("{0}: {1}".format(repr(n), t) for n, t in types.items()), "" if special is awkward.array.table.Table else clsrepr(special), "" if specialrow is awkward.array.table.Table.Row else clsrepr(specialrow)))
        self.rowname = rowname
        self.types = types
        self.special = special
        self.specialrow = specialrow

    def getitem(self, wheretype):
        if isinstance(wheretype, numba.types.StringLiteral):
            if wheretype.literal_value in self.types:
                return self.types[wheretype.literal_value]
            else:
                raise TypeError("{0} is not a column of Table".format(repr(wheretype.literal_value)))

        # elif literal list/tuple of strings

        else:
            return TableType(self.rowname, OrderedDict((n, x.getitem(wheretype)) for n, x in self.types.items()), special=self.special, specialrow=self.specialrow)

######################################################################## model and boxing

# don't start non-fieldnames with "f"
def _safename(name):
    return "f" + _safename._pattern.sub(lambda bad: "_" + "".join("{0:02x}".format(ord(x)) for x in bad.group(0)) + "_", name)
_safename._pattern = re.compile("[^a-zA-Z0-9]+")

@numba.extending.register_model(TableType)
class TableModel(numba.datamodel.models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [(_safename(n), x) for n, x in fe_type.types.items()]
        super(TableModel, self).__init__(dmm, fe_type, members)

@numba.extending.unbox(TableType)
def _TableType_unbox(typ, obj, c):
    key_objs = [HERE]
    value_objs = [c.pyapi.object_getitem(obj, n) for n in key_objs]
