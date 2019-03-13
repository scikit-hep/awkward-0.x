try: import pandas as pd
except ImportError:
    raise ImportError("Awkward-pandas requires pandas >= 0.23")

import types
import numbers

import numpy
from pandas.api.extensions import ExtensionDtype

from .accessor import DelegatedMethod, DelegatedProperty, delegated_method

import awkward.persist
import awkward.type

import awkward.array as awkarr


class AwkwardType(ExtensionDtype):
    name = 'awkward'
    type = awkarr.base.AwkwardArray
    kind = 'O'

    @classmethod
    def construct_from_string(cls, string):
        if string == cls.name:
            return cls()
        else:
            raise TypeError("Cannot construct a '{}' from"
                            "'{}'".format(cls, string))

# -----------------------------------------------------------------------------
# Pandas accessors
# -----------------------------------------------------------------------------

@pd.api.extensions.register_series_accessor("awkward")
class AwkwardAccessor:

    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self.content = pandas_obj.values
        #self._index = obj.index
        self.name = pandas_obj.name

    @staticmethod
    def _validate(obj):
        if not is_awkward_type(obj):
            raise AttributeError("Cannot use 'awkward' accessor on objects of "
                                 "dtype '{}'.".formate(obj.dtype))
    def content(self, value):
        return delegated_method(self.content, value)

    #def tolist(self, value):
    #    return delegated_method(self.tolist(), value)

def is_awkward_type(obj):
    t = getattr(obj, 'dtype', obj)
    try:
        return isinstance(t, AwkwardType) or issubclass(t, AwkwardType)
    except Exception:
        return False
