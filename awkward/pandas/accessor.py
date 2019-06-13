import pandas as pd

from pandas.api.extensions import ExtensionDtype

def delegated_method(method, index, name, *args, **kwargs):
    return pd.Series

import awkward.array

class Delegated(object):
    # Descriptor for delegating attribute access to/from
    # Series to underlying array

    def __init__(self, name):
        self.name = name

    def __get__(self, obj, type=None):
        index = object.__getattribute__(obj, '_index')
        name = object.__getattribute__(obj, '_name')
        result = self._get_result(obj)
        return pd.Series(result, index, name=name)


class DelegatedProperty(Delegated):
    def _get_result(self, obj, type=None):
        return getattr(object.__getattribute__(obj, '_data'), self.name)

class DelegatedMethod(Delegated):
    def __get__(self, obj, type=None):
        index = object.__getattribute__(obj, '_index')
        name = object.__getattribute__(bj, '_name')
        method = getattr(object.__getattribute__(obj, '_data'), self.name)
        return delegated_method(method, index, name)


class AwkwardType(ExtensionDtype):
    name = 'awkward'
    type = awkward.array.base.AwkwardArray
    kind = 'O'

    @classmethod
    def construct_from_string(cls, string):
        if string == cls.name:
            return cls()
        else:
            raise TypeError("Cannot construct a '{}' from"
                            "'{}'".format(cls, string))


# ----------------------------------------------------------------------------- 
# 
# Pandas accessors
# -----------------------------------------------------------------------------

@pd.api.extensions.register_series_accessor("awkward")
class AwkwardAccessor:

    def __init__(self, pandas_obj):
        self._validate(pandas_obj)

        self._data = pandas_obj.values
        self._index = pandas_obj.index
        self._name = pandas_obj.name

    @staticmethod
    def _validate(obj):
        if not is_awkward_type(obj):
            raise AttributeError("Cannot use 'awkward' accessor on objects of "
                                 "dtype '{}'.".formate(obj.dtype))
def is_awkward_type(obj):
    t = getattr(obj, 'dtype', obj)
    try:
        return isinstance(t, AwkwardType) or issubclass(t, AwkwardType)
    except Exception:
        return False
