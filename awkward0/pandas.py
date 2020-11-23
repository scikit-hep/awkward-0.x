 #!/usr/bin/env python

# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-0.x/blob/master/LICENSE

from __future__ import absolute_import
import pandas
import pandas.api.extensions

import awkward0.array

def delegated_method(method, index, name, *args, **kwargs):
    return pandas.Series

class Delegated(object):
    def __init__(self, name):
        self.name = name

    def __get__(self, obj, type=None):
        index = object.__getattribute__(obj, "_index")
        name = object.__getattribute__(obj, "_name")
        result = self._get_result(obj)
        return pandas.Series(result, index, name=name)

class DelegatedProperty(Delegated):
    def _get_result(self, obj, type=None):
        return getattr(object.__getattribute__(obj, "_data"), self.name)

class DelegatedMethod(Delegated):
    def __get__(self, obj, type=None):
        index = object.__getattribute__(obj, "_index")
        name = object.__getattribute__(bj, "_name")
        method = getattr(object.__getattribute__(obj, "_data"), self.name)
        return delegated_method(method, index, name)

class AwkwardSeriesDtype(pandas.api.extensions.ExtensionDtype):
    name = "awkward0"
    type = awkward0.array.base.AwkwardArray
    kind = "O"

    @classmethod
    def construct_from_string(cls, string):
        if string == cls.name:
            return cls()
        else:
            raise TypeError("Cannot construct a '{}' from '{}'".format(cls, string))

@pandas.api.extensions.register_series_accessor("awkward0")
class AwkwardAccessor(object):
    def __init__(self, pandas_obj):
        self._validate(pandas_obj)

        self._data = pandas_obj.values
        self._index = pandas_obj.index
        self._name = pandas_obj.name

    @staticmethod
    def _validate(obj):
        if not is_awkward_type(obj):
            raise AttributeError("Cannot use 'awkward0' accessor on objects of dtype '{}'.".format(obj.dtype))

def is_awkward_type(obj):
    t = getattr(obj, "dtype", obj)
    try:
        return isinstance(t, AwkwardSeriesDtype) or issubclass(t, AwkwardSeriesDtype)
    except Exception:
        return False

class AwkwardSeries(object):
    @property
    def dtype(self):
        return AwkwardSeriesDtype()

    @staticmethod
    def _findclass(cls):
        for base in cls.__bases__:
            if issubclass(base, awkward0.array.base.AwkwardArray):
                if not issubclass(base, AwkwardSeries):
                    return base
                else:
                    out = AwkwardSeries._findclass(base)
                    if out is not None:
                        return out
        return None

    def __array__(self, dtype=None):
        cls = self._findclass(type(self))

        if dtype is None:
            dtype = cls.dtype.fget(self)

        return cls.__array__(self, dtype=dtype)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        out = self._findclass(type(self)).__array_ufunc__(self, ufunc, method, *inputs, **kwargs)
        return out.pandas

    def isna(self):
        return self.numpy.zeros(self.shape, dtype=self.BOOLTYPE)

    @classmethod
    def _concat_same_type(cls, to_concat):
        return cls.concatenate(to_concat)

    @property
    def ChunkedArray(self):
        return mixin(self._findclass(type(self)).ChunkedArray.fget(self))

    @property
    def AppendableArray(self):
        return mixin(self._findclass(type(self)).AppendableArray.fget(self))

    @property
    def IndexedArray(self):
        return mixin(self._findclass(type(self)).IndexedArray.fget(self))

    @property
    def SparseArray(self):
        return mixin(self._findclass(type(self)).SparseArray.fget(self))

    @property
    def JaggedArray(self):
        return mixin(self._findclass(type(self)).JaggedArray.fget(self))

    @property
    def MaskedArray(self):
        return mixin(self._findclass(type(self)).MaskedArray.fget(self))

    @property
    def BitMaskedArray(self):
        return mixin(self._findclass(type(self)).BitMaskedArray.fget(self))

    @property
    def IndexedMaskedArray(self):
        return mixin(self._findclass(type(self)).IndexedMaskedArray.fget(self))

    @property
    def ObjectArray(self):
        return mixin(self._findclass(type(self)).ObjectArray.fget(self))

    @property
    def StringArray(self):
        return mixin(self._findclass(type(self)).StringArray.fget(self))

    @property
    def Table(self):
        return mixin(self._findclass(type(self)).Table.fget(self))

    @property
    def UnionArray(self):
        return mixin(self._findclass(type(self)).UnionArray.fget(self))

    @property
    def VirtualArray(self):
        return mixin(self._findclass(type(self)).VirtualArray.fget(self))

def mixin(tpe):
    return type(tpe._topandas_name, (AwkwardSeries, tpe, pandas.api.extensions.ExtensionArray), {})
