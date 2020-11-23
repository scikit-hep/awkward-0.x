#!/usr/bin/env python

# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-0.x/blob/master/LICENSE

import importlib
from collections import OrderedDict

import awkward0.array.base
import awkward0.persist
import awkward0.type
import awkward0.util

class VirtualArray(awkward0.array.base.AwkwardArray):
    """
    VirtualArray
    """

    class TransientKey(object):
        def __init__(self, id):
            self._id = id
        def __repr__(self):
            return "<VirtualArray.TransientKey {0}>".format(repr(self._id))
        def __str__(self):
            return "_" + repr(self._id)
        def __hash__(self):
            return hash((VirtualArray.TransientKey, self._id))
        def __eq__(self, other):
            return isinstance(other, VirtualArray.TransientKey) and self._id == other._id
        def __ne__(self, other):
            return not self.__eq__(other)
        def __getstate__(self):
            raise RuntimeError("VirtualArray.TransientKeys are not unique across processes, and hence should not be serialized")

    def __init__(self, generator, args=(), kwargs={}, cache=None, persistentkey=None, type=None, nbytes=None, persistvirtual=True):
        self.generator = generator
        self.args = args
        self.kwargs = kwargs
        self.cache = cache
        self.persistentkey = persistentkey
        self.type = type
        self.nbytes = nbytes
        self.persistvirtual = persistvirtual
        self._array = None if cache is None else self.key
        self._setitem = None
        self._delitem = None

    def copy(self, generator=None, args=None, kwargs=None, cache=None, persistentkey=None, type=None, nbytes=None, persistvirtual=None):
        # FIXME: arguments through **kwargs because undef is different from None (None has meaning for some of them)
        out = self.__class__.__new__(self.__class__)
        out._generator = self._generator
        out._args = self._args
        out._kwargs = self._kwargs
        out._cache = self._cache
        out._persistentkey = self._persistentkey
        out._type = self._type
        out._nbytes = self._nbytes
        out._persistvirtual = self._persistvirtual
        out._array = self._array
        if self._setitem is None:
            out._setitem = None
        else:
            out._setitem = OrderedDict(self._setitem.items())
        if self._delitem is None:
            out._delitem = None
        else:
            out._delitem = list(self._delitem)
        if generator is not None:
            out.generator = generator
        if args is not None:
            out.args = args
        if kwargs is not None:
            out.kwargs = kwargs
        if cache is not None:
            out.cache = cache
        if persistentkey is not None:
            out.persistentkey = persistentkey
        if type is not None:
            out.type = type
        if nbytes is not None:
            out.nbytes = nbytes
        if persistvirtual is not None:
            out.persistvirtual = persistvirtual
        return out

    def deepcopy(self, generator=None, args=None, kwargs=None, cache=None, persistentkey=None, type=None, nbytes=None, persistvirtual=None):
        out = self.copy(generator=generator, args=args, kwargs=kwargs, cache=cache, persistentkey=persistentkey, type=type, nbytes=nbytes, persistvirtual=persistvirtual)
        out._array = self._util_deepcopy(out._array)
        if out._setitem is not None:
            for n in list(out._setitem):
                out._setitem[n] = self._util_deepcopy(out._setitem[n])
        return out

    def empty_like(self, **overrides):
        if isinstance(self.array, self.numpy.ndarray):
            return self.numpy.empty_like(self.array)
        else:
            return self.array.empty_like(**overrides)

    def zeros_like(self, **overrides):
        if isinstance(self.array, self.numpy.ndarray):
            return self.numpy.zeros_like(self.array)
        else:
            return self.array.zeros_like(**overrides)

    def ones_like(self, **overrides):
        if isinstance(self.array, self.numpy.ndarray):
            return self.numpy.ones_like(self.array)
        else:
            return self.array.ones_like(**overrides)

    def __awkward_serialize__(self, serializer):
        self._valid()

        if self._persistvirtual:
            out = serializer.encode_call(
                ["awkward0", "VirtualArray"],
                serializer(self._generator, "VirtualArray.generator"),
                {"tuple": [
                    serializer(x, "VirtualArray.args")
                    for x in self._args
                ]},
                {"dict": {
                    n: serializer(x, "VirtualArray.kwargs")
                    for n, x in self._kwargs.items()
                }}
            )
            out["cacheable"] = True
            others = {}
            if self._persistentkey is not None:
                others["persistentkey"] = serializer(self._persistentkey)

            if self._type is not None:
                others["type"] = serializer.encode_call(
                    ["awkward0.persist", "json2type"],
                    {"json": awkward0.persist.type2json(self._type)},
                )
                others["type"]["whitelistable"] = True
            if others:
                out["kwargs"] = others
        else:
            out = serializer(self.array, "VirtualArray.array")
            out.pop("id")

        return out

    @property
    def generator(self):
        return self._generator

    @generator.setter
    def generator(self, value):
        if self.check_prop_valid:
            if not callable(value):
                raise TypeError("generator must be a callable")
        self._generator = value


    @property
    def args(self):
        return self._args

    @args.setter
    def args(self, value):
        if not isinstance(value, tuple):
            value = (value,)
        self._args = value

    @property
    def kwargs(self):
        return self._kwargs

    @kwargs.setter
    def kwargs(self, value):
        if self.check_prop_valid:
            if not isinstance(value, dict):
                raise TypeError("kwargs must be a dict")
        self._kwargs = dict(value)

    @property
    def cache(self):
        return self._cache

    @cache.setter
    def cache(self, value):
        if self.check_prop_valid:
            if not value is None and not (callable(getattr(value, "__getitem__", None)) and callable(getattr(value, "__setitem__", None)) and callable(getattr(value, "__delitem__", None))):
                raise TypeError("cache must be None, a dict, or have __getitem__/__setitem__/__delitem__ methods")
        self._cache = value

    @property
    def persistentkey(self):
        return self._persistentkey

    @persistentkey.setter
    def persistentkey(self, value):
        if self.check_prop_valid:
            if value is not None and not isinstance(value, awkward0.util.string):
                raise TypeError("persistentkey must be None or a string")
        self._persistentkey = value

    @property
    def persistvirtual(self):
        return self._persistvirtual

    @persistvirtual.setter
    def persistvirtual(self, value):
        if self.check_prop_valid:
            if not isinstance(value, (bool, self.numpy.bool_, self.numpy.bool)):
                raise TypeError("persistvirtual must be boolean")
        self._persistvirtual = bool(value)

    def _gettype(self, seen):
        if self._type is None or self.ismaterialized:
            return awkward0.type._fromarray(self.array, seen)
        else:
            return self._type.to

    def _util_layout(self, position, seen, lookup):
        args = (awkward0.type.LayoutArg("generator", self._generator),
                awkward0.type.LayoutArg("args", self._args),
                awkward0.type.LayoutArg("kwargs", dict(self._kwargs)))
        if self.ismaterialized:
            awkward0.type.LayoutNode(self.array, position + (0,), seen, lookup)
            args = args + (awkward0.type.LayoutArg("array", position + (0,)),)
        return args

    @property
    def type(self):
        if self._type is None or self.ismaterialized:
            return awkward0.type.ArrayType(len(self.array), awkward0.type._resolve(awkward0.type._fromarray(self.array, {}), {}))
        else:
            return self._type

    @type.setter
    def type(self, value):
        if self.check_prop_valid:
            if value is not None and not isinstance(value, awkward0.type.ArrayType):
                raise TypeError("type must be None or an awkward0 type (to set Numpy parameters, use awkward0.util.fromnumpy(shape, dtype, masked=False))")
        self._type = value

    def _getnbytes(self, seen):
        if id(self) in seen:
            return 0
        else:
            seen.add(id(self))
            if self._nbytes is None or self.ismaterialized:
                array = self.array
                return (array.nbytes if isinstance(array, self.numpy.ndarray) else array._getnbytes(seen))
            else:
                return self._nbytes

    @property
    def nbytes(self):
        return self._getnbytes(set())

    @nbytes.setter
    def nbytes(self, value):
        if self.check_prop_valid:
            if value is not None:
                if not self._util_isinteger(value):
                    raise TypeError("nbytes must be an integer or None")
                if value < 0:
                    raise ValueError("nbytes must be a non-negative integer or None")
            self._nbytes = value

    def __len__(self):
        return self.shape[0]

    def _valid(self):
        if self.check_whole_valid:
            pass

    @property
    def key(self):
        if self._persistentkey is not None:
            return self._persistentkey
        else:
            return self.TransientKey(id(self))

    @property
    def array(self):
        # Normal states:
        #   (1) no cache and _array is None: make a new one
        #   (2) no cache and _array is an array: return _array
        #   (3) have a cache and _array is None: make a new one (filling cache)
        #   (4) have a cache and _array is a key and cache[key] was evicted: make a new one (filling cache)
        #   (5) have a cache and _array is a key and cache[key] exists: return cache[key]
        #
        # Abnormal states (user manually changed cache after materialization):
        #   (6) no cache and _array is a key (user removed _cache): make a new one
        #   (7) have a cache and _array is an array (user added _cache): fill cache and return _array

        if self._array is None:
            # states (1) and (3)
            return self.materialize()

        elif self._cache is None:
            if isinstance(self._array, (VirtualArray.TransientKey, awkward0.util.string)):
                # abnormal state (6)
                return self.materialize()
            else:
                # state (2)
                return self._array

        else:
            if isinstance(self._array, (VirtualArray.TransientKey, awkward0.util.string)):
                try:
                    # state (5)
                    return self._cache[self._array]
                except KeyError:
                    # state (4), taking any error in __getitem__ as evidence that it was evicted
                    return self.materialize()
            else:
                # abnormal state (7)
                self._cache[self.key] = self._array
                return self._array

    @property
    def ismaterialized(self):
        if self._cache is None:
            return isinstance(self._array, (self.numpy.ndarray, awkward0.array.base.AwkwardArray))
        else:
            return self._array is not None and self._array in self._cache

    def materialize(self):
        array = self._util_toarray(self._generator(*self._args, **self._kwargs), self.DEFAULTTYPE)
        if self._setitem is not None:
            for n, x in self._setitem.items():
                array[n] = x
        if self._delitem is not None:
            for n in self._delitem:
                del array[n]

        if self._type is not None:
            materializedtype = awkward0.type.fromarray(array)
            if ((isinstance(self._type, awkward0.type.Type) and not self._type._eq(materializedtype, set(), ignoremask=True)) or
                (not isinstance(self._type, awkward0.type.Type) and not self._type == materializedtype)):
                raise TypeError("materialized array has type\n\n{0}\n\nexpected type\n\n{1}".format(awkward0.type._str(awkward0.type.fromarray(array), indent="    "), awkward0.type._str(self._type, indent="    ")))

        if self._cache is None:
            # states (1), (2), and (6)
            self._array = array
        else:
            # states (3) and (4)
            self._array = self.key
            self._cache[self._array] = array

        return array

    def __del__(self):
        # TransientKeys are based on runtime ids, which Python may reuse after an object is garbage collected
        # they *MUST* be removed from the cache to avoid confusion; persistentkeys can (and should) stay in
        if getattr(self, "_cache", None) is not None and isinstance(self._array, VirtualArray.TransientKey):
            try:
                del self._cache[self._array]
            except:
                pass

    def __iter__(self, checkiter=True):
        if checkiter:
            self._checkiter()
        return iter(self.array)

    def __array__(self, *args, **kwargs):
        self._checktonumpy()
        return self.numpy.array(self.array, *args, **kwargs)

    def __getitem__(self, where):
        return self.array[where]

    def __setitem__(self, where, what):
        self.array[where] = what
        if self._type is not None:
            self._type = awkward0.type.fromarray(self.array)
        if self._setitem is None:
            self._setitem = OrderedDict()
        self._setitem[where] = what

    def __delitem__(self, where):
        del self.array[where]
        if self._type is not None:
            self._type = awkward0.type.fromarray(self.array)
        if self._setitem is not None and where in self._setitem:
            del self._setitem
        if self._delitem is None:
            self._delitem = []
        if where not in self._delitem:
            self._delitem.append(where)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if "out" in kwargs:
            raise NotImplementedError("in-place operations not supported")

        if method != "__call__":
            return NotImplemented

        inputs = list(inputs)
        for i in range(len(inputs)):
            if isinstance(inputs[i], VirtualArray):
                inputs[i]._valid()
                inputs[i] = inputs[i].array

        return getattr(ufunc, method)(*inputs, **kwargs)

    @property
    def counts(self):
        return self._util_counts(self.array)

    def boolmask(self, maskedwhen=True):
        return self._util_boolmask(self.array, maskedwhen)

    def choose(self, n):
        return self.array.choose(n)

    def argchoose(self, n):
        return self.array.argchoose(n)

    def distincts(self, nested=False):
        return self.array.distincts(nested=nested)

    def argdistincts(self, nested=False):
        return self.array.argdistincts(nested=nested)

    def pairs(self, nested=False):
        return self.array.pairs(nested=nested)

    def argpairs(self, nested=False):
        return self.array.argpairs(nested=nested)

    def cross(self, other, nested=False):
        return self.array.cross(other, nested=nested)

    def argcross(self, other, nested=False):
        return self.array.argcross(other, nested=nested)

    def flattentuple(self):
        return self._util_flattentuple(self.array)

    def flatten(self, axis=0):
        return self._util_flatten(self.array, axis)

    def pad(self, length, maskedwhen=True, clip=False, axis=0):
        return self._util_pad(self.array, length, maskedwhen, clip, axis)

    def regular(self):
        return self._util_regular(self.array)

    def _hasjagged(self):
        return self._util_hasjagged(self.array)

    def _reduce(self, ufunc, identity, dtype):
        return self._util_reduce(self.array, ufunc, identity, dtype)

    def _prepare(self, ufunc, identity, dtype):
        array = self.array
        if isinstance(array, self.numpy.ndarray):
            if dtype is None and issubclass(array.dtype.type, (self.numpy.bool_, self.numpy.bool)):
                dtype = self.numpy.dtype(type(identity))
            if dtype is None:
                return array
            else:
                return array.astype(dtype)
        else:
            return array._prepare(ufunc, identity, dtype)

    def argmin(self):
        return self.array.argmin()

    def argmax(self):
        return self.array.argmax()

    def _util_columns(self, seen):
        if id(self) in seen:
            return []
        seen.add(id(self))
        return self._util_columns_descend(self.array, seen)

    def _util_rowname(self, seen):
        if id(self) in seen:
            raise TypeError("not a Table, so there is no rowname")
        seen.add(id(self))
        return self._util_rowname_descend(self.array, seen)

    def astype(self, dtype):
        return self.array.astype(dtype)

    def fillna(self, value):
        return self._util_fillna(self.array, value)

    @classmethod
    def _concatenate_axis0(cls, arrays):
        assert all(isinstance(x, VirtualArray) for x in arrays)
        return awkward0.array.base.AwkwardArray.concatenate([x.array for x in arrays], axis=0)

    @classmethod
    def _concatenate_axis1(cls, arrays):
        assert all(isinstance(x, VirtualArray) for x in arrays)
        return awkward0.array.base.AwkwardArray.concatenate([x.array for x in arrays], axis=1)

    @staticmethod
    def _topandas_doit(virtualarray):
        return virtualarray.array._topandas()

    _topandas_name = "VirtualSeries"

    def _topandas(self, seen):
        import awkward0.pandas
        if id(self) in seen:
            return seen[id(self)]
        else:
            out = seen[id(self)] = self.VirtualArray(self._topandas_doit, (self,), cache=self.cache, persistentkey=self.persistentkey, type=self.type, nbytes=self.nbytes, persistvirtual=self.persistvirtual)
            out.__class__ = awkward0.pandas.mixin(type(self))
            return out
