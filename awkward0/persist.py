#!/usr/bin/env python

# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-0.x/blob/master/LICENSE

import base64
import fnmatch
import importlib
import json
import numbers
import os
import pickle
import types
import zipfile
import zlib
from itertools import count
try:
    from collections.abc import Mapping, MutableMapping
except ImportError:
    from collections import Mapping, MutableMapping

import numpy

import awkward0.type
import awkward0.version

compression = [
        {"minsize": 8192, "types": [numpy.bool_, numpy.bool, numpy.integer], "contexts": "*", "pair": (zlib.compress, ("zlib", "decompress"))},
    ]

whitelist = [
        ["numpy", "frombuffer"],
        ["zlib", "decompress"],
        ["lzma", "decompress"],
        ["backports.lzma", "decompress"],
        ["lz4.block", "decompress"],
        ["awkward0", "*Array"],
        ["awkward0", "Table"],
        ["awkward0", "numpy", "frombuffer"],
        ["awkward0.util", "frombuffer"],
        ["awkward0.persist"],
        ["awkward0.arrow", "_ParquetFile", "fromjson"],
        ["awkward", "*Array"],
        ["awkward", "Table"],
        ["awkward", "numpy", "frombuffer"],
        ["awkward.util", "frombuffer"],
        ["awkward.persist"],
        ["awkward.arrow", "_ParquetFile", "fromjson"],
        ["uproot3_methods.classes.*"],
        ["uproot3_methods.profiles.*"],
        ["uproot_methods.classes.*"],
        ["uproot_methods.profiles.*"],
        ["uproot.tree", "_LazyFiles"],
        ["uproot.tree", "_LazyTree"],
        ["uproot.tree", "_LazyBranch"],
    ]

def frompython(obj):
    return base64.b64encode(pickle.dumps(obj)).decode("ascii")

def topython(string):
    return pickle.loads(base64.b64decode(string.encode("ascii")))

def spec2function(obj, whitelist=whitelist):
    awkwardlib = "awkward0"
    for white in whitelist:
        for n, p in zip(obj, white):
            if not fnmatch.fnmatchcase(n, p):
                break
        else:
            if obj[0] == "awkward0":
                obj = [awkwardlib] + obj[1:]
            elif obj[0] == "awkward":
                obj = [awkwardlib] + obj[1:]
            if obj[0].startswith("uproot_methods"):
                obj = ["uproot3_methods" + obj[0][14:]] + obj[1:]
            gen, genname = importlib.import_module(obj[0]), obj[1:]
            if not isinstance(gen, types.ModuleType):
                raise TypeError("first item of a function description must be a module")
            if genname[:1] == ["numpy"]:
                gen, genname = getattr(gen, genname[0]), genname[1:]
            while len(genname) > 0:
                gen, genname = getattr(gen, genname[0]), genname[1:]
                if isinstance(gen, types.ModuleType):
                    raise TypeError("non-first items of a function description must not be a module")
            break
    else:
        raise RuntimeError("callable not in whitelist; add it by passing a whitelist argument:\n\n    whitelist = awkward0.persist.whitelist + [{0}]".format(repr(obj)))
    return gen

def dtype2json(obj):
    if obj.subdtype is not None:
        dt, sh = obj.subdtype
        return (dtype2json(dt), sh)
    elif obj.names is not None:
        return [(n, dtype2json(obj[n])) for n in obj.names]
    else:
        return str(obj)

def json2dtype(obj):
    def recurse(obj):
        if isinstance(obj, (list, tuple)) and len(obj) > 0 and (isinstance(obj[-1], numbers.Integral) or isinstance(obj[0], str) or (isinstance(obj[-1], (list, tuple)) and all(isinstance(x, numbers.Integral) for x in obj[-1]))):
            return tuple(recurse(x) for x in obj)
        elif isinstance(obj, (list, tuple)):
            return [recurse(x) for x in obj]
        else:
            return obj
    return numpy.dtype(recurse(obj))

def type2json(obj):
    if isinstance(obj, awkward0.type.Type):
        labeled = obj._labeled()
    else:
        labeled = []

    seen = set()

    def takes(n):
        if n == float("inf"):
            return "inf"
        else:
            return int(n)

    def recurse(obj):
        if isinstance(obj, awkward0.type.Type):
            if id(obj) in seen:
                for i, x in enumerate(labeled):
                    if obj is x:
                        return {"ref": "T{0}".format(i)}

            else:
                seen.add(id(obj))
                if isinstance(obj, awkward0.type.ArrayType):
                    out = {"takes": takes(obj._takes), "to": recurse(obj._to)}

                elif isinstance(obj, awkward0.type.TableType):
                    out = {"fields": [[n, recurse(x)] for n, x in obj._fields.items()]}

                elif isinstance(obj, awkward0.type.UnionType):
                    out = {"possibilities": [recurse(x) for x in obj._possibilities]}

                elif isinstance(obj, awkward0.type.OptionType):
                    out = {"type": recurse(obj._type)}

                for i, x in enumerate(labeled):
                    if obj is x:
                        return {"set": "T{0}".format(i), "as": out}
                else:
                    return out

        elif isinstance(obj, numpy.dtype):
            return {"dtype": dtype2json(obj)}

        elif callable(obj):
            if obj.__module__ == "__main__":
                raise TypeError("cannot persist object type: its generator is defined in __main__, which won't be available in a subsequent session")
            if hasattr(obj, "__qualname__"):
                spec = [obj.__module__] + obj.__qualname__.split(".")
            else:
                spec = [obj.__module__, obj.__name__]

            gen, genname = importlib.import_module(spec[0]), spec[1:]
            while len(genname) > 0:
                gen, genname = getattr(gen, genname[0]), genname[1:]
            if gen is not obj:
                raise TypeError("cannot persist object type: its generator cannot be found via its __name__ (Python 2) or __qualname__ (Python 3)")

            return {"function": spec}

        else:
            raise TypeError("only awkward0.type.Type, numpy.dtype, and callables are types")

    return recurse(obj)

def json2type(obj, whitelist=whitelist):
    labels = {}

    def takes(n):
        if n == "inf":
            return float("inf")
        else:
            return n

    def recurse(obj):
        if not isinstance(obj, dict):
            raise TypeError("json2type is expecting a JSON object, found: {0}".format(repr(obj)))

        if "set" in obj:
            placeholder = labels[obj["set"]] = awkward0.type.Placeholder()
            placeholder.value = recurse(obj["as"])
            return placeholder

        elif "ref" in obj:
            return labels[obj["ref"]]

        elif "takes" in obj and "to" in obj:
            return awkward0.type.ArrayType(takes(obj["takes"]), recurse(obj["to"]))

        elif "fields" in obj:
            out = awkward0.type.TableType()
            for n, x in obj["fields"]:
                out[n] = recurse(x)
            return out

        elif "possibilities" in obj:
            return awkward0.type.UnionType(*[recurse(x) for x in obj["possibilities"]])

        elif "type" in obj:
            return awkward0.type.OptionType(recurse(obj["type"]))

        elif "dtype" in obj:
            return json2dtype(obj["dtype"])

        elif "function" in obj:
            return spec2function(obj["function"], whitelist=whitelist)

        else:
            raise ValueError("unexpected set of keys in JSON: {0}".format(", ".join(repr(x) for x in obj)))

    return awkward0.type._resolve(recurse(obj), {})

def jsonable(obj):
    if obj is None:
        return obj

    elif isinstance(obj, dict) and all(isinstance(n, str) for n in obj):
        return {n: jsonable(x) for n, x in obj.items()}

    elif isinstance(obj, list):
        return [jsonable(x) for x in obj]

    elif isinstance(obj, str):
        return str(obj)

    elif isinstance(obj, (bool, numpy.bool_, numpy.bool)):
        return bool(obj)      # policy: eliminate Numpy types

    elif isinstance(obj, (numbers.Integral, numpy.integer)):
        return int(obj)       # policy: eliminate Numpy types

    elif isinstance(obj, (numbers.Real, numpy.floating)) and numpy.isfinite(obj):
        return float(obj)     # policy: eliminate Numpy types

    else:
        raise TypeError("object cannot be losslessly serialized as JSON")

class ObjRef(object):
    def __init__(self, idgen=None):
        if idgen:
            self.idgen = iter(idgen)
        self._i2r = {}
        self._r2o = {}

    def nextid(self):
        return next(self.idgen)

    def __contains__(self, obj):
        return id(obj) in self._i2r

    def __setitem__(self, obj, ref):
        self._i2r[id(obj)] = ref
        self._r2o[ref] = obj

    def __getitem__(self, obj):
        if obj not in self:
            self[obj] = self.nextid()
        return self._i2r[id(obj)]

    def __delitem__(self, obj):
        assert obj in self
        del self._r2o[self._i2r[id(obj)]]
        del self._i2r[id(obj)]

    def get(self, obj, default=None):
        return self[obj] if obj in self else default

    def obj(self, ref):
        if ref in self._r2o:
            return self._r2o[ref]
        else:
            return awkward0.array.virtual.VirtualArray(lambda: self._r2o[ref])

class Serializer(object):
    def __init__(self, storage, prefix="", suffix="", schemasuffix=""):
        self.storage = storage
        self.suffix = suffix
        self.prefix = prefix
        self.schemasuffix = schemasuffix
        self.seen = ObjRef(idgen=count())

    def store(self, name, obj):
        schema = {"awkward0": awkward0.version.__version__, "schema": self(obj)}
        if self.prefix != "":
            schema["prefix"] = self.prefix

        schema = self._finalize_schema(schema) or schema

        self.storage[name + self.schemasuffix] = self._encode_schema(schema)
        return schema

    def load(self, *args, **kwargs):
        return deserialize(*args, storage=self.storage, seen=self.seen, **kwargs)

    def encode_call(self, *args, **kwargs):
        func, args = args[0], args[1:]
        out = {"call": self._obj2spec(func) if callable(func) else tuple(func)}
        if args:
            out["args"] = list(args)
        if kwargs:
            out["kwargs"] = kwargs
        return out

    def encode_json(self, obj):
        return {"json": jsonable(obj)}

    def encode_python(self, obj):
        return {"python": frompython(obj)}

    @classmethod
    def _encode_primitive(cls, obj):
        if isinstance(obj, numpy.dtype):
            return {"dtype": dtype2json(obj)}

    @classmethod
    def _obj2spec(cls, obj, test=True):
        if hasattr(obj, "__qualname__"):
            spec = [obj.__module__] + obj.__qualname__.split(".")
        else:
            spec = [obj.__module__, obj.__name__]

        if test:
            val = importlib.import_module(spec[0])
            for key in spec[1:]:
                val = getattr(val, key)
            assert val == obj

        return spec

    def _encode_complex(self, obj, context):
        if callable(getattr(obj, "__awkward_serialize__", None)):
            return obj.__awkward_serialize__(self)

        if hasattr(obj, "tojson") and hasattr(type(obj), "fromjson"):
            try:
                return self.encode_call(self._obj2spec(type(obj).fromjson), self.encode_json(obj.tojson()))
            except:
                pass

        if isinstance(obj, numpy.ndarray):
            return self._encode_numpy(obj, context)

        if hasattr(obj, "__module__") and (hasattr(obj, "__qualname__") or hasattr(obj, "__name__")) and obj.__module__ != "__main__":
            try:
                return {"function": self._obj2spec(obj)}
            except:
                pass

        try:
            return self.encode_json(obj)
        except TypeError:
            pass

        try:
            return self.encode_python(obj)
        except:
            pass

    def _encode_numpy(self, obj, context):
        key = str(self.seen[obj]) + self.suffix
        self.storage[self.prefix + key] = obj
        return {"read": key}

    def _encode_schema(self, schema):
        return json.dumps(schema).encode("ascii")

    def _finalize_schema(self, schema):
        pass

    def __call__(self, obj, context=""):
        out = self._encode_primitive(obj)

        if out is not None:
            return out

        if obj in self.seen:
            return {"ref": self.seen[obj]}
        else:
            ident = self.seen[obj]

        out = self._encode_complex(obj, context)
        if out is None:
            raise TypeError("failed to encode {0} (type: {1})".format(repr(obj), type(obj)))

        if "id" in out:
            if out["id"] is False:
                del self.seen[obj]
            elif out["id"] != self.seen[obj]:
                raise RuntimeError("unexpected id change")
        else:
            out["id"] = ident

        return out

    def fill(self, obj, context, prefix, suffix, schemasuffix, storage, compression, **kwargs):
        assert self.prefix == prefix
        assert self.suffix == suffix
        assert self.schemasuffix == schemasuffix
        assert self.storage == storage
        assert self.compression == compression
        return self(obj, context=context)

class BlobSerializer(Serializer):
    class CompressPolicy(object):
        enc2dec = {
            zlib.compress: ("zlib", "decompress"),
        }

        @classmethod
        def parse(cls, x):
            if isinstance(x, cls):
                return x
            elif isinstance(x, dict):
                return cls(**x)
            elif callable(x):
                return cls(enc=x)
            elif len(x) == 2 and callable(x[0]):
                return cls(enc=x[0], dec=x[1])
            else:
                raise TypeError("can't parse compression policy {0}".format(x))

        def __init__(self, pair=None, enc=None, dec=None, minsize=0, types=object, contexts="*"):
            if pair is not None:
                enc, dec = pair
            if dec is None:
                dec = self.enc2dec[enc]
            if isinstance(types, list):
                types = tuple(types)
            elif not isinstance(types, tuple):
                types = types,
            if isinstance(contexts, list):
                contexts = tuple(contexts)
            elif not isinstance(contexts, tuple):
                contexts = contexts,
            assert callable(enc)
            assert isinstance(dec, tuple)
            assert 0 <= minsize
            self.enc = enc
            self.dec = dec
            self.minsize = minsize
            self.types = types
            self.contexts = contexts

        @property
        def pair(self):
            return (self.enc, self.dec)

        def test(self, obj, context):
            return (obj.nbytes >= self.minsize and issubclass(obj.dtype.type, tuple(self.types)) and any(fnmatch.fnmatchcase(context, p) for p in self.contexts))

    @classmethod
    def _parse_compression(cls, comp):
        if comp is None or comp is False:
            comp = []
        elif comp is True:
            comp = [{"minsize": 0, "types": object, "contexts": "*", "pair": (zlib.compress, ("zlib", "decompress"))}]
        elif not isinstance(comp, (list, tuple)):
            comp = [comp]

        return list(map(cls.CompressPolicy.parse, comp))

    def __init__(self, *args, **kwargs):
        self.compression = self._parse_compression(kwargs.pop("compression", compression))
        super(BlobSerializer, self).__init__(*args, **kwargs)

    def _put_raw(self, data, ref=None):
        if ref is None:
            ref = data
        key = str(self.seen[ref]) + self.suffix
        self.storage[self.prefix + key] = data
        return dict(read=key)

    def _encode_numpy(self, obj, context):
        if obj.ndim > 1:
            dtype = numpy.dtype((obj.dtype, obj.shape[1:]))
        else:
            dtype = obj.dtype

        buf = None
        for policy in self._parse_compression(self.compression):
            if policy.test(obj, context):
                buf = self.encode_call(policy.dec, self._put_raw(policy.enc(obj.ravel()), ref=obj))
                break
        else:
            buf = self._put_raw(obj.ravel(), ref=obj)

        return self.encode_call(["awkward0", "numpy", "frombuffer"], buf, self(dtype), self(obj.shape[0]))

def serialize(obj, storage, name="", delimiter="-", **kwargs):
    if delimiter is None:
        delimiter = ""
    if name:
        kwargs.setdefault("prefix", name + delimiter)
    return BlobSerializer(storage, **kwargs).store(name, obj)

def deserialize(storage, name="", whitelist=whitelist, cache=None, seen=None):
    import awkward0.array.virtual

    schema = storage[name]
    if isinstance(schema, numpy.ndarray):
        schema = schema.tostring()
    if isinstance(schema, bytes):
        schema = schema.decode("ascii")
    schema = json.loads(schema)

    if "awkward" not in schema and "awkward0" not in schema:
        raise ValueError("JSON object is not an Awkward Array schema (missing 'awkward' or 'awkward0' field). schema is: {}".format(schema))

    prefix = schema.get("prefix", "")
    if seen is None:
        seen = ObjRef()

    if isinstance(whitelist, str):
        whitelist = [whitelist]
    elif len(whitelist) > 0 and isinstance(whitelist[0], str):
        whitelist = [whitelist]

    def unfill(schema):
        if isinstance(schema, dict):
            if "call" in schema and isinstance(schema["call"], list) and len(schema["call"]) > 0:
                gen = spec2function(schema["call"], whitelist=whitelist)
                args = [unfill(x) for x in schema.get("args", [])]

                kwargs = {}
                if schema.get("cacheable", False):
                    kwargs["cache"] = cache
                if schema.get("whitelistable", False):
                    kwargs["whitelist"] = whitelist
                if "kwargs" in schema:
                    kwargs.update({n: unfill(x) for n, x in schema["kwargs"].items()})

                out = gen(*args, **kwargs)

            elif "read" in schema:
                if schema.get("absolute", False):
                    out = storage[schema["read"]]
                else:
                    out = storage[prefix + schema["read"]]

            elif "list" in schema:
                out = [unfill(x) for x in schema["list"]]

            elif "tuple" in schema:
                out = tuple(unfill(x) for x in schema["tuple"])

            elif "dict" in schema:
                out = {n: unfill(x) for n, x in schema["dict"].items()}

            elif "pairs" in schema:
                out = [(n, unfill(x)) for n, x in schema["pairs"]]

            elif "dtype" in schema:
                out = json2dtype(schema["dtype"])

            elif "function" in schema:
                out = spec2function(schema["function"], whitelist=whitelist)

            elif "json" in schema:
                out = schema["json"]

            elif "python" in schema:
                out = topython(schema["python"])

            elif "ref" in schema:
                out = seen.obj(schema["ref"])

            else:
                raise ValueError("unrecognized JSON object with fields {0}".format(", ".join(repr(x) for x in schema)))

            if "id" in schema:
                seen[out] = schema["id"]
            return out

        elif isinstance(schema, list):
            raise ValueError("unrecognized JSON list with length {0}".format(len(schema)))

        else:
            raise ValueError("unrecognized JSON object: {0}".format(repr(schema)))

    return unfill(schema["schema"])

def keys(storage, name="", subschemas=True):
    schema = storage[name]
    if isinstance(schema, numpy.ndarray):
        schema = schema.tostring()
    if isinstance(schema, bytes):
        schema = schema.decode("ascii")
    schema = json.loads(schema)

    prefix = schema.get("prefix", "")

    def recurse(schema):
        if isinstance(schema, dict):
            if "call" in schema and isinstance(schema["call"], list) and len(schema["call"]) > 0:
                for x in schema.get("args", []):
                    for y in recurse(x):
                        yield y
                for x in schema.get("kwargs", {}).values():
                    for y in recurse(x):
                        yield y
                for x in schema.get("*", []):
                    for y in recurse(x):
                        yield y
                for x in schema.get("**", {}).values():
                    for y in recurse(x):
                        yield y

            elif "read" in schema:
                if schema.get("absolute", False):
                    yield schema["read"]
                else:
                    yield prefix + schema["read"]

            elif "list" in schema:
                for x in schema["list"]:
                    for y in recurse(x):
                        yield y

            elif "tuple" in schema:
                for x in schema["tuple"]:
                    for y in recurse(x):
                        yield y

            elif "dict" in schema:
                for x in schema["dict"].values():
                    for y in recurse(x):
                        yield y

            elif "pairs" in schema:
                for n, x in schema["pairs"]:
                    for y in recurse(x):
                        yield y

            elif "dtype" in schema:
                pass

            elif "function" in schema:
                pass

            elif "json" in schema:
                pass

            elif "python" in schema:
                pass

            elif "ref" in schema:
                pass

    yield name
    for x in recurse(schema["schema"]):
        yield x

def save(file, array, name=None, mode="a", **options):
    if isinstance(array, dict):
        arrays = array
    else:
        arrays = {"": array}

    if name is not None:
        arrays = {name + n: x for n, x in arrays.items()}

    arraynames = list(arrays)
    for i in range(len(arraynames)):
        for j in range(i + 1, len(arraynames)):
            if arraynames[i].startswith(arraynames[j]) or arraynames[j].startswith(arraynames[i]):
                raise KeyError("cannot write both {0} and {1} to zipfile because one is a prefix of the other", repr(arraynames[i]), repr(arraynames[j]))

    if isinstance(file, getattr(os, "PathLike", ())):
        file = os.fspath(file)
    elif hasattr(file, "__fspath__"):
        file = file.__fspath__()
    elif file.__class__.__module__ == "pathlib":
        import pathlib
        if isinstance(file, pathlib.Path):
             file = str(file)

    if isinstance(file, str) and not file.endswith(".awkd"):
        file = file + ".awkd"

    alloptions = {"delimiter": "-", "suffix": ".raw", "schemasuffix": ".json", "compression": compression}
    alloptions.update(options)
    options = alloptions

    class Wrap(object):
        def __init__(self, f):
            self.f = f
        def __setitem__(self, where, what):
            if isinstance(what, numpy.ndarray):
                what = what.tostring()
            self.f.writestr(where, what, compress_type=zipfile.ZIP_STORED)

    with zipfile.ZipFile(file, mode=mode, compression=zipfile.ZIP_STORED) as f:
        namelist = f.namelist()
        for name in arraynames:
            if any(n.startswith(name) for n in namelist):
                raise KeyError("cannot add {0} to zipfile because the following already exist: {1}".format(repr(name), ", ".join(repr(n) for n in namelist if n.startswith(name))))

        wrapped = Wrap(f)
        for name, array in arrays.items():
            serialize(array, wrapped, name=name, **options)

def load(file, **options):
    f = Load(file, **options)
    if list(f) == [""]:
        out = f[""]
        f.close()
        return out
    else:
        return f

class Load(Mapping):
    def __init__(self, file, **options):
        class Wrap(object):
            def __init__(self):
                self.f = zipfile.ZipFile(file, mode="r")
            def __getitem__(self, where):
                return self.f.read(where)

        self._file = Wrap()

        alloptions = {"schemasuffix": ".json", "whitelist": whitelist, "cache": None}
        alloptions.update(options)
        self.schemasuffix = alloptions.pop("schemasuffix")
        self.options = alloptions

    def __getitem__(self, where):
        return deserialize(self._file, name=where + self.schemasuffix, whitelist=self.options["whitelist"], cache=self.options["cache"])

    def __iter__(self):
        for n in self._file.f.namelist():
            if n.endswith(".json"):
                yield n[:-5]

    def __len__(self):
        count = 0
        for n in self._file.f.namelist():
            if n.endswith(".json"):
                count += 1
        return count

    def __repr__(self):
        return "<awkward0.load ({0} members)>".format(len(self))

    def close(self):
        self._file.f.close()

    def __del__(self):
        self.close()

    def __enter__(self, *args, **kwds):
        return self

    def __exit__(self, *args, **kwds):
        self.close()

class hdf5(MutableMapping):
    def __init__(self, group, **options):
        alloptions = {"compression": compression, "whitelist": whitelist, "cache": None}
        alloptions.update(options)
        self.options = alloptions
        self.options["delimiter"] = "/"
        self.options["schemasuffix"] = "/schema.json"

        class Wrap(object):
            def __init__(self):
                self.g = group
            def __getitem__(self, where):
                return self.g[where][()]
            def __setitem__(self, where, what):
                self.g[where] = numpy.frombuffer(what, dtype=numpy.uint8)

        self._group = Wrap()

    def __getitem__(self, where):
        return deserialize(self._group, name=where + self.options["schemasuffix"], whitelist=self.options["whitelist"], cache=self.options["cache"])

    def __setitem__(self, where, what):
        options = dict(self.options)
        if "awkwardlib" in options:
            del options["awkwardlib"]
        if "whitelist" in options:
            del options["whitelist"]
        if "cache" in options:
            del options["cache"]
        self._group.g.create_group(where)
        serialize(what, self._group, name=where, **options)

    def __delitem__(self, where):
        for subname in keys(self._group, name=where + self.options["schemasuffix"]):
            del self._group.g[subname]
        del self._group.g[where]

    def __iter__(self):
        schemaname = self.options["schemasuffix"].split("/")[-1]
        for subname in self._group.g:
            if schemaname in self._group.g[subname]:
                yield subname

    def __len__(self):
        schemaname = self.options["schemasuffix"].split("/")[-1]
        count = 0
        for subname in self._group.g:
            if schemaname in self._group.g[subname]:
                count += 1
        return count

    def __repr__(self):
        return "<awkward0.hdf5 {0} ({1} members)>".format(repr(self._group.g.name), len(self))
