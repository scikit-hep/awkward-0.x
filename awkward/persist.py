#!/usr/bin/env python

# Copyright (c) 2018, DIANA-HEP
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import base64
import fnmatch
import importlib
import json
import numbers
import os
import pickle
import zipfile
import zlib
try:
    from collections.abc import Mapping, MutableMapping
except ImportError:
    from collections import Mapping, MutableMapping

import awkward.type
import awkward.util
import awkward.version

compression = [
    {"minsize": 8192, "types": [awkward.util.numpy.bool_, awkward.util.numpy.bool, awkward.util.numpy.integer], "contexts": "*", "pair": (zlib.compress, ("zlib", "decompress"))},
    ]

partner = {
    zlib.compress: ("zlib", "decompress"),
    }

whitelist = [["numpy", "frombuffer"],
             ["zlib", "decompress"],
             ["awkward", "*Array"],
             ["awkward", "Table"],
             ["awkward.persist", "*"],
             ["awkward.arrow", "ParquetFile", "fromjson"]]

def frompython(obj):
    return base64.b64encode(pickle.dumps(obj)).decode("ascii")

def topython(string):
    return pickle.loads(base64.b64decode(string.encode("ascii")))

def spec2function(obj, whitelist=whitelist):
    for white in whitelist:
        for n, p in zip(obj, white):
            if not fnmatch.fnmatchcase(n, p):
                break
        else:
            gen, genname = importlib.import_module(obj[0]), obj[1:]
            while len(genname) > 0:
                gen, genname = getattr(gen, genname[0]), genname[1:]
            break
    else:
        raise RuntimeError("callable not in whitelist; add it by passing a whitelist argument:\n\n    whitelist = awkward.persist.whitelist + [{0}]".format(obj))
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
    return awkward.util.numpy.dtype(recurse(obj))

def type2json(obj):
    if isinstance(obj, awkward.type.Type):
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
        if isinstance(obj, awkward.type.Type):
            if id(obj) in seen:
                for i, x in enumerate(labeled):
                    if obj is x:
                        return {"ref": "T{0}".format(i)}

            else:
                seen.add(id(obj))
                if isinstance(obj, awkward.type.ArrayType):
                    out = {"takes": takes(obj._takes), "to": recurse(obj._to)}

                elif isinstance(obj, awkward.type.TableType):
                    out = {"fields": [[n, recurse(x)] for n, x in obj._fields.items()]}

                elif isinstance(obj, awkward.type.UnionType):
                    out = {"possibilities": [recurse(x) for x in obj._possibilities]}

                elif isinstance(obj, awkward.type.OptionType):
                    out = {"type": recurse(obj._type)}

                for i, x in enumerate(labeled):
                    if obj is x:
                        return {"set": "T{0}".format(i), "as": out}
                else:
                    return out

        elif isinstance(obj, awkward.util.numpy.dtype):
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
            raise TypeError("only awkward.type.Type, numpy.dtype, and callables are types")

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
            placeholder = labels[obj["set"]] = awkward.type.Placeholder()
            placeholder.value = recurse(obj["as"])
            return placeholder

        elif "ref" in obj:
            return labels[obj["ref"]]

        elif "takes" in obj and "to" in obj:
            return awkward.type.ArrayType(takes(obj["takes"]), recurse(obj["to"]))

        elif "fields" in obj:
            out = awkward.type.TableType()
            for n, x in obj["fields"]:
                out[n] = recurse(x)
            return out

        elif "possibilities" in obj:
            return awkward.type.UnionType(*[recurse(x) for x in obj["possibilities"]])

        elif "type" in obj:
            return awkward.type.OptionType(recurse(obj["type"]))

        elif "dtype" in obj:
            return json2dtype(obj["dtype"])

        elif "function" in obj:
            return spec2function(obj["function"], whitelist=whitelist)

        else:
            raise ValueError("unexpected set of keys in JSON: {0}".format(", ".join(repr(x) for x in obj)))

    return awkward.type._resolve(recurse(obj), {})

def jsonable(obj):
    if obj is None:
        return obj

    elif isinstance(obj, dict) and all(isinstance(n, str) for n in obj):
        return {n: jsonable(x) for n, x in obj.items()}

    elif isinstance(obj, list):
        return [jsonable(x) for x in obj]

    elif isinstance(obj, str):
        return str(obj)

    elif isinstance(obj, (bool, awkward.util.numpy.bool_, awkward.util.numpy.bool)):
        return bool(obj)      # policy: eliminate Numpy types

    elif isinstance(obj, (numbers.Integral, awkward.util.numpy.integer)):
        return int(obj)       # policy: eliminate Numpy types

    elif isinstance(obj, (numbers.Real, awkward.util.numpy.floating)) and awkward.util.numpy.isfinite(obj):
        return float(obj)     # policy: eliminate Numpy types

    else:
        raise TypeError("object cannot be losslessly serialized as JSON")

def serialize(obj, storage, name=None, delimiter="-", suffix=None, schemasuffix=None, compression=compression, **kwargs):
    import awkward.array.base
    import awkward.array.virtual

    for n in kwargs:
        if n not in ():
            raise TypeError("unrecognized serialization option: {0}".format(repr(n)))

    if name is None or name == "":
        name = ""
        prefix = ""
    elif delimiter is None:
        prefix = name
    else:
        prefix = name + delimiter

    if suffix is None:
        suffix = ""

    if schemasuffix is None:
        schemasuffix = ""

    if compression is None:
        compression = []
    if isinstance(compression, dict) or callable(compression) or (len(compression) == 2 and callable(compression[0])):
        compression = [compression]

    normalized = []
    for x in compression:
        if isinstance(x, dict):
            pass

        elif callable(x):
            if not x in partner:
                raise ValueError("decompression partner for {0} not known".format(x))
            x = {"pair": (x, partner[x])}

        elif len(x) == 2 and callable(x[0]):
            x = {"pair": x}

        minsize = x.get("minsize", 0)
        tpes = x.get("types", (object,))
        if not isinstance(tpes, tuple):
            try:
                tpes = tuple(tpes)
            except TypeError:
                tpes = (tpes,)
        contexts = x.get("contexts", "*")
        pair = x["pair"]

        normalized.append({"minsize": minsize, "types": tpes, "contexts": contexts, "pair": pair})

    seen = {}
    def fill(obj, context, prefix, suffix, schemasuffix, storage, compression, **kwargs):
        if id(obj) in seen:
            return {"ref": seen[id(obj)]}

        ident = len(seen)
        seen[id(obj)] = ident

        if type(obj) is awkward.util.numpy.dtype:
            return {"dtype": dtype2json(obj)}

        elif type(obj) is awkward.util.numpy.ndarray and len(obj.shape) != 0:
            if len(obj.shape) > 1:
                dtype = awkward.util.numpy.dtype((obj.dtype, obj.shape[1:]))
            else:
                dtype = obj.dtype

            for policy in normalized:
                minsize, tpes, contexts, pair = policy["minsize"], policy["types"], policy["contexts"], policy["pair"]
                if obj.nbytes >= minsize and issubclass(obj.dtype.type, tuple(tpes)) and any(fnmatch.fnmatchcase(context, p) for p in contexts):
                    compress, decompress = pair
                    storage[prefix + str(ident) + suffix] = compress(obj)

                    return {"id": ident,
                            "call": ["numpy", "frombuffer"],
                            "args": [{"call": decompress, "args": [{"read": str(ident) + suffix}]},
                                     {"dtype": dtype2json(dtype)},
                                     {"json": len(obj)}]}

            else:
                storage[prefix + str(ident) + suffix] = obj.tostring()
                return {"id": ident,
                        "call": ["numpy", "frombuffer"],
                        "args": [{"read": str(ident) + suffix},
                                 {"dtype": dtype2json(dtype)},
                                 {"json": len(obj)}]}

        elif hasattr(obj, "__awkward_persist__"):
            return obj.__awkward_persist__(ident, fill, prefix, suffix, schemasuffix, storage, compression, **kwargs)

        else:
            if hasattr(obj, "__module__") and (hasattr(obj, "__qualname__") or hasattr(obj, "__name__")) and obj.__module__ != "__main__":
                if hasattr(obj, "__qualname__"):
                    spec = [obj.__module__] + obj.__qualname__.split(".")
                else:
                    spec = [obj.__module__, obj.__name__]

                gen, genname = importlib.import_module(spec[0]), spec[1:]
                while len(genname) > 0:
                    gen, genname = getattr(gen, genname[0]), genname[1:]

                if gen is obj:
                    return {"id": ident, "function": spec}

            if hasattr(obj, "tojson") and hasattr(type(obj), "fromjson") and getattr(importlib.import_module(type(obj).__module__), type(obj).__name__) is type(obj):
                try:
                    return {"id": ident, "call": [type(obj).__module__, type(obj).__name__, "fromjson"], "args": [{"json": obj.tojson()}]}
                except:
                    pass

            try:
                obj = jsonable(obj)
            except TypeError:
                try:
                    return {"id": ident, "python": awkward.persist.frompython(obj)}

                except Exception as err:
                    raise TypeError("could not persist component as an array, awkward-array, importable function/class, JSON, or pickle; pickle error is\n\n    {0}: {1}".format(err.__class__.__name__, str(err)))
            else:
                return {"id": ident, "json": obj}

    schema = {"awkward": awkward.version.__version__,
              "schema": fill(obj, "", prefix, suffix, schemasuffix, storage, compression, **kwargs)}
    if prefix != "":
        schema["prefix"] = prefix

    storage[name + schemasuffix] = json.dumps(schema).encode("ascii")
    return schema

def deserialize(storage, name="", whitelist=whitelist, cache=None):
    import awkward.array.virtual

    schema = storage[name]
    if isinstance(schema, awkward.util.numpy.ndarray):
        schema = schema.tostring()
    if isinstance(schema, bytes):
        schema = schema.decode("ascii")
    schema = json.loads(schema)

    if "awkward" not in schema:
        raise ValueError("JSON object is not an awkward-array schema (missing 'awkward' field)")

    prefix = schema.get("prefix", "")
    seen = {}

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
                if schema["ref"] in seen:
                    out = seen[schema["ref"]]
                else:
                    out = awkward.array.virtual.VirtualArray(lambda: seen[schema["ref"]])

            else:
                raise ValueError("unrecognized JSON object with fields {0}".format(", ".join(repr(x) for x in schema)))

            if "id" in schema:
                seen[schema["id"]] = out
            return out

        elif isinstance(schema, list):
            raise ValueError("unrecognized JSON list with length {0}".format(len(schema)))

        else:
            raise ValueError("unrecognized JSON object: {0}".format(repr(schema)))

    return unfill(schema["schema"])

def keys(storage, name="", subschemas=True):
    schema = storage[name]
    if isinstance(schema, awkward.util.numpy.ndarray):
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
        return "<awkward.load ({0} members)>".format(len(self))

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
                return self.g[where].value
            def __setitem__(self, where, what):
                self.g[where] = awkward.util.numpy.frombuffer(what, dtype=awkward.util.numpy.uint8)

        self._group = Wrap()

    def __getitem__(self, where):
        return deserialize(self._group, name=where + self.options["schemasuffix"], whitelist=self.options["whitelist"], cache=self.options["cache"])

    def __setitem__(self, where, what):
        options = dict(self.options)
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
        return "<awkward.hdf5 {0} ({1} members)>".format(repr(self._group.g.name), len(self))
