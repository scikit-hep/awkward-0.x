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

import fnmatch
import importlib
import json
import numbers
import os
import zipfile
import zlib
try:
    from collections.abc import Mapping
except ImportError:
    from collections import Mapping

import numpy

import awkward.type
import awkward.util
import awkward.version

compression = [
    {"minsize": 8192, "types": [numpy.bool_, numpy.bool, numpy.integer], "contexts": "*", "pair": (zlib.compress, ("zlib", "decompress"))},
    ]

partner = {
    zlib.compress: ("zlib", "decompress"),
    }

whitelist = [["numpy", "frombuffer"],
             ["zlib", "decompress"],
             ["awkward", "*Array"],
             ["awkward", "Table"],
             ["awkward.persist", "*"]]

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
    return numpy.dtype(recurse(obj))

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

def serialize(obj, storage, name=None, delimiter="-", suffix=None, schemasuffix=None, compression=compression, **kwargs):
    import awkward.array.base

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
        types = x.get("types", (object,))
        if not isinstance(types, tuple):
            types = (types,)
        contexts = x.get("contexts", "*")
        pair = x["pair"]

        normalized.append({"minsize": minsize, "types": types, "contexts": contexts, "pair": pair})

    seen = {}
    def fill(obj, context, **kwargs):
        if id(obj) in seen:
            return {"ref": seen[id(obj)]}

        ident = len(seen)
        seen[id(obj)] = ident

        if type(obj) is numpy.ndarray and len(obj.shape) != 0:
            if len(obj.shape) > 1:
                dtype = dtype2json(numpy.dtype((obj.dtype, obj.shape[1:])))
            else:
                dtype = dtype2json(obj.dtype)

            for policy in normalized:
                minsize, types, contexts, pair = policy["minsize"], policy["types"], policy["contexts"], policy["pair"]
                if obj.nbytes >= minsize and issubclass(obj.dtype.type, tuple(types)) and any(fnmatch.fnmatchcase(context, p) for p in contexts):
                    compress, decompress = pair
                    storage[prefix + str(ident) + suffix] = compress(obj)

                    return {"id": ident,
                            "call": ["numpy", "frombuffer"],
                            "args": [{"call": decompress, "args": [{"read": str(ident) + suffix}]},
                                     {"call": ["awkward.persist", "json2dtype"], "args": [dtype]},
                                     len(obj)]}

            else:
                storage[prefix + str(ident) + suffix] = obj.tostring()
                return {"id": ident,
                        "call": ["numpy", "frombuffer"],
                        "args": [{"read": str(ident) + suffix},
                                 {"call": ["awkward.persist", "json2dtype"], "args": [dtype]},
                                 len(obj)]}

        elif hasattr(obj, "__awkward_persist__"):
            return obj.__awkward_persist__(ident, fill, **kwargs)

        else:
            raise TypeError("cannot serialize {0} instance (has no __awkward_persist__ method)".format(type(obj)))

    schema = {"awkward": awkward.version.__version__,
              "schema": fill(obj, "", **kwargs)}
    if prefix != "":
        schema["prefix"] = prefix

    storage[name + schemasuffix] = json.dumps(schema).encode("ascii")
    return schema

def deserialize(storage, name="", whitelist=whitelist, cache=None):
    import awkward.array.virtual

    schema = storage[name]
    if isinstance(schema, numpy.ndarray):
        schema = schema.tostring()
    if isinstance(schema, bytes):
        schema = schema.decode("ascii")
    schema = json.loads(schema)

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

                if "*" in schema:
                    args = args + [unfill(x) for x in schema["*"]]
                if "**" in schema:
                    kwargs.update({n: unfill(x) for n, x in schema["**"].items()})

                out = gen(*args, **kwargs)
                if "id" in schema:
                    seen[schema["id"]] = out
                return out

            elif "read" in schema:
                if schema.get("absolute", False):
                    return storage[schema["read"]]
                else:
                    return storage[prefix + schema["read"]]

            elif "list" in schema:
                return [unfill(x) for x in schema["list"]]

            elif "tuple" in schema:
                return tuple(unfill(x) for x in schema["tuple"])

            elif "pairs" in schema:
                return [(n, unfill(x)) for n, x in schema["pairs"]]

            elif "function" in schema:
                return spec2function(schema["function"], whitelist=whitelist)

            elif "ref" in schema:
                if schema["ref"] in seen:
                    return seen[schema["ref"]]
                else:
                    return awkward.array.virtual.VirtualArray(lambda: seen[schema["ref"]])
                       
            else:
                return schema

        else:
            return schema

    return unfill(schema["schema"])

def save(file, mode="a", options=None, **arrays):
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

    if isinstance(file, str) and not file.endswith(".akd"):
        file = file + ".akd"

    alloptions = {"delimiter": "-", "suffix": ".raw", "schemasuffix": ".json", "compression": compression}
    if options is not None:
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

class load(Mapping):
    def __init__(self, file, options=None):
        class Wrap(object):
            def __init__(self):
                self.f = zipfile.ZipFile(file, mode="r")
            def __getitem__(self, where):
                return self.f.read(where)
            def close(self):
                self.f.close()

        self._file = Wrap()

        alloptions = {"schemasuffix": ".json", "whitelist": whitelist, "cache": None}
        if options is not None:
            alloptions.update(options)
        self.schemasuffix = alloptions.pop("schemasuffix")
        self.options = alloptions
        
    def __getitem__(self, where):
        return deserialize(self._file, name=where + self.schemasuffix, **self.options)

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

    def __del__(self):
        self._file.close()

def tohdf5(group, options=None, **arrays):
    alloptions = {"compression": compression}
    if options is not None:
        alloptions.update(options)
    options = alloptions
    options["delimiter"] = "/"
    options["schemasuffix"] = "/schema.json"

    class Wrap(object):
        def __init__(self):
            self.g = group
        def __setitem__(self, where, what):
            self.g[where] = numpy.frombuffer(what, dtype=numpy.uint8)

    f = Wrap()
    for name, array in arrays.items():
        group.create_group(name)
        serialize(array, f, name=name, **options)

class fromhdf5(Mapping):
    def __init__(self, group, options=None):
        class Wrap(object):
            def __init__(self):
                self.g = group
            def __getitem__(self, where):
                return self.g[where].value

        self._group = Wrap()

        alloptions = {"whitelist": whitelist, "cache": None}
        if options is not None:
            alloptions.update(options)
        self.options = alloptions

    def __getitem__(self, where):
        return deserialize(self._group, name=where + "/schema.json", **self.options)

    def __iter__(self):
        for subname in self._group.g:
            if "schema.json" in self._group.g[subname]:
                yield subname

    def __len__(self):
        count = 0
        for subname in self._group.g:
            if "schema.json" in self._group.g[subname]:
                count += 0
        return count

    def __repr__(self):
        return "<HDF5 group {0} of awkward arrays>".format(repr(self._group.g.name))
